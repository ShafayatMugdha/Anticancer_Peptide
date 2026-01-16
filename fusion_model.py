import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score,
    roc_curve, auc, precision_recall_curve
)

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === Load PSSM Features ===
pssm_dir = r"E:\Anticancer\code\dataset\pssm_output\acp740"
pssm_expected_rows = 30
label_dict = {f"seq_{i}": 1 if i < 376 else 0 for i in range(740)}

X_rfe, X_dl, y = [], [], []
for file in glob.glob(os.path.join(pssm_dir, "*.pssm")):
    name = os.path.basename(file).replace(".pssm", "")
    if name not in label_dict:
        continue
    with open(file) as f:
        lines = f.readlines()
    matrix = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 22 and parts[0].isdigit():
            try:
                matrix.append([float(x) for x in parts[2:22]])
            except ValueError:
                matrix = []
                break
    if len(matrix) == pssm_expected_rows:
        X_rfe.append(np.array(matrix).flatten())
        X_dl.append(matrix)
        y.append(label_dict[name])

X_rfe = np.array(X_rfe, dtype=np.float32)
X_dl = np.array(X_dl, dtype=np.float32)
y = np.array(y)

# === Normalize RFE features ===
scaler = MinMaxScaler()
X_rfe_scaled = scaler.fit_transform(X_rfe)

# === Apply SVM-RFE ===
rfe_selector = RFE(SVC(kernel='linear'), n_features_to_select=200)
X_rfe_selected = rfe_selector.fit_transform(X_rfe_scaled, y)

# === Train/Validation Split ===
X_rfe_train, X_rfe_val, X_dl_train, X_dl_val, y_train, y_val = train_test_split(
    X_rfe_selected, X_dl, y, test_size=0.2, stratify=y, random_state=42
)

# === Train BiLSTM ===
model_bilstm = Sequential([
    Bidirectional(LSTM(64, return_sequences=True), input_shape=(30, 20)),
    Dropout(0.3),
    Bidirectional(LSTM(64)),
    Dropout(0.4),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_bilstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_bilstm.fit(
    X_dl_train, y_train,
    validation_data=(X_dl_val, y_val),
    epochs=30,
    batch_size=32,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True),
               ReduceLROnPlateau(patience=3, factor=0.5)],
    verbose=1
)

# === Get BiLSTM output probabilities ===
prob_train_dl = model_bilstm.predict(X_dl_train).flatten().reshape(-1, 1)
prob_val_dl = model_bilstm.predict(X_dl_val).flatten().reshape(-1, 1)

# === Concatenate SVM-RFE features with BiLSTM predictions ===
X_train_fusion = np.concatenate([X_rfe_train, prob_train_dl], axis=1)
X_val_fusion = np.concatenate([X_rfe_val, prob_val_dl], axis=1)

# === Train meta-classifier ===
meta_clf = LogisticRegression(max_iter=1000)
meta_clf.fit(X_train_fusion, y_train)

# === Evaluate fusion model ===
y_pred_prob = meta_clf.predict_proba(X_val_fusion)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)

acc = round(accuracy_score(y_val, y_pred) * 100, 2)
prec = round(precision_score(y_val, y_pred) * 100, 2)
rec = round(recall_score(y_val, y_pred) * 100, 2)
f1 = round(f1_score(y_val, y_pred) * 100, 2)
mcc = round(matthews_corrcoef(y_val, y_pred) * 100, 2)
auc_score_val = round(roc_auc_score(y_val, y_pred_prob), 5)

print("\n✅ Fusion Model (SVM-RFE + BiLSTM Probabilities) Evaluation:")
print(f"Accuracy: {acc}% | Precision: {prec}% | Recall: {rec}% | F1: {f1}% | MCC: {mcc} | AUC: {auc_score_val}")

# === Updated ROC Curve (Aligned and Interpolated) ===
fpr, tpr, _ = roc_curve(y_val, y_pred_prob)

# Ensure (0,0) and (1,1) are included
fpr = np.insert(fpr, 0, 0.0)
tpr = np.insert(tpr, 0, 0.0)
fpr = np.append(fpr, 1.0)
tpr = np.append(tpr, 1.0)

# Interpolate for smooth curve
mean_fpr = np.linspace(0, 1, 100)
interp_tpr = np.interp(mean_fpr, fpr, tpr)
interp_tpr[0] = 0.0
interp_tpr[-1] = 1.0
roc_auc = auc(mean_fpr, interp_tpr)

# Plot ROC
plt.figure(figsize=(10, 8))
plt.plot(mean_fpr, interp_tpr, lw=2, color="darkorange", label=f"Fusion Model (AUC = {roc_auc:.4f})")
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle='--')
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curve: Fusion Model (SVM-RFE + BiLSTM)", fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("fusion_roc_curve_aligned.png", dpi=300)
plt.show()

# === Precision-Recall Curve ===
prec_curve, rec_curve, _ = precision_recall_curve(y_val, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(rec_curve, prec_curve, label='Fusion')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve: Fusion Model")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("fusion_pr_curve.png")
plt.show()

# === Prediction Probability Histogram ===
plt.figure(figsize=(8, 6))
plt.hist(y_pred_prob[y_val == 1], bins=30, alpha=0.6, label='Positive Class')
plt.hist(y_pred_prob[y_val == 0], bins=30, alpha=0.6, label='Negative Class')
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("Fusion: Prediction Probabilities")
plt.legend()
plt.tight_layout()
plt.savefig("fusion_prediction_histogram.png")
plt.show()

# === Save results ===
pd.DataFrame([{
    'Model': 'Fusion (RFE + BiLSTM)',
    'Accuracy': acc,
    'Precision': prec,
    'Recall': rec,
    'F1 Score': f1,
    'MCC': mcc,
    'AUC': auc_score_val
}]).to_csv("fusion_model_final_results.csv", index=False)

print("\n✅ Fusion results saved to 'fusion_model_final_results.csv' and ROC saved to 'fusion_roc_curve_aligned.png'")
