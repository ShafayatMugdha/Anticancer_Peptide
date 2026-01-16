import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === Load Raw PSSM Matrix ===
pssm_dir = r"E:\Anticancer\code\dataset\pssm_output\acp740"
pssm_expected_rows = 30
label_dict = {f"seq_{i}": 1 if i < 376 else 0 for i in range(740)}

X_seq, y_seq = [], []
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
        X_seq.append(matrix)
        y_seq.append(label_dict[name])

X_seq = np.array(X_seq, dtype=np.float32)
y_seq = np.array(y_seq)

# === Train-test split ===
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42)

# === Class weights ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# === Build BiLSTM model ===
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
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[EarlyStopping(patience=10, restore_best_weights=True),
               ReduceLROnPlateau(patience=5, factor=0.5)],
    verbose=1
)

# === Evaluation ===
y_pred_prob = model_bilstm.predict(X_val).flatten()
y_pred = (y_pred_prob >= 0.5).astype(int)

acc = round(accuracy_score(y_val, y_pred) * 100, 2)
prec = round(precision_score(y_val, y_pred) * 100, 2)
rec = round(recall_score(y_val, y_pred) * 100, 2)
f1 = round(f1_score(y_val, y_pred) * 100, 2)
mcc = round(matthews_corrcoef(y_val, y_pred) * 100, 2)
auc_score = round(roc_auc_score(y_val, y_pred_prob), 5)

print("\n✅ BiLSTM Evaluation:")
print(f"Accuracy: {acc}% | Precision: {prec}% | Recall: {rec}% | F1: {f1}% | MCC: {mcc} | AUC: {auc_score}")

# === ROC Curve ===
fpr, tpr, _ = roc_curve(y_val, y_pred_prob)

# Add (0,0) if missing
if fpr[0] != 0 or tpr[0] != 0:
    fpr = np.insert(fpr, 0, 0)
    tpr = np.insert(tpr, 0, 0)

# Add (1,1) if missing
if fpr[-1] != 1 or tpr[-1] != 1:
    fpr = np.append(fpr, 1)
    tpr = np.append(tpr, 1)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f"BiLSTM (AUC = {auc_score:.4f})")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve: BiLSTM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("bilstm_roc_curve.png")
plt.show()

# === Precision-Recall Curve ===
prec_curve, rec_curve, _ = precision_recall_curve(y_val, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(rec_curve, prec_curve, label='BiLSTM')
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve: BiLSTM")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("bilstm_pr_curve.png")
plt.show()

# === Prediction Probability Histogram ===
plt.figure(figsize=(8, 6))
plt.hist(y_pred_prob[y_val == 1], bins=30, alpha=0.6, label='Positive Class')
plt.hist(y_pred_prob[y_val == 0], bins=30, alpha=0.6, label='Negative Class')
plt.xlabel("Predicted Probability")
plt.ylabel("Frequency")
plt.title("BiLSTM: Prediction Probabilities")
plt.legend()
plt.tight_layout()
plt.savefig("bilstm_prediction_histogram.png")
plt.show()

# === Save results ===
pd.DataFrame([{
    'Model': 'BiLSTM',
    'Accuracy': acc,
    'Precision': prec,
    'Recall': rec,
    'F1 Score': f1,
    'MCC': mcc,
    'AUC': auc_score
}]).to_csv("bilstm_final_results.csv", index=False)
print("\n✅ BiLSTM results saved to 'bilstm_final_results.csv'")
