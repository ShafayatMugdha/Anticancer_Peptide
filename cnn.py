# =========================
# ✅ PART 1: Shared Preprocessing
# =========================
import os
import glob
import numpy as np
import pandas as pd
from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score

import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# === Load Raw PSSM Matrix ===
pssm_dir = r"E:\\Anticancer\\code\\dataset\\pssm_output\\acp740"
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

print(f"\n✅ Loaded {X_seq.shape[0]} samples | Shape: {X_seq.shape}")
print(f"✅ Class distribution: {Counter(y_seq)}")

# === Train-test split ===
X_train, X_val, y_train, y_val = train_test_split(X_seq, y_seq, test_size=0.2, stratify=y_seq, random_state=42)

# === Class weights ===
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}
print("✅ Class Weights:", class_weight_dict)

# === Callbacks ===
early_stop = EarlyStopping(patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(patience=5, factor=0.5)


# =========================
# ✅ PART 2: 1D-CNN Model
# =========================
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization

model_cnn1d = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(30, 20), padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Conv1D(128, 3, activation='relu', padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.4),

    GlobalMaxPooling1D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_cnn1d.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_cnn1d.summary()

history_cnn1d = model_cnn1d.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


# =========================
# ✅ PART 3: BiLSTM Model
# =========================
from tensorflow.keras.layers import Bidirectional, LSTM

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
model_bilstm.summary()

history_bilstm = model_bilstm.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


# =========================
# ✅ PART 4: CNN + BiLSTM (Hybrid Model)
# =========================
model_hybrid = Sequential([
    Conv1D(64, 3, activation='relu', input_shape=(30, 20), padding='same'),
    BatchNormalization(),
    MaxPooling1D(pool_size=2),
    Dropout(0.3),

    Bidirectional(LSTM(64)),
    Dropout(0.4),

    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model_hybrid.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_hybrid.summary()

history_hybrid = model_hybrid.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=50,
    batch_size=32,
    class_weight=class_weight_dict,
    callbacks=[early_stop, reduce_lr],
    verbose=1
)


# =========================
# ✅ PART 5: Evaluation for All Models
# =========================
def evaluate_dl_model(model, name):
    y_pred_prob = model.predict(X_val).flatten()
    y_pred = (y_pred_prob >= 0.5).astype(int)

    acc = round(accuracy_score(y_val, y_pred) * 100, 2)
    prec = round(precision_score(y_val, y_pred) * 100, 2)
    rec = round(recall_score(y_val, y_pred) * 100, 2)
    f1 = round(f1_score(y_val, y_pred) * 100, 2)
    mcc = round(matthews_corrcoef(y_val, y_pred) * 100, 2)
    auc_score = round(roc_auc_score(y_val, y_pred_prob), 5)

    print(f"\n✅ {name} Evaluation:")
    print(f"Accuracy: {acc}% | Precision: {prec}% | Recall: {rec}% | F1: {f1}% | MCC: {mcc} | AUC: {auc_score}")

    return {
        'Model': name,
        'Accuracy': acc,
        'Precision': prec,
        'Recall': rec,
        'F1 Score': f1,
        'MCC': mcc,
        'AUC': auc_score
    }

# Collect all results
deep_results = []
deep_results.append(evaluate_dl_model(model_cnn1d, "1D-CNN"))
deep_results.append(evaluate_dl_model(model_bilstm, "BiLSTM"))
deep_results.append(evaluate_dl_model(model_hybrid, "CNN+BiLSTM"))

# Convert to DataFrame
results_df = pd.DataFrame(deep_results)
results_df.to_csv("deep_learning_models_comparison.csv", index=False)
print("\n✅ All deep learning results saved to 'deep_learning_models_comparison.csv'")
