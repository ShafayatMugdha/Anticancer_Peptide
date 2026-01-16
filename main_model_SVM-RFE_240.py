import pandas as pd
import numpy as np
import glob
import os
import collections
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_curve, auc
)
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE

# === Load PSSM Features ===
pssm_dir = r"E:\Anticancer\code\dataset\pssm_output\acp240"
pssm_expected_rows = 30
pssm_columns = 20
label_dict = {f"seq_{i}": 1 if i < 120 else 0 for i in range(240)}

X, y = [], []
pssm_files = glob.glob(os.path.join(pssm_dir, "*.pssm"))
print(f"🧪 Found {len(pssm_files)} .pssm files")

for file in pssm_files:
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

    if len(matrix) == 0:
        continue
    elif len(matrix) < pssm_expected_rows:
        pad_len = pssm_expected_rows - len(matrix)
        matrix.extend([[0.0] * pssm_columns] * pad_len)
    elif len(matrix) > pssm_expected_rows:
        matrix = matrix[:pssm_expected_rows]

    if len(matrix) != pssm_expected_rows:
        continue

    X.append(np.array(matrix).flatten())
    y.append(label_dict[name])

X = np.array(X)
y = np.array(y)
print(f"\n✅ Loaded {X.shape[0]} valid samples with {X.shape[1]} features")
print(f"Class distribution: {collections.Counter(y)}")

# === Normalize ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === SVM-RFE Feature Selection ===
print("\n🔍 Selecting top 200 features using SVM-RFE...")
rfe_selector = RFE(SVC(kernel='linear'), n_features_to_select=200)
X_selected = rfe_selector.fit_transform(X_scaled, y)
print(f"✅ Selected shape: {X_selected.shape}")

# === Define Classifiers ===
classifiers = [
    ('Random Forest', RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=3, min_samples_leaf=1, bootstrap=False, random_state=42)),
    ('Logistic Regression', LogisticRegression(max_iter=1000)),
    ('SVM', SVC(C=10, gamma=0.01, kernel='rbf', probability=True)),
    ('KNN', KNeighborsClassifier(n_neighbors=5)),
    ('Naive Bayes', GaussianNB()),
    ('XGBoost', XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, min_child_weight=3, gamma=0.2, use_label_encoder=False, eval_metric='logloss', random_state=42)),
    ('LightGBM', LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.1, min_child_samples=10, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=0.1, random_state=42))
]

voting_clf = VotingClassifier(estimators=classifiers, voting='soft', n_jobs=-1)

# === Cross-validation Setup ===
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results_summary = []
plt.figure(figsize=(10, 8))
mean_fpr = np.linspace(0, 1, 100)

# === Evaluation Helper ===
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
    else:
        y_pred = model.predict(X_test)
        y_prob = y_pred

    acc = accuracy_score(y_test, y_pred) * 100
    prec = precision_score(y_test, y_pred) * 100
    rec = recall_score(y_test, y_pred) * 100
    f1 = f1_score(y_test, y_pred) * 100
    mcc = matthews_corrcoef(y_test, y_pred) * 100
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fpr = np.concatenate([[0], fpr, [1]])
    tpr = np.concatenate([[0], tpr, [1]])
    auc_score = auc(fpr, tpr)
    return fpr, tpr, auc_score, acc, prec, rec, f1, mcc

# === Individual Classifier Evaluation ===
for name, model in classifiers:
    tprs = []
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'MCC': [], 'AUC': []}

    for train_idx, test_idx in skf.split(X_selected, y):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        fpr, tpr, auc_score, acc, prec, rec, f1, mcc = evaluate_model(model, X_train, X_test, y_train, y_test)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)

        metrics['Accuracy'].append(acc)
        metrics['Precision'].append(prec)
        metrics['Recall'].append(rec)
        metrics['F1'].append(f1)
        metrics['MCC'].append(mcc)
        metrics['AUC'].append(auc_score)

    mean_tpr = np.mean(tprs, axis=0)
    mean_auc = np.mean(metrics['AUC'])
    plt.plot(mean_fpr, mean_tpr, label=f"{name} (AUC = {mean_auc:.2f})")

    results_summary.append({
        "Classifier": name,
        "Accuracy": np.mean(metrics['Accuracy']),
        "Precision": np.mean(metrics['Precision']),
        "Recall": np.mean(metrics['Recall']),
        "F1 Score": np.mean(metrics['F1']),
        "MCC": np.mean(metrics['MCC']),
        "AUC": mean_auc
    })

# === Ensemble Evaluation ===
tprs = []
metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'MCC': [], 'AUC': []}

for train_idx, test_idx in skf.split(X_selected, y):
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    fpr, tpr, auc_score, acc, prec, rec, f1, mcc = evaluate_model(voting_clf, X_train, X_test, y_train, y_test)
    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)

    metrics['Accuracy'].append(acc)
    metrics['Precision'].append(prec)
    metrics['Recall'].append(rec)
    metrics['F1'].append(f1)
    metrics['MCC'].append(mcc)
    metrics['AUC'].append(auc_score)

mean_tpr = np.mean(tprs, axis=0)
mean_auc = np.mean(metrics['AUC'])
plt.plot(mean_fpr, mean_tpr, label=f"Ensemble (AUC = {mean_auc:.2f})")

results_summary.append({
    "Classifier": "Ensemble",
    "Accuracy": np.mean(metrics['Accuracy']),
    "Precision": np.mean(metrics['Precision']),
    "Recall": np.mean(metrics['Recall']),
    "F1 Score": np.mean(metrics['F1']),
    "MCC": np.mean(metrics['MCC']),
    "AUC": mean_auc
})

# === Final ROC Plot ===
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves (ACP240 - 10-fold CV)", fontsize=13)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc="lower right")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("roc_curve_all_classifiers_acp240.png", dpi=300)
plt.show()

# === Save & Show Final Results ===
evaluation_df = pd.DataFrame(results_summary).sort_values(by="AUC", ascending=False)
print("\n📊 Classifier Performance Summary (ACP240):\n")
print(evaluation_df)

evaluation_df.to_csv("classifier_performance_summary_acp240.csv", index=False)
print("✅ Results saved to 'classifier_performance_summary_acp240.csv' and ROC to 'roc_curve_all_classifiers_acp240.png'")
