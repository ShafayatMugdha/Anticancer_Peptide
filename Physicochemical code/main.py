import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, roc_auc_score, roc_curve
)

# === Load Dataset ===
df = pd.read_csv("physicochemical_combined_clean.csv")
X = df.drop(columns=[col for col in ["ID", "label"] if col in df.columns])
y = df["label"].values

# === Remove all-zero columns and normalize ===
X = X.loc[:, (X != 0).any(axis=0)]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === SVM-RFE for Top 150 Features ===
print("🔍 Selecting top 150 features using SVM-RFE...")
svc = SVC(kernel='linear')
rfe = RFE(estimator=svc, n_features_to_select=150, step=0.1)
X_selected = rfe.fit_transform(X_scaled, y)
print(f"✅ Feature selection complete: {X_selected.shape[1]} features retained.\n")

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

# === Stratified Cross-Validation ===
skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results = []
plt.figure(figsize=(10, 7))
mean_fpr = np.linspace(0, 1, 100)

# === Evaluate Individual Models ===
for name, model in classifiers:
    tprs = []
    metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'MCC': [], 'AUC': []}

    for train_idx, test_idx in skf.split(X_selected, y):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.decision_function(X_test)

        metrics['Accuracy'].append(accuracy_score(y_test, y_pred) * 100)
        metrics['Precision'].append(precision_score(y_test, y_pred) * 100)
        metrics['Recall'].append(recall_score(y_test, y_pred) * 100)
        metrics['F1'].append(f1_score(y_test, y_pred) * 100)
        metrics['MCC'].append(matthews_corrcoef(y_test, y_pred) * 100)
        roc_auc = roc_auc_score(y_test, y_prob)
        metrics['AUC'].append(roc_auc)

        fpr, tpr, _ = roc_curve(y_test, y_prob)

        # Add (0,0) if missing
        if fpr[0] != 0 or tpr[0] != 0:
            fpr = np.insert(fpr, 0, 0)
            tpr = np.insert(tpr, 0, 0)

        # Add (1,1) if missing
        if fpr[-1] != 1 or tpr[-1] != 1:
            fpr = np.append(fpr, 1)
            tpr = np.append(tpr, 1)

        tprs.append(np.interp(mean_fpr, fpr, tpr))

    avg_tpr = np.mean(tprs, axis=0)
    # Enforce exact start and end points
    avg_tpr[0] = 0.0
    avg_tpr[-1] = 1.0
    avg_auc = np.mean(metrics['AUC'])
    plt.plot(mean_fpr, avg_tpr, label=f"{name} (AUC={avg_auc:.2f})")

    results.append({
        "Model": name,
        "Accuracy": np.mean(metrics["Accuracy"]),
        "Precision": np.mean(metrics["Precision"]),
        "Recall": np.mean(metrics["Recall"]),
        "F1 Score": np.mean(metrics["F1"]),
        "MCC": np.mean(metrics["MCC"]),
        "AUC": avg_auc
    })

# === Evaluate Ensemble Model ===
ensemble = VotingClassifier(estimators=classifiers, voting='soft', n_jobs=-1)
tprs = []
metrics = {'Accuracy': [], 'Precision': [], 'Recall': [], 'F1': [], 'MCC': [], 'AUC': []}

for train_idx, test_idx in skf.split(X_selected, y):
    X_train, X_test = X_selected[train_idx], X_selected[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    ensemble.fit(X_train, y_train)
    y_pred = ensemble.predict(X_test)
    y_prob = ensemble.predict_proba(X_test)[:, 1]

    metrics['Accuracy'].append(accuracy_score(y_test, y_pred) * 100)
    metrics['Precision'].append(precision_score(y_test, y_pred) * 100)
    metrics['Recall'].append(recall_score(y_test, y_pred) * 100)
    metrics['F1'].append(f1_score(y_test, y_pred) * 100)
    metrics['MCC'].append(matthews_corrcoef(y_test, y_pred) * 100)
    roc_auc = roc_auc_score(y_test, y_prob)
    metrics['AUC'].append(roc_auc)

    fpr, tpr, _ = roc_curve(y_test, y_prob)

    # Add (0,0) if missing
    if fpr[0] != 0 or tpr[0] != 0:
        fpr = np.insert(fpr, 0, 0)
        tpr = np.insert(tpr, 0, 0)

    # Add (1,1) if missing
    if fpr[-1] != 1 or tpr[-1] != 1:
        fpr = np.append(fpr, 1)
        tpr = np.append(tpr, 1)

    tprs.append(np.interp(mean_fpr, fpr, tpr))

avg_tpr = np.mean(tprs, axis=0)
avg_tpr[0] = 0.0
avg_tpr[-1] = 1.0
avg_auc = np.mean(metrics['AUC'])
plt.plot(mean_fpr, avg_tpr, label=f"Ensemble (AUC={avg_auc:.2f})")

results.append({
    "Model": "Ensemble",
    "Accuracy": np.mean(metrics["Accuracy"]),
    "Precision": np.mean(metrics["Precision"]),
    "Recall": np.mean(metrics["Recall"]),
    "F1 Score": np.mean(metrics["F1"]),
    "MCC": np.mean(metrics["MCC"]),
    "AUC": avg_auc
})

# === Final ROC Plot ===
plt.plot([0, 1], [0, 1], 'k--')
plt.title("ROC Curves - SVM-RFE + Stratified K-Fold (Top 150 Features)")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("roc_curve_svm_rfe_kfold_physicochemical_150features.png", dpi=300)
plt.show()

# === Save Results ===
results_df = pd.DataFrame(results).sort_values(by="AUC", ascending=False)

print("\n📊 Final Cross-Validated Model Performance:\n")
print(results_df)

results_df.to_csv("final_cv_model_performance_150features.csv", index=False)
print("✅ Saved: 'roc_curve_svm_rfe_kfold_physicochemical_150features.png' and 'final_cv_model_performance_150features.csv'")
