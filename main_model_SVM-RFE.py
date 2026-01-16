'''
train set
'''

import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import collections

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve, auc
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.feature_selection import RFE

# === Load PSSM Features ===
pssm_dir = r"E:\Anticancer\code\dataset\pssm_output\acp740"
pssm_expected_rows = 30

label_dict = {f"seq_{i}": 1 if i < 376 else 0 for i in range(740)}
X, y = [], []

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
        X.append(np.array(matrix).flatten())
        y.append(label_dict[name])

X = np.array(X)
y = np.array(y)

print(f"Loaded {X.shape[0]} samples with {X.shape[1]} features")
print(f"Class distribution: {collections.Counter(y)}")

# === Normalize using StandardScaler ===
scaler = StandardScaler()
X_normalized = scaler.fit_transform(X)

# === Feature Selection (SVM-RFE) ===
svm_selector = SVC(kernel='linear')
rfe = RFE(svm_selector, n_features_to_select=200)
X_selected = rfe.fit_transform(X_normalized, y)
print(f"\n✅ SVM-RFE applied: {X_selected.shape[1]} features selected from {X.shape[1]}")

# === Classifiers ===
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

skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
results_summary = []
plt.figure(figsize=(10, 8))

# === Updated evaluation function ===
def evaluate_model(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    if hasattr(model, 'predict_proba'):
        predictions_prob = model.predict_proba(X_test)[:, 1]
        predictions = (predictions_prob >= 0.5).astype(int)
    else:
        predictions = model.predict(X_test)
        predictions_prob = predictions

    accuracy = round(accuracy_score(y_test, predictions) * 100, 2)
    precision = round(precision_score(y_test, predictions) * 100, 2)
    recall = round(recall_score(y_test, predictions) * 100, 2)
    f1 = round(f1_score(y_test, predictions) * 100, 2)
    mcc = round(matthews_corrcoef(y_test, predictions) * 100, 2)

    fpr, tpr, _ = roc_curve(y_test, predictions_prob)

    # Fix ROC curve to start at (0,0) and end at (1,1)
    fpr = np.insert(fpr, 0, 0.0)
    tpr = np.insert(tpr, 0, 0.0)
    fpr = np.append(fpr, 1.0)
    tpr = np.append(tpr, 1.0)

    roc_auc = auc(fpr, tpr)

    return fpr, tpr, roc_auc, accuracy, precision, recall, f1, mcc

# === Train & Evaluate each classifier with ROC plotting ===
for classifier_name, classifier in classifiers:
    mean_tpr = 0.0
    mean_fpr = np.linspace(0, 1, 100)
    metrics = {'Classifier': classifier_name, 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'MCC': [], 'AUC': []}

    for fold, (train_index, test_index) in enumerate(skf.split(X_selected, y), 1):
        X_train_fold, X_test_fold = X_selected[train_index], X_selected[test_index]
        y_train_fold, y_test_fold = y[train_index], y[test_index]

        fpr, tpr, roc_auc, accuracy, precision, recall, f1, mcc = evaluate_model(
            classifier, X_train_fold, X_test_fold, y_train_fold, y_test_fold
        )

        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0  # ensure tpr starts at 0
        mean_tpr += interp_tpr

        metrics['Accuracy'].append(accuracy)
        metrics['Precision'].append(precision)
        metrics['Recall'].append(recall)
        metrics['F1 Score'].append(f1)
        metrics['MCC'].append(mcc)
        metrics['AUC'].append(roc_auc)

    mean_tpr /= skf.get_n_splits()
    plt.plot(mean_fpr, mean_tpr, lw=2, label=f'{classifier_name} (Mean AUC = {np.mean(metrics["AUC"]):.2f})')

    results_summary.append({
        'Classifier': classifier_name,
        'Accuracy': np.mean(metrics['Accuracy']),
        'Precision': np.mean(metrics['Precision']),
        'Recall': np.mean(metrics['Recall']),
        'F1 Score': np.mean(metrics['F1 Score']),
        'MCC': np.mean(metrics['MCC']),
        'AUC': np.mean(metrics['AUC'])
    })

# === Ensemble ROC Evaluation ===
mean_tpr = 0.0
mean_fpr = np.linspace(0, 1, 100)
metrics = {'Classifier': 'Ensemble', 'Accuracy': [], 'Precision': [], 'Recall': [], 'F1 Score': [], 'MCC': [], 'AUC': []}

for fold, (train_index, test_index) in enumerate(skf.split(X_selected, y), 1):
    X_train_fold, X_test_fold = X_selected[train_index], X_selected[test_index]
    y_train_fold, y_test_fold = y[train_index], y[test_index]

    fpr, tpr, roc_auc, accuracy, precision, recall, f1, mcc = evaluate_model(
        voting_clf, X_train_fold, X_test_fold, y_train_fold, y_test_fold
    )

    interp_tpr = np.interp(mean_fpr, fpr, tpr)
    interp_tpr[0] = 0.0
    mean_tpr += interp_tpr

    metrics['Accuracy'].append(accuracy)
    metrics['Precision'].append(precision)
    metrics['Recall'].append(recall)
    metrics['F1 Score'].append(f1)
    metrics['MCC'].append(mcc)
    metrics['AUC'].append(roc_auc)

mean_tpr /= skf.get_n_splits()
plt.plot(mean_fpr, mean_tpr, lw=2, label=f'Ensemble (Mean AUC = {np.mean(metrics["AUC"]):.2f})')

results_summary.append({
    'Classifier': 'Ensemble',
    'Accuracy': np.mean(metrics['Accuracy']),
    'Precision': np.mean(metrics['Precision']),
    'Recall': np.mean(metrics['Recall']),
    'F1 Score': np.mean(metrics['F1 Score']),
    'MCC': np.mean(metrics['MCC']),
    'AUC': np.mean(metrics['AUC'])
})

# === Final ROC Curve Plotting ===
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')  # diagonal
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate", fontsize=12)
plt.ylabel("True Positive Rate", fontsize=12)
plt.title("ROC Curves for Classifiers with SVM-RFE Feature Selection", fontsize=14)
plt.legend(loc="lower right")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("roc_curve_all_classifiers.png", dpi=300)
plt.show()

# === Save evaluation results ===
evaluation_df = pd.DataFrame(results_summary)
print("\nClassifier Performance Summary:")
print(evaluation_df.sort_values(by='AUC', ascending=False))

evaluation_df.to_csv("classifier_performance_summary.csv", index=False)
print("✅ Results saved as 'classifier_performance_summary.csv' and ROC curve as 'roc_curve_all_classifiers.png'")



'''
Test set 
'''

# import pandas as pd
# import numpy as np
# import glob
# import os

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_curve, auc
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier, VotingClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from xgboost import XGBClassifier
# from sklearn.feature_selection import RFE

# # === Load PSSM Features ===
# pssm_dir = r"E:\Anticancer\code\dataset\pssm_output\acp740"
# pssm_expected_rows = 30

# label_dict = {f"seq_{i}": 1 if i < 376 else 0 for i in range(740)}
# X, y = [], []

# for file in glob.glob(os.path.join(pssm_dir, "*.pssm")):
#     name = os.path.basename(file).replace(".pssm", "")
#     if name not in label_dict:
#         continue
#     with open(file) as f:
#         lines = f.readlines()
#     matrix = []
#     for line in lines:
#         parts = line.strip().split()
#         if len(parts) >= 22 and parts[0].isdigit():
#             try:
#                 matrix.append([float(x) for x in parts[2:22]])
#             except ValueError:
#                 matrix = []
#                 break
#     if len(matrix) == pssm_expected_rows:
#         X.append(np.array(matrix).flatten())
#         y.append(label_dict[name])

# X = np.array(X)
# y = np.array(y)

# # === Normalize ===
# scaler = MinMaxScaler()
# X_normalized = scaler.fit_transform(X)

# # === Feature Selection (SVM-RFE) ===
# svm_selector = SVC(kernel='linear')
# rfe = RFE(svm_selector, n_features_to_select=200)
# X_selected = rfe.fit_transform(X_normalized, y)
# print(f"\n✅ SVM-RFE applied: {X_selected.shape[1]} features selected from {X.shape[1]}")

# # === Train/Test Split ===
# X_train, X_val, y_train, y_val = train_test_split(X_selected, y, test_size=0.2, stratify=y, random_state=42)

# # === Define Base Classifiers ===
# base_classifiers = [
#     ('Logistic Regression', LogisticRegression(max_iter=1000)),
#     ('SVM', SVC(C=10, gamma=0.01, kernel='rbf', probability=True)),
#     ('Random Forest', RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=3, min_samples_leaf=1, bootstrap=False, random_state=42)),
#     ('KNN', KNeighborsClassifier(n_neighbors=5)),
#     ('Naive Bayes', GaussianNB()),
#     ('XGBoost', XGBClassifier(n_estimators=500, max_depth=8, learning_rate=0.03, subsample=0.9, colsample_bytree=0.9, min_child_weight=3, gamma=0.2, use_label_encoder=False, eval_metric='logloss', random_state=42))
# ]

# # === Add Ensemble (without self-recursion) ===
# classifiers = base_classifiers.copy()
# voting_clf = VotingClassifier(estimators=base_classifiers, voting='soft', n_jobs=-1)
# classifiers.append(('Ensemble', voting_clf))

# # === Evaluate each classifier ===
# results = []

# for name, model in classifiers:
#     model.fit(X_train, y_train)

#     if hasattr(model, 'predict_proba'):
#         probas = model.predict_proba(X_val)[:, 1]
#         preds = (probas >= 0.5).astype(int)
#     else:
#         preds = model.predict(X_val)
#         probas = preds  # fallback

#     acc = round(accuracy_score(y_val, preds) * 100, 2)
#     prec = round(precision_score(y_val, preds) * 100, 2)
#     rec = round(recall_score(y_val, preds) * 100, 2)
#     f1 = round(f1_score(y_val, preds) * 100, 2)
#     mcc = round(matthews_corrcoef(y_val, preds) * 100, 2)
#     fpr, tpr, _ = roc_curve(y_val, probas)
#     roc_auc = auc(fpr, tpr)

#     results.append({
#         'Model': name,
#         'Accuracy': acc,
#         'Precision': prec,
#         'Recall': rec,
#         'F1 Score': f1,
#         'MCC': mcc,
#         'AUC': round(roc_auc, 5)
#     })

# # === Show and Save Final Table ===
# final_df = pd.DataFrame(results)
# print("\n=== Final Validation Set Results for All Classifiers ===")
# print(final_df)

# # final_df.to_csv("final_validation_all_models.csv", index=False)
# print("\n✅ All model validation results saved to 'final_validation_all_models.csv'")
