import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import joblib

# === Paths ===
base_path = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\data\preprocessed"
model_path = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\models\classifiers\ensemble_model.pkl"

# === Load Data ===
X_train = pd.read_csv(os.path.join(base_path, "X_train_scaled.csv"))
y_train = pd.read_csv(os.path.join(base_path, "y_train.csv"))['status']
X_test = pd.read_csv(os.path.join(base_path, "X_test_scaled.csv"))
y_test = pd.read_csv(os.path.join(base_path, "y_test.csv"))['status']

# === Define Base Classifiers ===
clf1 = LogisticRegression(max_iter=1000)
clf2 = RandomForestClassifier(n_estimators=100, random_state=42)
clf3 = GradientBoostingClassifier(n_estimators=100, random_state=42)

# === Ensemble Model ===
ensemble = VotingClassifier(estimators=[
    ('lr', clf1),
    ('rf', clf2),
    ('gb', clf3)
], voting='soft')

# === Train ===
ensemble.fit(X_train, y_train)

# === Evaluate ===
y_pred = ensemble.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred)

print(f"Ensemble Accuracy: {acc:.4f}")
print(f"Ensemble F1-score: {f1:.4f}")
print(f"Ensemble AUC: {auc:.4f}")

# === Save Model ===
joblib.dump(ensemble, model_path)
print(f"Ensemble model saved to {model_path}")
