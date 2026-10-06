import os
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

from fairlearn.metrics import (
    MetricFrame,
    selection_rate,
    true_positive_rate,
    false_positive_rate,
    demographic_parity_difference,
    equalized_odds_difference
)
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# === Paths ===
base_path = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\data\preprocessed"
model_path = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\models\classifiers\ensemble_model.pkl"

# === Load Data ===
X_test = pd.read_csv(os.path.join(base_path, "X_test_scaled.csv"))
y_test = pd.read_csv(os.path.join(base_path, "y_test.csv"))['status']
sensitive_df = pd.read_csv(os.path.join(base_path, "covid_clinical_preprocessed.csv"))  # includes Gender and Nationality

# Align indices (assuming same row order)
sensitive_attributes = sensitive_df.loc[X_test.index][['Gender', 'Nationality']]

# === Load Ensemble Model ===
ensemble = joblib.load(model_path)
y_pred = ensemble.predict(X_test)

# === Standard Performance Metrics ===
print("Performance Metrics")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred):.4f}")

# === Fairness Evaluation ===
for attr in ['Gender', 'Nationality']:
    print(f"\nFairness Metrics by '{attr}'")

    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "TPR": true_positive_rate,
            "FPR": false_positive_rate,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=sensitive_attributes[attr]
    )

    print(metric_frame.by_group)

    # Display disparity in accuracy
    accuracy_diff = metric_frame.difference(method='between_groups')['accuracy']
    print(f"Disparity in accuracy: {accuracy_diff:.4f}")

    # Group fairness metrics
    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=sensitive_attributes[attr])
    eod = equalized_odds_difference(y_test, y_pred, sensitive_features=sensitive_attributes[attr])

    print(f"Demographic Parity Difference: {dpd:.4f}")
    print(f"Equalized Odds Difference: {eod:.4f}")

    # === Plot Metrics ===
    metric_frame.by_group.plot(kind='bar', figsize=(10, 6))
    plt.title(f"Fairness Metrics by '{attr}'")
    plt.ylabel("Metric Value")
    plt.xticks(rotation=0)
    plt.grid(True, linestyle='--', alpha=0.6)
    plot_path = os.path.join(base_path, f"fairness_plot_{attr}.png")
    plt.savefig(plot_path)
    print(f"Saved fairness plot to {plot_path}")
    plt.close()
