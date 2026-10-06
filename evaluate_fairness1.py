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
output_pdf = os.path.join(base_path, "fairness_report.pdf")
output_csv = os.path.join(base_path, "fairness_metrics_summary.csv")

# === Load Data ===
X_test = pd.read_csv(os.path.join(base_path, "X_test_scaled.csv"))
y_test = pd.read_csv(os.path.join(base_path, "y_test.csv"))['status']
sensitive_df = pd.read_csv(os.path.join(base_path, "covid_clinical_preprocessed.csv"))
sensitive_attributes = sensitive_df.loc[X_test.index][['Gender', 'Nationality']]

# === Load Model & Predict ===
ensemble = joblib.load(model_path)
y_pred = ensemble.predict(X_test)

# === Standard Metrics ===
print("Performance Metrics")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1-score: {f1_score(y_test, y_pred):.4f}")
print(f"AUC: {roc_auc_score(y_test, y_pred):.4f}")

# === Fairness Evaluation ===
all_metrics = []

for attr in ['Gender', 'Nationality']:
    print(f"\nFairness Metrics by '{attr}'")
    s_attr = sensitive_attributes[attr]
    metric_frame = MetricFrame(
        metrics={
            "accuracy": accuracy_score,
            "selection_rate": selection_rate,
            "TPR": true_positive_rate,
            "FPR": false_positive_rate,
        },
        y_true=y_test,
        y_pred=y_pred,
        sensitive_features=s_attr
    )

    print(metric_frame.by_group)

    disparity_accuracy = metric_frame.difference(method="between_groups")
    dpd = demographic_parity_difference(y_test, y_pred, sensitive_features=s_attr)
    eod = equalized_odds_difference(y_test, y_pred, sensitive_features=s_attr)

    print(f"Disparity in accuracy: {disparity_accuracy['accuracy']:.4f}")
    print(f"Demographic Parity Difference: {dpd:.4f}")
    print(f"Equalized Odds Difference: {eod:.4f}")

    # Save metrics for summary
    summary_row = {
        "Attribute": attr,
        "Disparity_Accuracy": disparity_accuracy,
        "Demographic_Parity_Diff": dpd,
        "Equalized_Odds_Diff": eod
    }
    all_metrics.append(summary_row)

    # Plot and save
    plot_path = os.path.join(base_path, f"fairness_plot_{attr}.png")
    metric_frame.by_group.plot(kind="bar", figsize=(10, 6), title=f"Fairness Metrics by {attr}")
    plt.ylabel("Score")
    plt.tight_layout()
    plt.savefig(plot_path)
    print(f"Saved fairness plot to {plot_path}")
    plt.close()

# === Save Summary Files ===
df_summary = pd.DataFrame(all_metrics)
df_summary.to_csv(output_csv, index=False)
print(f"\n Saved fairness summary CSV to: {output_csv}")

# Generate PDF with images
from matplotlib.backends.backend_pdf import PdfPages
with PdfPages(output_pdf) as pdf:
    for attr in ['Gender', 'Nationality']:
        img_path = os.path.join(base_path, f"fairness_plot_{attr}.png")
        img = plt.imread(img_path)
        plt.figure(figsize=(8, 6))
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Fairness Metrics: {attr}")
        pdf.savefig()
        plt.close()

print(f"Saved fairness PDF report to: {output_pdf}")
