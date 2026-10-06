import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Paths
input_file = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\data\preprocessed\covid_clinical_balanced.csv"
output_folder = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\data\preprocessed"

# Load dataset
df = pd.read_csv(input_file)

# Drop non-feature columns
X = df.drop(columns=['patient_id', 'status', 'Nationality', 'Gender'])
y = df['status'].astype(int)  # already 0/1, just enforce int type

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Save outputs
pd.DataFrame(X_train_scaled, columns=X.columns).to_csv(os.path.join(output_folder, "X_train_scaled.csv"), index=False)
pd.DataFrame(X_test_scaled, columns=X.columns).to_csv(os.path.join(output_folder, "X_test_scaled.csv"), index=False)
y_train.to_frame(name='status').to_csv(os.path.join(output_folder, "y_train.csv"), index=False)
y_test.to_frame(name='status').to_csv(os.path.join(output_folder, "y_test.csv"), index=False)

print("Standardized train/test splits saved successfully with correct labels.")
