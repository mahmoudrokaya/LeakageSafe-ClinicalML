import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import os

# Define file paths
input_file = 'E:/Mahmoud/Exams/46/462/New-papers/Paper4-Under-Processing/HFAGM_Project/data/raw/covid_clinical.csv'
output_folder = 'E:/Mahmoud/Exams/46/462/New-papers/Paper4-Under-Processing/HFAGM_Project/data/preprocessed'
os.makedirs(output_folder, exist_ok=True)

# Load dataset
df = pd.read_csv(input_file)

# Drop rows with all NaNs
df.dropna(how='all', inplace=True)

# Encode labels
label_column = 'status'  # Adjust if your column name is different
df[label_column] = df[label_column].map({'recovered': 0, 'deceased': 1})

# Identify features
feature_cols = df.drop(columns=[label_column]).select_dtypes(include=['float64', 'int64']).columns.tolist()

# Impute missing values with median
df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())

# Normalize features
scaler = MinMaxScaler()
df[feature_cols] = scaler.fit_transform(df[feature_cols])

# Save preprocessed data
preprocessed_path = os.path.join(output_folder, 'covid_clinical_preprocessed.csv')
df.to_csv(preprocessed_path, index=False)
print(f"Saved preprocessed data to: {preprocessed_path}")

# Optional: Balance classes by upsampling
df_majority = df[df[label_column] == 0]
df_minority = df[df[label_column] == 1]

if len(df_majority) != len(df_minority):
    df_minority_upsampled = resample(df_minority,
                                     replace=True,
                                     n_samples=len(df_majority),
                                     random_state=42)
    df_balanced = pd.concat([df_majority, df_minority_upsampled])
    df_balanced = df_balanced.sample(frac=1, random_state=42)  # Shuffle

    balanced_path = os.path.join(output_folder, 'covid_clinical_balanced.csv')
    df_balanced.to_csv(balanced_path, index=False)
    print(f"Saved balanced data to: {balanced_path}")
else:
    print("Classes are already balanced.")
