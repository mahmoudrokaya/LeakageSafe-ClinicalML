import pandas as pd

# Load the dataset
df = pd.read_csv(r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project\data\preprocessed\covid_clinical_balanced.csv")

# Print the column names
print("Column names in the dataset:")
print(df.columns.tolist())