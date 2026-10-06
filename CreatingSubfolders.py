import os

# Base project path
base_path = r"E:\Mahmoud\Exams\46\462\New-papers\Paper4-Under-Processing\HFAGM_Project"

# Define full structure with nested subfolders
folder_structure = [
    "config",
    "data/raw",
    "data/preprocessed",
    "data/synthetic",
    "preprocessing",
    "models/generators",
    "models/encoders",
    "models/classifiers",
    "training/contrastive",
    "training/ensemble",
    "fairness/metrics",
    "fairness/constraints",
    "experiments/scenario1",
    "experiments/scenario2",
    "experiments/scenario3",
    "experiments/scenario4",
    "utils",
    "visualizations/plots",
    "visualizations/monitoring",
    "saved_models/generators",
    "saved_models/encoders",
    "saved_models/classifiers",
    "outputs/evaluation",
    "outputs/logs"
]

# Create folders
for folder in folder_structure:
    full_path = os.path.join(base_path, folder)
    os.makedirs(full_path, exist_ok=True)
    print(f"Created: {full_path}")
