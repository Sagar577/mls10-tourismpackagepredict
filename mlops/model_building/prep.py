import pandas as pd
from sklearn.model_selection import train_test_split # for data preprocessing and pipeline creation
from sklearn.preprocessing import LabelEncoder # for encoding the category variables
#from datasets import load_dataset
from huggingface_hub import HfApi # for hugging face authentication and uploading files
import os

# CONFIGURATION
# This matches the repo created in the previous step
REPO_ID = "SagarAtHf/tourismpackagepredict"
DATA_DIR = "mls10-wellness_tourism_mlops/data"
HF_TOKEN = os.getenv("HF_TOKEN")
DATASET_PATH = "hf://datasets/SagarAtHf/tourismpackagepredict/tourism.csv"

def prepare_data():
    print("Step 1: Loading dataset from Hugging Face...")
    # Load the dataset from HF.
    # If dataset has multiple files, load_dataset handles it automatically.
    # dataset = load_dataset(REPO_ID)

    df = pd.read_csv(DATASET_PATH)
    print("Dataset loaded from hugging face successfully.")


    # Converting the 'train' split (default) to a pandas DataFrame
    # df = pd.DataFrame(dataset['train'])

# 2.1 Remove unnecessary columns
    print("Step 2: Performing data cleaning...")
    # Remove unnecessary columns -> customerID is Primary Key and there exists an unnamed column which appears to be Serial number. Both are not useful for modeling
    cols_to_drop = ['CustomerID', 'Unnamed: 0']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

# 2.2 Standardize Categorical Values
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('Fe Male', 'Female')
    if 'MaritalStatus' in df.columns:
        # Merging 'Single' into 'Unmarried' to simplify the categories
        df['MaritalStatus'] = df['MaritalStatus'].replace('Single', 'Unmarried')

# 2.3 Label Encoding for categorical variables
    # This converts columns like 'Occupation' from strings to numbers
    cat_cols = df.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    for col in cat_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        print(f" Encoded column: {col}")



    print("Step 3: Splitting into train and test sets...")

    # Define target variable
    target_col = 'ProdTaken'

    # Split into X (features) and y (target)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # Perform train-test split
    # Stratified split to ensure equal proportion of buyers in both sets
    Xtrain, Xtest, ytrain, ytest = train_test_split(
        X, y, test_size=0.2, random_state=42,stratify=y)

    # Save locally
    os.makedirs(DATA_DIR, exist_ok=True)
    paths = {
        "Xtrain.csv": os.path.join(DATA_DIR, "Xtrain.csv"),
        "Xtest.csv": os.path.join(DATA_DIR, "Xtest.csv"),
        "ytrain.csv": os.path.join(DATA_DIR, "ytrain.csv"),
        "ytest.csv": os.path.join(DATA_DIR, "ytest.csv")
    }

    for name, path in paths.items():
        # Convert Series to DataFrame for consistent to_csv behavior if needed, otherwise ensure it works for both
        obj = eval(name.split(".")[0])
        if isinstance(obj, pd.Series):
            obj.to_csv(path, index=False, header=True)
        else:
            obj.to_csv(path, index=False)

    print(f"Local files saved in {DATA_DIR}")

    print("Step 4: Uploading processed data back to Hugging Face...")
    api = HfApi(token=HF_TOKEN)

    # Upload files
    for filename, local_path in paths.items():
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=filename,
            repo_id=REPO_ID,
            repo_type="dataset",
        )

    print("Preprocessing complete. Train/Test sets uploaded to Hugging Face.")

if __name__ == "__main__":
    prepare_data()
