import pandas as pd
import os
import joblib
import mlflow
import mlflow.sklearn
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score, classification_report

# for hugging face space authentication to upload files
from huggingface_hub import login, HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError, HfHubHTTPError
import mlflow


mlflow.set_tracking_uri("http://localhost:5000")
# Set the name for the experiment
mlflow.set_experiment("mls10-mlops-tourismpackagepredict-experiment")

# CONFIGURATION
MODEL_REPO_ID = "SagarAtHf/tourismpackagepredict-model" # Corrected MODEL_REPO_ID
DATASET_REPO_ID = "SagarAtHf/tourismpackagepredict" # This is for the dataset files
DATA_DIR = "mls10-wellness_tourism_mlops/data"
# Ensure HF_TOKEN is set directly or as an environment variable
HF_TOKEN = os.getenv("HF_TOKEN")
# For testing, you can uncomment the line below and replace with your actual token


# Check if there's an active run and end it if so
if mlflow.active_run():
    mlflow.end_run()

api = HfApi()

def train_with_grid_search():
    # 1. Load data from Hugging Face datasets
    print("Step 1: Loading train/test datasets from Hugging Face...")

    # Download files locally first
    Xtrain_local_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="Xtrain.csv", repo_type="dataset", token=HF_TOKEN)
    Xtest_local_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="Xtest.csv", repo_type="dataset", token=HF_TOKEN)
    ytrain_local_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="ytrain.csv", repo_type="dataset", token=HF_TOKEN)
    ytest_local_path = hf_hub_download(repo_id=DATASET_REPO_ID, filename="ytest.csv", repo_type="dataset", token=HF_TOKEN)

    Xtrain = pd.read_csv(Xtrain_local_path)
    Xtest = pd.read_csv(Xtest_local_path)
    # ytrain and ytest are loaded as DataFrames, convert to Series if they contain only one column
    ytrain = pd.read_csv(ytrain_local_path).squeeze()
    ytest = pd.read_csv(ytest_local_path).squeeze()
    print("Datasets loaded successfully.")

    # 2. Define Feature Groups
    numeric_cols = [
        'Age', 'CityTier', 'DurationOfPitch', 'NumberOfPersonVisiting', 
        'NumberOfFollowups', 'PreferredPropertyStar', 'NumberOfTrips', 
        'Passport', 'PitchSatisfactionScore', 'OwnCar', 
        'NumberOfChildrenVisiting', 'MonthlyIncome'
    ]

    categorical_cols = [
        'TypeofContact', 'Occupation', 'Gender', 
        'ProductPitched', 'MaritalStatus', 'Designation'
    ]

    # 3. Create Preprocessor & Pipeline
    """
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_cols),
        (OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    )"""
    preprocessor = make_column_transformer(
        (StandardScaler(), numeric_cols),
        (OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_cols),
        remainder='drop' # Drops any column not explicitly mentioned
    )

    # 4. Set the class weight to handle class imbalance
    class_weight = ytrain.value_counts()[0] / ytrain.value_counts()[1]

    # 5. Pipeline & Grid Search Setup
    # Define base XGBoost model
    xgb_model = xgb.XGBClassifier(scale_pos_weight=class_weight, random_state=42)

    model_pipeline = make_pipeline(preprocessor, xgb_model)

    #6. Define hyperparameter grid
    param_grid = {
        'xgbclassifier__n_estimators': [ 75, 100],
        'xgbclassifier__max_depth': [3, 4],
        'xgbclassifier__colsample_bytree': [0.5, 0.6],
        'xgbclassifier__colsample_bylevel': [0.5, 0.6],
        'xgbclassifier__learning_rate': [0.05, 0.1],
        'xgbclassifier__reg_lambda': [0.5, 0.6],
    }

    #7. Execute Experiment

    with mlflow.start_run(run_name="Optimized_XGB_Model_PROD_Standalone"):
        grid_search = GridSearchCV(model_pipeline, param_grid, cv=5, n_jobs=-1)
        grid_search.fit(Xtrain, ytrain)

        # Log all parameter combinations and their mean test scores
        results = grid_search.cv_results_
        for i in range(len(results['params'])):
            param_set = results['params'][i]
            mean_score = results['mean_test_score'][i]
            std_score = results['std_test_score'][i]

        # Log each combination as a separate MLflow run
            with mlflow.start_run(nested=True):
                mlflow.log_params(param_set)
                mlflow.log_metric("mean_test_score", mean_score)
                mlflow.log_metric("std_test_score", std_score)

        # Log best parameters separately in main run
        mlflow.log_params(grid_search.best_params_)

         # Store and evaluate the best model
        best_model = grid_search.best_estimator_

        classification_threshold = 0.45

        #8. Comprehensive Metrics Logging

        y_pred_train_proba = best_model.predict_proba(Xtrain)[:, 1]
        y_pred_train = (y_pred_train_proba >= classification_threshold).astype(int)

        y_pred_test_proba = best_model.predict_proba(Xtest)[:, 1]
        y_pred_test = (y_pred_test_proba >= classification_threshold).astype(int)

        train_report = classification_report(ytrain, y_pred_train, output_dict=True)
        test_report = classification_report(ytest, y_pred_test, output_dict=True)


        y_pred = best_model.predict(Xtest)
        y_proba = best_model.predict_proba(Xtest)[:, 1] # Needed for ROC-AUC

        # Log the metrics for the best model
        metrics = {
            "train_accuracy": train_report['accuracy'],
            "train_precision": train_report['1']['precision'],
            "train_recall": train_report['1']['recall'],
            "train_f1-score": train_report['1']['f1-score'],
            "test_accuracy": test_report['accuracy'],
            "test_precision": test_report['1']['precision'],
            "test_recall": test_report['1']['recall'],
            "test_f1-score": test_report['1']['f1-score'],
            "test_roc_auc": roc_auc_score(ytest, y_proba)
        }

        # Log the metrics for the best model
        mlflow.log_metrics(metrics)

        # Print all metrics
        print("Logged Metrics:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")

        #10: Save the Model Artifact
        os.makedirs("mls10-wellness_tourism_mlops/model_building", exist_ok=True)
        model_path = "mls10-wellness_tourism_mlops/model_building/productionmodel.joblib"
        joblib.dump(best_model, model_path)

        #11: Log the model artifact
        mlflow.log_artifact(model_path, artifact_path="model")
        print(f"Model saved as artifact at: {model_path}")

        print("Model Training & Tracking Complete.")
        print(f"Metrics are: {metrics}")

        #12 Upload to Hugging Face
        REPO_TYPE = "model"

        #13: Check if the space exists
        try:
            api.repo_info(repo_id=MODEL_REPO_ID, repo_type=REPO_TYPE) # Use MODEL_REPO_ID here
            print(f"Space '{MODEL_REPO_ID}' already exists. Using it.")
        except RepositoryNotFoundError:
            print(f"Space '{MODEL_REPO_ID}' not found. Creating new space...")
            create_repo(repo_id=MODEL_REPO_ID, repo_type=REPO_TYPE, private=False)
            print(f"Space '{MODEL_REPO_ID}' created.")

        print(f"Uploading to Hugging Face Model Hub..")
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="productionmodel.joblib", # Corrected typo here
            repo_id=MODEL_REPO_ID,
            repo_type=REPO_TYPE,
        )

        print("Optimized model is uploaded to Hugging Face.")

if __name__ == "__main__":
    train_with_grid_search()
