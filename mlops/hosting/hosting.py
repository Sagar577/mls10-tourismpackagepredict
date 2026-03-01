from huggingface_hub import HfApi
import os

# CONFIGURATION
REPO_ID = "SagarAtHf/tourismpackagepredict"

# CHANGE THIS LINE: 
# It was "mls10-wellness_tourism_mlops/deployment"
# It should now be "mlops/deployment" because that is the folder in your Git repo
DEPLOY_FOLDER = "mlops/deployment"

def deploy():
    # Use the token from GitHub Secrets/Environment
    hf_token = os.getenv("HF_TOKEN")
    print("deploy function is being called")
    print("DEPLOY_FOLDER is  " + DEPLOY_FOLDER)
    if not hf_token:
        print("❌ Error: HF_TOKEN not found.")
        return

    api = HfApi(token=hf_token)

    # Verify the directory exists before attempting upload
    if not os.path.isdir(DEPLOY_FOLDER):
        print(f"❌ Error: The directory '{DEPLOY_FOLDER}' was not found.")
        print("Available files in current directory:", os.listdir("."))
        # If 'mlops' exists, show what's inside it to help debug
        if os.path.exists("mlops"):
             print("Contents of 'mlops':", os.listdir("mlops"))
        return

    print(f"🚀 Uploading contents of {DEPLOY_FOLDER} to Hugging Face Space...")
    api.upload_folder(
        folder_path=DEPLOY_FOLDER,
        repo_id=REPO_ID,
        repo_type="space",
        path_in_repo="", 
    )

    print(f"✅ Deployment successful!")

if __name__ == "__main__":
    deploy()
