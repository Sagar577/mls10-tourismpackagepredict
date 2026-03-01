from huggingface_hub import HfApi
import os

# CONFIGURATION
# This is the Space where the Streamlit UI will live
REPO_ID = "SagarAtHf/tourismpackagepredict"

# UPDATED: Path now points to the 'mlops' directory you created
DEPLOY_FOLDER = "mlops/deployment"

def deploy():
    # Explicitly retrieve the token from environment variables for GitHub Actions
    hf_token = os.getenv("HF_TOKEN")
    
    if not hf_token:
        print("❌ Error: HF_TOKEN not found in environment variables.")
        return

    api = HfApi(token=hf_token)

    # 1. Ensure the Space exists
    print(f"Checking if Space {REPO_ID} exists...")
    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="docker", 
            exist_ok=True
        )
    except Exception as e:
        print(f"Note on repo creation: {e}")

    # 2. Upload the folder
    # Verify the directory exists before attempting upload to prevent the ValueError
    if not os.path.isdir(DEPLOY_FOLDER):
        print(f"❌ Error: {DEPLOY_FOLDER} is not a valid directory.")
        # Print current directory content to help debug pathing
        print("Current directory contents:", os.listdir("."))
        return

    print(f"🚀 Uploading contents of {DEPLOY_FOLDER} to Hugging Face Space...")
    api.upload_folder(
        folder_path=DEPLOY_FOLDER,
        repo_id=REPO_ID,
        repo_type="space",
        path_in_repo="", # Upload directly to the root of the Space
    )

    print(f"✅ Deployment successful! View your app at: https://huggingface.co/spaces/{REPO_ID}")

if __name__ == "__main__":
    deploy()
