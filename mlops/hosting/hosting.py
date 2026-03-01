from huggingface_hub import HfApi
import os

# CONFIGURATION
# This is the Space where the Streamlit UI will live
REPO_ID = "SagarAtHf/tourismpackagepredict"

# Retrieve HF_TOKEN from environment variables
# Ensure this is set either as a Colab Secret (named HF_TOKEN) or via os.environ['HF_TOKEN'] = 'your_token'
#HF_TOKEN = os.getenv("HF_TOKEN")
#Token = os.getenv("HF_TOKEN")
DEPLOY_FOLDER = "mls10-wellness_tourism_mlops/deployment"

def deploy():
    """if not os.getenv("HF_TOKEN"):
        print("❌ Error: HF_TOKEN not found in environment variables.")
        print("Please ensure 'HF_TOKEN' is set as a Colab secret or an environment variable.")
        return
    """
    #api = HfApi(token=os.getenv("HF_TOKEN"))
    api = HfApi()
    # 1. Ensure the Space exists (Set to Streamlit SDK)
    print(f"Checking if Space {REPO_ID} exists...")
    try:
        api.create_repo(
            repo_id=REPO_ID,
            repo_type="space",
            space_sdk="docker", # Since we are using the custom Dockerfile
            exist_ok=True
        )
    except Exception as e:
        print(f"Note on repo creation: {e}")

    # 2. Upload the entire deployment folder
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
