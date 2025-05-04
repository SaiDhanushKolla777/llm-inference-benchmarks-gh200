import os
from huggingface_hub import login

def hf_login():
    token = os.environ.get("HF_TOKEN")
    if not token:
        raise ValueError("No HF_TOKEN provided in environment variables.")
    login(token=token, add_to_git_credential=False)
