import subprocess
import yaml
import os

def main():
    with open("config/model_config.yaml") as f:
        config = yaml.safe_load(f)

    # vLLM reads HF_TOKEN from env
    os.environ["HF_TOKEN"] = os.environ.get("HF_TOKEN", "")

    cmd = [
        "python3", "-m", "vllm.entrypoints.openai.api_server",
        "--model", config["model_name"],
        "--dtype", "bfloat16",
        "--host", "127.0.0.1",
        "--port", "8000"
    ]
    print("Starting vLLM server...")
    print("Command:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
