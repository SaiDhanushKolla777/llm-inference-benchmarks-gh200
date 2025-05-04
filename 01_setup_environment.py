import subprocess
from auth.hf_auth import hf_login

def check_cuda():
    try:
        output = subprocess.check_output(["nvidia-smi"]).decode()
        print("nvidia-smi output:\n", output)
    except Exception as e:
        print(f"CUDA not detected or nvidia-smi not found: {e}")

def main():
    print("=== Checking CUDA/GPU ===")
    check_cuda()
    print("\n=== Authenticating with Hugging Face Hub ===")
    hf_login()
    print("\nSetup complete! You are ready to run benchmarks.")

if __name__ == "__main__":
    main()
