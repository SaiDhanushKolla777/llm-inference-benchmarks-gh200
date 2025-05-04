import yaml
import requests
import time
from utils.benchmark_utils import get_gpu_memory, log_markdown_result

def main():
    with open("config/model_config.yaml") as f:
        config = yaml.safe_load(f)

    prompt = config["prompt"]
    payload = {
        "model": config["model_name"],
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": config["max_new_tokens"],
        "temperature": config["temperature"],
        "top_p": config["top_p"]
    }

    url = "http://127.0.0.1:8000/v1/chat/completions"
    start = time.time()
    response = requests.post(url, json=payload, timeout=120)
    latency_ms = (time.time() - start) * 1000
    result = response.json()
    output_text = result["choices"][0]["message"]["content"]
    tokens = len(output_text.split())
    tokens_per_sec = tokens / (latency_ms / 1000)
    gpu_mem = get_gpu_memory()

    log_markdown_result("vLLM", latency_ms, tokens_per_sec, gpu_mem)
    print(f"[vLLM] Latency: {latency_ms:.2f} ms | Tokens/sec: {tokens_per_sec:.1f} | GPU Mem: {gpu_mem} MB")
    print("Output:", output_text)

if __name__ == "__main__":
    main()
