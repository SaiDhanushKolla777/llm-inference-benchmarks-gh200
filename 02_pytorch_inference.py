import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.benchmark_utils import torch_measure_latency, get_gpu_memory, log_markdown_result

def main():
    with open("config/model_config.yaml") as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        use_auth_token=True,
        revision=config.get("hf_revision", "main")
    )
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name"],
        torch_dtype=torch.bfloat16,
        device_map="auto",
        revision=config.get("hf_revision", "main"),
        use_auth_token=True
    )
    model.eval()

    prompt = config["prompt"]
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    def generate():
        return model.generate(
            **inputs,
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"]
        )

    output, latency_ms = torch_measure_latency(generate)
    output_len = output.shape[-1] - inputs["input_ids"].shape[-1]
    tokens_per_sec = output_len / (latency_ms / 1000)
    gpu_mem = get_gpu_memory()

    log_markdown_result("PyTorch", latency_ms, tokens_per_sec, gpu_mem)

    print(f"[PyTorch] Latency: {latency_ms:.2f} ms | Tokens/sec: {tokens_per_sec:.1f} | GPU Mem: {gpu_mem} MB")
    print("Output:", tokenizer.decode(output[0][-output_len:], skip_special_tokens=True))

if __name__ == "__main__":
    main()
