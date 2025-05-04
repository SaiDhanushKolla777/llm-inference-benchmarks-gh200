import yaml
import jax
import jax.numpy as jnp
from transformers import AutoTokenizer, FlaxAutoModelForCausalLM
from utils.benchmark_utils import jax_latency, get_gpu_memory, log_markdown_result

def main():
    with open("config/model_config.yaml") as f:
        config = yaml.safe_load(f)

    tokenizer = AutoTokenizer.from_pretrained(
        config["model_name"],
        use_auth_token=True,
        revision=config.get("hf_revision", "main"),
    )
    model = FlaxAutoModelForCausalLM.from_pretrained(
        config["model_name"],
        revision=config.get("hf_revision", "main"),
        dtype=jnp.bfloat16,
        use_auth_token=True
    )

    prompt = config["prompt"]
    inputs = tokenizer(prompt, return_tensors="jax")

    def generate_jax(input_ids):
        return model.generate(
            input_ids=input_ids,
            max_new_tokens=config["max_new_tokens"],
            temperature=config["temperature"],
            top_p=config["top_p"]
        )

    output, latency_ms = jax_latency(generate_jax, inputs["input_ids"])
    output_len = output.sequences.shape[-1] - inputs["input_ids"].shape[-1]
    tokens_per_sec = output_len / (latency_ms / 1000)
    gpu_mem = get_gpu_memory()

    log_markdown_result("JAX", latency_ms, tokens_per_sec, gpu_mem)

    print(f"[JAX] Latency: {latency_ms:.2f} ms | Tokens/sec: {tokens_per_sec:.1f} | GPU Mem: {gpu_mem} MB")
    print("Output:", tokenizer.decode(output.sequences[0][-output_len:], skip_special_tokens=True))

if __name__ == "__main__":
    main()
