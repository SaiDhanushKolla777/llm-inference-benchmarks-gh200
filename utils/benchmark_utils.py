import time
import torch
import jax
import subprocess

def torch_measure_latency(func, *args, **kwargs):
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_event.record()
    out = func(*args, **kwargs)
    end_event.record()
    torch.cuda.synchronize()
    latency_ms = start_event.elapsed_time(end_event)
    return out, latency_ms

def get_gpu_memory():
    try:
        result = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.used --format=csv,nounits,noheader".split()
        )
        return int(result.decode().split("\n")[0])
    except Exception:
        return -1

def jax_latency(fn, *args, **kwargs):
    jax.clear_caches()
    start = time.time()
    result = fn(*args, **kwargs)
    jax.device_get(result)
    end = time.time()
    return result, (end-start)*1000

def log_markdown_result(framework, latency, tokens_sec, mem_mb, out_path="benchmark_results.md"):
    header = "| Framework | Latency (ms) | Tokens/sec | GPU Memory (MB) |\n|-----------|--------------|------------|-----------------|\n"
    line = f"| {framework:<9} | {latency:>10.2f} | {tokens_sec:>10.1f} | {mem_mb:>15} |\n"
    try:
        with open(out_path, "r") as f:
            content = f.read()
        if "Framework" not in content:
            with open(out_path, "w") as f:
                f.write(header)
        with open(out_path, "a") as f:
            f.write(line)
    except FileNotFoundError:
        with open(out_path, "w") as f:
            f.write(header)
            f.write(line)
