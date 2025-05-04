# llm-inference-benchmarks-gh200
Production-ready benchmarking suite: compare Hugging Face LLM inference on NVIDIA GH200 between PyTorch, JAX/Flax, and vLLM. Unified latency, throughput, and memory logging.  Designed for LLM system and ML acceleration with clean config and reproducible scripts.

---

# LLM Inference Benchmarks on NVIDIA GH200

Welcome! 👋 This repository is a hands-on benchmarking suite for Large Language Model (LLM) inference on NVIDIA’s groundbreaking Grace Hopper Superchip (GH200). Here, you’ll find code and configs to test-and fairly compare-the performance of Hugging Face LLMs (like LLaMA 2) across three of today’s most important machine learning frameworks: **PyTorch, JAX/Flax, and vLLM**.

Whether you’re an ML systems researcher, a hardware acceleration engineer, or just curious about the state-of-the-art for LLM infrastructure, this project is here to help you measure, understand, and optimize transformer inference at scale.

---

## 🌟 Why This Project?

The world of LLMs is evolving at breakneck speed. With increasingly powerful hardware and a rapidly expanding ecosystem of inference engines, choosing the right software stack for *your* use case has never been more important.

I created this repository to answer a straightforward, but surprisingly nuanced question:

> **“How do PyTorch, JAX/Flax, and vLLM actually compare for real-world LLM inference on a modern accelerator like NVIDIA’s GH200?”**

By making every stage-from model authentication, to metrics logging, to GPU profiling-transparent and reproducible, this project aims to give you (and me!) deeper insight into both the frameworks and the underlying hardware.

---

## 🖼️ What’s Inside

This suite is designed to be modular, open, and easy to extend. Here’s what you’ll find:

- **Secure Hugging Face Integration:**  
  Authenticate automatically to load gated models like LLaMA-2.

- **Config-Driven Workflow:**  
  Set your model, prompt, and generation hyperparameters in a single YAML file-no code edits needed for quick experiments.

- **Unified Benchmark Scripts:**  
  Run apples-to-apples benchmarks in PyTorch, JAX/Flax, and vLLM. Each logs latency, tokens/sec throughput, and GPU memory.

- **Clear, Reproducible Output:**  
  All results are appended as Markdown tables to a single `benchmark_results.md`-no more digging through console output.

- **Ready for GH200 & Beyond:**  
  Works great on NVIDIA GH200’s unique ARM64 architecture, but is easily adaptable for other GPU servers, cloud VMs, and x86 workstations.

---

## 🏗️ Project Structure

```
llm-inference-benchmarks-gh200/
├── 01_setup_environment.py              # Environment check + Hugging Face auth
├── 02_pytorch_inference.py              # PyTorch benchmarking
├── 03_jax_inference.py                  # JAX/Flax benchmarking
├── 04_vllm_server_setup.py              # Launch vLLM as an OpenAI API server
├── 05_vllm_inference_client.py          # Send a prompt to vLLM, log metrics
├── config/
│   └── model_config.yaml                # All your model/prompt/generation params
├── auth/
│   └── hf_auth.py                       # Hugging Face token login
├── utils/
│   └── benchmark_utils.py               # Latency, throughput, memory, markdown utils
├── requirements.txt                     # Python dependencies
├── benchmark_results.md                 # Results (populated when you run scripts)
└── README.md                            # This file!
```

---

## 🚦 Quickstart: How To Use

### 1. Clone & Set Up Your Virtual Environment

```
git clone https://github.com/SaiDhanushKolla777/llm-inference-benchmarks-gh200.git
cd llm-inference-benchmarks-gh200
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install Python Dependencies

```
pip install -r requirements.txt
```
> **Note:**  
> On NVIDIA GH200 (or any ARM64 system), see vLLM install instructions below.

### 3. Export Your Hugging Face Token

All scripts use secure authentication for gated models.  
Get your token from [Hugging Face Settings](https://huggingface.co/settings/tokens).

```
export HF_TOKEN=your_actual_token_here
```

### 4. Run Environment Check & Authentication

Let’s make sure everything is in order:

```
python 01_setup_environment.py
```

### 5. Run the Benchmarks

Each script will append its results to `benchmark_results.md`.

```
python 02_pytorch_inference.py
python 03_jax_inference.py
python 04_vllm_server_setup.py &   # Start vLLM server (run in background/new terminal)
python 05_vllm_inference_client.py
```

---

## 📝 Model & Prompt Configuration

All your settings live in `config/model_config.yaml`.  
Here’s a sample:

```
model_name: meta-llama/Llama-2-7b-hf
prompt: "What are three unique advantages of the GH200 Grace Hopper Superchip for LLM inference?"
max_new_tokens: 64
temperature: 0.2
top_p: 0.9
hf_revision: main
```

Want to benchmark a different model or prompt? Just edit this YAML-no code changes needed.

---

## ⚠️ vLLM on ARM64 / GH200

One important hardware note:  
If you’re running this on NVIDIA GH200 (ARM64), you’ll need to install `vLLM` from source, since binary wheels for ARM aren’t available yet.

```
sudo apt-get update
sudo apt-get install -y build-essential python3-dev git
git clone https://github.com/vllm-project/vllm.git
cd vllm
pip install -r requirements.txt
pip install .
```

If you’re on x86_64 (standard CUDA server), you can simply install `vllm` with pip as noted in `requirements.txt`.

---

## 🔬 Extending This Project

- **Try new models:** Swap out the YAML for OPT, GPT-NeoX, Mistral, or vision transformers.
- **Batch/Multiple Prompts:** Benchmark with multiple prompts for broader insight.
- **Quantization:** Experiment with int8, fp16, or bfloat16 inference for speed/memory tradeoffs.
- **Profiling:** Integrate NVIDIA Nsight, `torch.profiler`, or JAX profiling tools for deeper system visibility.
- **Serve Anywhere:** With vLLM’s OpenAI API compatibility, you can integrate this with open-source chat UIs, LangChain, and more.

---

## 📚 References & Further Reading

- [NVIDIA Grace Hopper Superchip (GH200)](https://www.nvidia.com/en-us/data-center/grace-hopper-superchip/)
- [LLaMA 2 on Hugging Face](https://huggingface.co/meta-llama/)
- [vLLM: Open and Fast LLM Inference](https://github.com/vllm-project/vllm)
- [PyTorch](https://pytorch.org/)
- [JAX](https://jax.readthedocs.io/en/latest/)
- [Flax](https://flax.readthedocs.io/en/latest/)

---

## 💬 About Me & Contributions

I built this project to close the gap between marketing benchmarks and hands-on, transparent ML systems methods. If you’re trying these scripts, benchmarking your own models, or adapting this for new hardware-let me know!  
Issues and discussions are welcome: [Open an issue](https://github.com/SaiDhanushKolla777/llm-inference-benchmarks-gh200/issues) or join the conversation [here](https://github.com/SaiDhanushKolla777/llm-inference-benchmarks-gh200/discussions).

