<div align="center">

![nano-pearl](static/nano_pearl.gif)

[![Status](https://img.shields.io/badge/status-active-brightgreen)](#) 
[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.4-EE4C2C)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900)](#)
[![ArXiv](https://img.shields.io/badge/arXiv-2408.11850-b31b1b)](https://arxiv.org/abs/2408.11850)
[![Conference](https://img.shields.io/badge/ICLR-2025-4B7BEC)](#)

<em>A lightweight parallel speculative decoding implementation in nano-vllm style.<br></em>

</div>

# 🚀 nano-PEARL

> **A lightweight parallel speculative decoding implementation in nano-vllm style**

nano-PEARL is a single-node, multi-GPU parallel speculative decoding engine. It decouples Draft and Target models onto separate device groups and runs them concurrently with on-the-fly verification, prefix KV caching, CUDA Graphs, page attention, flash attention and tensor parallelism — aiming for high throughput without sacrificing output quality.

## 🎉 Latest News

🚧 **Coming Soon**: More updates and features are in development!

- [2025.10] 🔥 We release the source code of nano-PEARL. Any PR is warmly welcomed!

## 📦 Installation

Our nano-PEARL is built based on [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm), and the installation is almost same with nano-vllm (only need to additionally install `rich` for colorful log). 🎨


First create an environment with `python>=3.12`:
```shell
conda create -n nano-pearl python=3.12 -y
conda activate nano-pearl
```

Then, install packages with `uv` or `pip`:

**From source:**
```shell
uv pip install -e . # from current path
```

**From GitHub:**
```shell
pip install git+https://github.com/smart-lty/nano-PEARL.git # from github
```
⚠️ When you directly use pip for installation, you may encounter that build flash-attn needs torch installed. In this case, you should **install torch first**, and then re-run the installation command.

⚠️ If the installation of flash-attn is very slow, we strongly recommend that you download a whl file and **build flash attn from wheel**.


## ✨ Key Features

- 🔄 **Draft-Target Disaggregation**: The draft model and the target model are loaded in separate devices, avoiding load-imbalance and resource competition.
- ⚡ **Parallel Inference**: Both the draft model and the target model run inference in parallel, fully exploiting the GPU utilization!
- 🎯 **Adaptive Draft Length**: 
  - When the alignment is good, the draft model could generate draft tokens without being interrupted by the target model.
  - When the alignment is poor, the target model could prevent the draft model from generating trash draft tokens.
- 🤖 **Auto-Set Hyper-parameters**: Automatically configure optimal parameters for your hardware setup.
- 🚀 **High Performance**: Built on CUDA Graphs and tensor parallelism for maximum throughput.
- 💾 **Memory Efficient**: Prefix KV caching reduces memory usage while maintaining performance.
## 🚀 Quick Start

The `nano-PEARL` API mirrors `vLLM` / `nano-vllm`'s interface. The main difference is in the `LLM` engine initialization, where you must **specify both a target model and a draft model**, along with their respective tensor-parallel (TP) sizes.

See `example.py` for usage: a minimal example of running parallel speculative decoding on 2 GPUs (e.g., 1 for the target model, 1 for the draft model):

```python
from nano_pearl import PEARLConfig, PEARLEngine, SamplingParams, logger

def main():
    draft_model_path = "/path/to/draft/model"
    target_model_path = "/path/to/target/model"
    
    config = PEARLConfig(draft_model_path, target_model_path, draft_tensor_parallel_size=1, target_tensor_parallel_size=1, gpu_memory_utilization=0.9)
    engine = PEARLEngine(config)
    
    prompt = "Explain quantum computing in simple terms"
    sampling_params = SamplingParams(temperature=0.0, max_tokens=256, ignore_eos=False)
    engine.add_request(prompt, sampling_params)
    
    output_text, num_tokens, num_acc_tokens, elapsed_time = engine.generate()
```
## 📊 BenchMark Results

See `bench.py` for benchmark.

## ⚙️ Implementation Details


## ⚙️ Implementation Details

This section details the core configuration parameters.

#### 📦 Model & Parallelism Parameters (example.py, bench.py, pearl_engine.py)

* `draft_model_path: str`
    * **Description**: Path to the draft model (small model). Supports local paths or Hugging Face hub identifiers.
* `target_model_path: str`
    * **Description**: Path to the target model (large model). Supports local paths or Hugging Face hub identifiers.
* `draft_tensor_parallel_size: int`
    * **Description**: The tensor-parallel (TP) size allocated to the **draft model**. For example, `1` loads the draft model onto a single GPU.
* `target_tensor_parallel_size: int`
    * **Description**: The tensor-parallel (TP) size allocated to the **target model**.
    * **Constraint**: The sum `draft_tensor_parallel_size + target_tensor_parallel_size` must not exceed the total number of available GPUs (currently <= 8).

#### 💾 Memory & Batching Parameters (pearl_engine.py)

* `gpu_memory_utilization: float`
    * **Description**: A value between `0.0` and `1.0` representing the fraction of GPU memory allocated for the KV Cache. This is the **key parameter** for memory management.
    * **Recommendation**: `0.9` (90%) is a safe and efficient starting point.

> **Note:** The following parameters must be tuned based on available GPU memory and specific deployment scenarios.

* `max_num_batched_tokens: int`
    * **Description**: The maximum total number of tokens (i.e., `batch_size * sequence_length`) processed by the engine in a single batch.
    * **Recommendation**: `16384` for 80GB GPUs (H100/A100); `8192` for 40GB GPUs.
* `max_num_seqs: int`
    * **Description**: The maximum number of sequences (requests) processed by the engine concurrently.
    * **Recommendation**: `512` for 80GB GPUs; `128` or `256` for 40GB GPUs.
* `max_model_len: int`
    * **Description**: The maximum context length supported by the model (including prompt and generated tokens).
* `set_gamma_batch_size: list[int]`
    * **Description**: Specifies the batch sizes for which `gamma` is auto-tuned (when `gamma = -1`).
    * **Recommendation**: Default: `[1, 2, 4, 8, 16, 32, 64]`. This list can be reduced (e.g., `[1, 8, 32]`) in resource-constrained scenarios to reduce profiling time.

#### 🧠 Algorithm & Engine Parameters (example.py, bench.py)

* `gamma: int`
    * **Description**: The **window size for adaptive draft length** in the PEARL algorithm; i.e., the number of tokens the draft model speculates.
    * **Recommendation**: Set to `-1` (default) to enable **auto-setting**. The engine will automatically select a near-optimal value.
* `enforce_eager: bool`
    * **Description**: If `True`, forces Eager mode and disables CUDA Graphs.
    * **Recommendation**: Keep as `False` (default) for maximum performance. Set to `True` only for debugging purposes.
* `temperature: float`
  t * **Description**: The sampling temperature. `0.0` indicates greedy sampling, used in `example.py` and benchmarks for deterministic outputs.
    * **Recommendation**: Use `0.0` for deterministic output or benchmarks; use `> 0.0` (e.g., `0.7`) for creative tasks.
* `max_tokens: int`
    * **Description**: The maximum number of new tokens to generate for a given request.
    * **Recommendation**: Set a reasonable limit (e.g., `256`, `512`, `1024`) to prevent runaway generations.
* `ignore_eos: bool`
    * **Description**: If `True`, the generation process ignores the EOS (End-of-Sentence) token and continues until `max_tokens` is reached.
    * **Recommendation**: Keep as `False` (default) unless generating a fixed-length output is required.



## 📋 TODOs

- [ ]  **Dynamic TP Size**: Support dynamic TP size, including TP=6/7, hence the 8 GPUs can be fully used!
- [ ]  **Draft Model Temperature**: Support setting a non-zero temperature for the draft model.
- [ ]  **Continuous Batching**: Support continuous batching and chunked prefill.
- [ ]  **Aligend Models**: Support finetuned models for PEARL(qwen3).

## 🐛 Bug Fixing
Coming Soon!

## 📄 File structure
```
.
├── LICENSE
├── README.md
├── bench.py                     # Benchmark testing script
├── example.py                   # Usage examples
├── nano_pearl                   # Core library directory
│   ├── __init__.py              # Package initialization file
│   ├── layers                   # Neural network layer implementations
│   │   ├── activation.py        # Activation functions
│   │   ├── attention.py         # Attention mechanisms
│   │   ├── embed_head.py        # Embedding and head layers
│   │   ├── layernorm.py         # LayerNorm implementation
│   │   ├── linear.py            # Linear layers
│   │   ├── rotary_embedding.py  # Rotary position encoding
│   │   └── sampler.py           # Samplers
│   ├── models                   # Model implementations
│   │   ├── __init__.py          # Models package initialization
│   │   ├── llama.py             # LLaMA model implementation
│   │   ├── qwen2.py             # Qwen2 model implementation
│   │   └── qwen3.py             # Qwen3 model implementation
│   ├── pearl_config.py          # PEARL configuration management
│   ├── pearl_engine             # PEARL engine core
│   │   ├── block_manager.py     # Block manager
│   │   ├── pearl_engine.py      # Main engine implementation
│   │   ├── pearl_model_runner.py # Model runner
│   │   ├── scheduler.py         # Scheduler
│   │   └── sequence.py          # Sequence management
│   └── utils                    # Utility functions
│       ├── __init__.py          # Utilities package initialization
│       ├── context.py           # Context management
│       ├── loader.py            # Model loader
│       └── pearl_logger.py      # Logger
├── pyproject.toml               # Python project configuration
├── static                       # Static resources
│   ├── default_prompts.txt      # Default prompts
│   └── nano_pearl.gif           # Project logo animation
└── uv.lock                      # uv dependency lock file
```

## 🙏 Acknowledgements


This project has been influenced by many execellent projects in the LLM community, such as [nano-vllm](https://github.com/GeeeekExplorer/nano-vllm) and [PEARL](https://github.com/smart-lty/ParallelSpeculativeDecoding). The nano-PEARL logo is designed by Veo 3. 

## 📚 Citation
```bibtex
@inproceedings{
liu2025pearl,
title={{PEARL}: Parallel Speculative Decoding with Adaptive Draft Length},
author={Tianyu Liu and Yun Li and Qitan Lv and Kai Liu and Jianchen Zhu and Winston Hu and Xiao Sun},
booktitle={The Thirteenth International Conference on Learning Representations},
year={2025},
url={https://openreview.net/forum?id=QOXrVMiHGK}
}

@misc{liu2025pearlparallelspeculativedecoding,
      title={PEARL: Parallel Speculative Decoding with Adaptive Draft Length}, 
      author={Tianyu Liu and Yun Li and Qitan Lv and Kai Liu and Jianchen Zhu and Winston Hu and Xiao Sun},
      year={2025},
      eprint={2408.11850},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2408.11850}, 
}
@misc{nanopearl,
  author = {Tianyu Liu},
  title = {nanochat: The best ChatGPT that $100 can buy},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/smart-lty/nano-PEARL}
}
```
