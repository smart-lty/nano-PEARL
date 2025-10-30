<div align="center">

![nano-pearl](static/nano_pearl.gif)

[![Status](https://img.shields.io/badge/status-active-brightgreen)](#) 
[![Python](https://img.shields.io/badge/python-%E2%89%A53.12-blue)](#)
[![PyTorch](https://img.shields.io/badge/PyTorch-%E2%89%A52.4-EE4C2C)](#)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900)](#)
[![ArXiv](https://img.shields.io/badge/arXiv-2408.11850-b31b1b)](https://arxiv.org/abs/2408.11850)
[![Conference](https://img.shields.io/badge/ICLR-2025-4B7BEC)](#)

<em>A <strong>high-throughput</strong> parallel speculative decoding engine in nano-vllm style.<br></em>

</div>

# 🚀 nano-PEARL

> **A high-throughput parallel speculative decoding engine in nano-vllm style**

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

⚠️ If the installation of flash-attn is very slow, we strongly recommand you to download a whl file and **build flash attn from wheel**.


## ✨ Key Features

- 🔄 **Draft-Target Disaggregation**: The draft model and the target model are loaded in separate devices, avoiding load-imbalance and resource competition.
- ⚡ **Parallel Inference**: Both the draft model and the target model run inference in parallel, fully exploiting the GPU utilization!
- 🎯 **Adaptive Draft Length**: 
  - When the alignment is good, the draft model could generate draft tokens without being interrupted by the target model.
  - When the alignment is poor, the target model could prevent the draft model from generating trash draft tokens.
- 🤖 **Auto-Set Hyper-parameters**: Automatically configure optimal parameters for your hardware setup.
- 🚀 **High Performance**: Built on CUDA Graphs and tensor parallelism for maximum throughput.
- 💾 **Memory Efficient**: Prefix KV caching reduces memory usage while maintaining performance.
- 
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

The core configuration of `nano-PEARL` is split into two main classes:

1.  **`PEARLConfig`**: Defined in `pearl_config.py`, this is used for **engine initialization**. It manages models, parallel settings, memory allocation, and batching strategies.
2.  **`SamplingParams`**: This is used when **submitting a request**. It controls the sampling behavior (like temperature, `max_tokens`, etc.) for a single generation task.

---

### 1. PEARLConfig (Engine Configuration)

This is the most important configuration object, passed when initializing the `PEARLEngine`.

#### 📦 Model & Parallelism Parameters

* `draft_model_path: str`
    * **Description**: The path to the draft model (small model). Can be a local path or a Hugging Face hub repository name.
* `target_model_path: str`
    * **Description**: The path to the target model (large model).
* `draft_tensor_parallel_size: int`
    * **Description**: The tensor-parallel (TP) size allocated for the **draft model**. For example, `1` means the draft model will be loaded entirely onto a single GPU.
* `target_tensor_parallel_size: int`
    * **Description**: The tensor-parallel (TP) size allocated for the **target model**.
    * **Constraint**: The sum `draft_tensor_parallel_size + target_tensor_parallel_size` must be less than or equal to the total number of available GPUs (currently supports up to 8).

#### 💾 Memory & Batching Parameters

* `gpu_memory_utilization: float`
    * **Description**: A value between `0.0` and `1.0` representing the fraction of GPU memory to be used for the KV Cache. This is the **key parameter** for controlling memory usage.
    * **Recommendation**: `0.9` (90%) is a safe and efficient starting point.
* `max_num_batched_tokens: int`
    * **Description**: The maximum total number of tokens (i.e., `batch_size * sequence_length`) that the engine can process in a single batch.
    * **Recommendation**: `16384` for 80GB GPUs (H100/A100); `8192` for 40GB GPUs.
* `max_num_seqs: int`
    * **Description**: The maximum number of sequences (requests) the engine can process concurrently.
    * **Recommendation**: `512` for 80GB GPUs; `128` or `256` for 40GB GPUs.
* `max_model_len: int`
    * **Description**: The maximum context length supported by the model (including prompt and generated tokens).
* `kvcache_block_size: int`
    * **Description**: The block size for the KVCache in paged-attention. The default of `256` is suitable for most cases.
* `num_kvcache_blocks: int`
    * **Description**: The total number of KVCache blocks. If set to `-1` (default), the engine will automatically calculate the maximum number of blocks based on `gpu_memory_utilization`.

#### 🧠 Algorithm & Engine Parameters

* `gamma: int`
    * **Description**: This is the **window size for adaptive draft length** in the PEARL algorithm, i.e., how many tokens the draft model "looks ahead" at one time.
    * **Recommendation**: Set to `-1` (default) to enable **auto-setting**. The engine will automatically select a near-optimal value based on the model configuration.
* `enforce_eager: bool`
    * **Description**: Whether to force Eager mode.
    * **Recommendation**: Keep as `False` (default) to enable CUDA Graphs for maximum performance. Only set to `True` for debugging purposes.

---

### 2. SamplingParams (Request Configuration)

These parameters are passed during `engine.add_request()` to control the generation behavior for a **single request**.

* `temperature: float`
    * **Description**: The sampling temperature. `0.0` indicates greedy sampling, which is used in `example.py` and benchmarks for deterministic outputs.
* `max_tokens: int`
    * **Description**: The maximum number of new tokens to generate for this request.
* `ignore_eos: bool`
    * **Description**: If `True`, the generation process will ignore the EOS (End-of-Sentence) token and continue until `max_tokens` is reached.

---

### 3. Engine Generate Output

A call to `engine.generate()` returns a tuple with 4 elements:

1.  `output_text: list[str]`
    * **Description**: A list of the generated text strings for each request in the batch.
2.  `num_tokens: list[int]`
    * **Description**: A list of the **total number of new tokens actually generated** for each request.
3.  `num_acc_tokens: list[list[int]]`
    * **Description**: **[Key Spec-Dec Metric]** A nested list. The outer list corresponds to each request. The inner list records the **number of tokens accepted at each verification step**.
    * **Example**: `[[5, 4, 6, 2]]` means the first request had 4 verification steps, accepting 5, 4, 6, and 2 tokens, respectively.
    * **Usage**: `sum(num_acc_tokens[0]) / len(num_acc_tokens[0])` (as shown in `example.py`) is used to calculate the **Mean Acceptance Tokens (MAT)**.
4.  `elapsed_time: float`
    * **Description**: The total time in seconds spent processing the entire batch.

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
