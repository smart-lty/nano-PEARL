<div align="center">

![nano-pearl](static/nano_pearl.gif)

[![Website](https://img.shields.io/badge/website-nano--PEARL-orange)](https://smart-lty.github.io/nano-PEARL/)
[![Status](https://img.shields.io/badge/status-active-brightgreen)](#) 
[![ArXiv](https://img.shields.io/badge/arXiv-2408.11850-b31b1b)](https://arxiv.org/abs/2408.11850)
[![Conference](https://img.shields.io/badge/ICLR-2025-4B7BEC)](#)

<em><strong>Unleashing Batch Throughput: </strong><br>Reimagined Speculative Decoding with Parallelism and Adaptive Draft Length<br></em>

</div>

# 🚀 nano-PEARL

nano-PEARL is a single-node, multi-GPU parallel speculative decoding engine that decouples Draft and Target models onto separate device groups. With on-the-fly verification, prefix KV caching, CUDA Graphs, paged attention, flash attention, and tensor parallelism, it delivers exceptional throughput while maintaining output quality.

<font color="#ADD8E6">**Ready to jumpstart your research?** All advanced acceleration features are built-in—focus on your algorithm, not the engineering.</font>

> <font color="#90EE90">✨ No more integration worries from aggressive reviewers!</font> 

## 🎉 Latest News

🚧 **Coming Soon**: More updates and features are in development!

📚 **Paper Collection**: We are seeking for papers that follows parallel speculative decoding paradigm! We are going to prepare a paper collection of these papers for more comprehensive understanding.
- [2025.11] 🔥 We release more benchmark results of nano-PEARL on NVIDIA L40S!
- [2025.11] 🔥 Our web page of [nano-PEARL](https://smart-lty.github.io/nano-PEARL/) is established!
- [2025.10] 🔥 We release the source code of nano-PEARL. Any PR is warmly welcomed!

## ✨ Key Features

- 🔄 **Draft-Target Disaggregation**: The draft model and the target model are loaded in separate devices, avoiding load-imbalance and resource competition.
- ⚡ **Parallel Inference**: Both the draft model and the target model run inference in parallel, fully exploiting the GPU utilization!
- 🎯 **Adaptive Draft Length**: 
  - When the alignment is good, the draft model could generate draft tokens without being interrupted by the target model.
  - When the alignment is poor, the target model could prevent the draft model from generating trash draft tokens.
- 🤖 **Auto-Set Hyper-parameters**: Automatically configure optimal parameters for your hardware setup.
- 🚀 **High Performance**: Built on CUDA Graphs and tensor parallelism for maximum throughput.
- 💾 **Memory Efficient**: Prefix KV caching reduces memory usage while maintaining performance.


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
<details><summary><b>Some hints for potential problem solving</b></summary>

> ⚠️ When you directly use pip for installation, you may encounter that build flash-attn needs torch installed. In this case, you should **install torch first**, and then re-run the installation command.

> ⚠️ If the installation of flash-attn is very slow, we strongly recommend you to download a whl file and **build flash attn from wheel**.
</details>

## 🚀 Quick Start

The `nano-PEARL` API mirrors `vLLM` / `nano-vllm`'s interface. The main difference is in the `LLM` engine initialization, where you must **specify both a target model and a draft model**, along with their respective tensor-parallel (TP) sizes. Detailed Quick Start is included in [Quick Start](https://smart-lty.github.io/nano-PEARL/quickstart.html).

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

We conduct extensive evaluation of nano-PEARL on various datasets / Hardware settings. Below is the evaluation of nano-PEARL with NVIDIA H200 on HumanEval with batch size 32. nano-PEARL achieves a maximal $3.06\times$ throughput speedup with **3546.72 tok/s** for 70B LLMs!

Feel Free to check more benchmark results at our [benchmark page](https://smart-lty.github.io/nano-PEARL/benchmark.html)!

![bench_example](static/benchmark_example.png)

## 📋 TODOs

- [ ]  **Dynamic TP Size**: Support dynamic TP size, including TP=6/7, hence the 8 GPUs can be fully used!
- [ ]  **Draft Model Temperature**: Support setting a non-zero temperature for the draft model.
- [ ]  **Continuous Batching**: Support continuous batching and chunked prefill.
- [ ]  **Adaptive Gamma**: Support dynamic `gamma` tuning based on context size and model's performance.
- [ ]  **PEARL-2**: Support fine-tuning / distilling a PEARL-specific draft model for further acceleration.

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
```
