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

nano-PEARL is a single-node, multi-GPU parallel speculative decoding engine. It decouples Draft and Target models onto separate device groups and runs them concurrently with on-the-fly verification, prefix KV caching, CUDA Graphs, and tensor parallelism — aiming for high throughput without sacrificing output quality.

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

## 📊 BenchMark Results

Coming Soon!

## 📋 TODOs

- [ ]  **Dynamic TP Size**: Support dynamic TP size, including TP=6/7, hence the 8 GPUs can be fully used!
- [ ]  **Draft Model Temperature**: Support setting a non-zero temperature for the draft model.
- [ ]  **Continuous Batching**: Support continuous batching and chunked prefill.

## 🐛 Bug Fixing
Coming Soon!

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