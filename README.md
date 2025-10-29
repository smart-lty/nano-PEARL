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

# nano-PEARL
nano-PEARL is a single-node, multi-GPU parallel speculative decoding engine. It decouples Draft and Target models onto separate device groups and runs them concurrently with on-the-fly verification, prefix KV caching, CUDA Graphs, and tensor parallelism — aiming for high throughput without sacrificing output quality.

# 🎉 NEWS

TBD

# 🔥 Key Features
- **Draft-Target Disaggregation** : The draft model and the target model are loaded in seperate devices. Avoiding load-imbalance and resource competition.
- **Parallel Inference**: Both the draft model and the target model run inference in parallel, fully exploiting the GPU utilization!
- **Adaptive Draft Length**: 
  - When the alignment is good, the draft model could generate draft tokens without being interrupted by the  target model.
  - When the alignment is poor, the target model could prevent the draft model from generating trash draft tokens.
- **Auto-Set Hyper-parameters**


# Acknowledgements

TBD.

# Citation
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