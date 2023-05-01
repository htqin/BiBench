## *BiPointNet: Binary Neural Network for Point Clouds*

Created by [Haotong Qin](https://htqin.github.io/)\*, [Mingyuan Zhang](https://scholar.google.com/citations?user=2QLD4fAAAAAJ&hl=en)\*, [Yifu Ding](https://yifu-ding.github.io/), Aoyu Li, [Fisher Yu](https://www.yf.io/), and [Xianglong Liu](https://xlliu-beihang.github.io/) from Beihang University, Nanyang Technological University, and ETH Zurich.

![prediction example](https://htqin.github.io/Imgs/ICLR/overview_v1.png)

### Introduction

This project is the official implementation of our accepted ICML 2023 paper ***BiBench: Benchmarking and Analyzing Network Binarization*** [[PDF](https://arxiv.org/pdf/2301.11233.pdf)]. Network binarization emerges as one of the most promising compression approaches offering extraordinary computation and memory savings by minimizing the bit-width. However, recent research has shown that applying existing binarization algorithms to diverse tasks, architectures, and hardware in realistic scenarios is still not straightforward. Common challenges of binarization, such as accuracy degradation and efficiency limitation, suggest that its attributes are not fully understood. To close this gap, we present ***BiBench***, a rigorously designed benchmark with in-depth analysis for network binarization. We first carefully scrutinize the requirements of binarization in the actual production and define evaluation tracks and metrics for a comprehensive and fair investigation. Then, we evaluate and analyze a series of milestone binarization algorithms that function at the operator level and with extensive influence. Our benchmark reveals that 1) the binarized operator has a crucial impact on the performance and deployability of binarized networks; 2) the accuracy of binarization varies significantly across different learning tasks and neural architectures; 3) binarization has demonstrated promising efficiency potential on edge devices despite the limited hardware support. The results and analysis also lead to a promising paradigm for accurate and efficient binarization. We believe that BiBench will contribute to the broader adoption of binarization and serve as a foundation for future research.

### Citation

If you find our work useful in your research, please consider citing:

```
@inproceedings{qin2023bibench,
  author    = {Haotong Qin and Mingyuan Zhang 
  and Yifu Ding and Aoyu Li and Ziwei Liu 
  and Fisher Yu and Xianglong Liu},
  title     = {BiBench: Benchmarking and Analyzing Network Binarization},
  booktitle = {ICML},
  year      = {2023}
}
```

***The full code is coming soon...***