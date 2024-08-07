# HyFi

PyTorch implementation for Closer through commonality: Enhancing hypergraph contrast learning with shared groups
<br />

# Overview

![](Framework.png)

Hypergraphs provide a superior modeling framework for representing complex multidimensional relationships in the context of real-world interactions that often occur in groups, overcoming the limitations of traditional homogeneous graphs. However, there have been few studies on hypergraph-based contrastive learning, and existing graph-based contrastive learning methods have not been able to fully exploit the high-order correlation information in hypergraphs. Here, we propose a Hypergraph Fine-grained contrastive learning (HyFi) method designed to exploit the complex high-dimensional information inherent in hypergraphs. While avoiding traditional graph augmentation methods that corrupt the hypergraph topology, the proposed method provides a simple and efficient learning augmentation function by adding noise to node features. Furthermore, we expands beyond the traditional dichotomous relationship between positive and negative samples in contrastive learning by introducing a new relationship of weak positives. It demonstrates the importance of fine-graining positive samples in contrastive learning. Therefore, HyFi is able to produce high-quality embeddings, and outperforms both supervised and unsupervised baselines in average rank on node classification across 10 datasets. Our approach effectively exploits high-dimensional hypergraph information, shows significant improvement over existing graph-based contrastive learning methods, and is efficient in terms of training speed and GPU memory cost.

<br />

# Docker Container
- Docker container use HyFi project directory as volume 
- File change will be apply directly to file in docker container
<br />

# Experiment 
1. `make up` : build docker image and start docker container
2. `python3 train.py` : start experiment in docker container
3. you can check the hyperparameter settings in `config.yaml`
<br />

# Requirements

```
pandas
numpy
torch == 2.0.1
torch_scatter
torch_geometric
tqdm
scikit-learn
dgl
matplotlib
```


# Acknowledgements

The data sets and part of our code refer to the [TriCL](https://github.com/wooner49/TriCL) repo. Many thanks for them.

<!-- # Reference
This code is free and open source for only academic/research purposes (non-commercial). If you use this code as part of any published research, please acknowledge the following paper.
```
soon
``` -->