# RaBitQ

The RaBitQ algorithm is a drop-in replacement of binary quantization and (uniform) scalar quantization, with its [1-bit version](https://arxiv.org/abs/2405.12497) (released in May 2024) and [multi-bit version](https://arxiv.org/abs/2409.09913) (released in Sep 2024), respectively.

<!-- It provides significantly better accuracy under the same space budget and is theoretically proven to be asymptotically optimal.  -->

<!-- For a given input data vector $\mathbf{x}$ and a bit budget $B$, the RaBitQ algorithm outputs a code vector $\mathbf{x}_u$ and a rescaling factor $\Delta_x$ such that we can estimate similarity metrics based on the code vector $\mathbf{x}_u$ and the rescaling factor $\Delta_x$ as accurately as possible. -->

The key advantages of RaBitQ include

- **High Accuracy with Tiny Space** - RaBitQ achieves the state-of-the-art accuracy under diverse space budgets for the estimation of similarity metrics. It produces promising accuracy with even **1-bit per dimension**.
- **Fast Distance Estimation** - RaBitQ supports to estimate the similarity metrics with high efficiency based on bitwise operations or [FastScan](https://arxiv.org/abs/1704.07355).
- **Theoretical Error Bound** - RaBitQ provides an asymptotically optimal error bound for the estimation of distances and inner product. The error bound can be used for reliable ordering and [reranking](reranking.md).


## Workflow 

The RaBitQ algorithm includes two steps:

1. **Random Rotation** - Sample a random rotation and apply it to all vectors (including the raw data vectors, the center vector and the raw query vectors). See [Rotator](rotator.md) for more details.

2. **Quantization** - After the random rotation, the quantization algorithm quantizes a vector of floating-point numbers into a vector of low-bit unsigned integers. See [Quantizer](quantizer.md) for more details.


After the quantization, we can estimate the similarity metrics including Euclidean distance, inner product and cosine similarity based on the code vector $\mathbf{x}_u$ and the rescaling factor $\Delta_x$. See [Estimator](estimator.md) for more details.

