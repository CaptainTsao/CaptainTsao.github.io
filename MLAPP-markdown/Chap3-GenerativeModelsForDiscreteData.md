
# 3 Generative models for discrete data
## 3.1 Introduction

在第2.2.3.2节中，我们讨论了如何将贝叶斯规则应用于生成分类器来对特征向量$\mathbf{x}$进行分类，形式为
$$
p(y=c\vert\mathbf{x},\boldsymbol{\theta})\propto p(\mathbf{x}\vert y=c,\boldsymbol{\theta})p(y=c\vert\boldsymbol{\theta})        \tag{3.1}
$$

使用这种模型的关键是为类条件密度$p(\mathbf{x}\vert y=c, \boldsymbol{\theta})$指定一个合适的形式，它定义了我们希望在每个类中看到的数据类型。在本章中，我们将重点讨论观测数据是离散符号的情况。我们还讨论了如何推断这类模型的未知参数$\boldsymbol{\theta}$。

## 3.2 Bayesian concept learning

注意标准的二分类技术，需要正反两个例子。

### 3.2.1 Likelihood

我们必须解释为什么在看到$\mathcal{D}=\{16, 8, 2, 64\}$后，我为什么选择$h_{\text{two}}\triangleq "\text{powers of two}"$，$h_{\text{even}}\triangleq "\text{even numbers}"$，两个假设都与证据一致。关键的直觉是我们希望避免**可疑的巧合**。如果真实的概念是偶数，那为什么我们只看到的恰好是2的幂数呢。

为了对此进行描述，我们假设例子是从概念的扩展中随机抽取的。Tenenbaum称其为强采样假设。给定这个假设，从h中独立采样N个项的概率给定为
$$
p(\mathcal{D}\vert h) = \left[\frac{1}{\text{size}(h)} \right]^N = \left[\frac{1}{\vert h\vert} \right]^N       \tag{3.2}
$$

这个关键方程体现了Tenenbaum所说的尺寸原理，这意味着该模型支持与数据一致的最简单假设。这就是通常说的奥卡姆剃刀原理。

为了


### 3.2.2 Prior

假设$D=\{16, 8, 2, 64\}$。给定数据，概念$h^{\prime}$。

贝叶斯真理的这一主观方面引起了很多的争议，因为例如，这意味着孩子和数学教授将得到不同的答案。实际上，它们可能不仅具有不同的先验，而且具有不同的假设空间。但是我们，可以通过将孩子和数学教授的假设空间定义为相同，然后在某些“高级”概念上将孩子的先验权重设置为零来做到这一点。因此，先验空间和假设空间之间是没有明显的区别的。

