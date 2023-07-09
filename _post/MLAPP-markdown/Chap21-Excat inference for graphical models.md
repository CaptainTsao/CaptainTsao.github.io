[toc]

# 20 Exact inference for graphical models

## 20.1 Introduction

在17.4.3节中，我们讨论了前向-后向算法，这种算法可以精确的计算任何链式结构的图模型的后验分布$p(x_t\vert\mathbf{v},\boldsymbol{\theta})$，其中$\mathbf{x}$是隐变量，$\mathbf{v}$是可见变量。这个算法可以修改为计算后验mode以及后验采样。对于线性高斯链(linear Gaussian chain)的一个类似算法是Kalman平滑器，在18.3.2节中有所讨论。本章我们的目的是将这写精确推理算法推广到任意的图。得到的方法应用到了有向图与无向图模型。我们将讨论大量的算法，但是丢弃推导的简介性。