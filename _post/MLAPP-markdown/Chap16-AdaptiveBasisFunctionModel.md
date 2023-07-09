<!-- TOC -->

- [16 自适应基函数模型(Adaptive basis function models)](#16-自适应基函数模型adaptive-basis-function-models)
  - [16.1 引言(Introduction)](#161-引言introduction)
  - [16.2 分类与回归树(CART)](#162-分类与回归树cart)
    - [16.2.1 基础](#1621-基础)
    - [16.2.2 生成一棵树(Growing a tree)](#1622-生成一棵树growing-a-tree)
      - [16.2.2.1 回归树](#16221-回归树)
      - [16.2.2.2 分类树](#16222-分类树)
    - [16.2.3 剪枝树](#1623-剪枝树)
    - [16.2.4 树的优缺点(Pros and cons of trees)](#1624-树的优缺点pros-and-cons-of-trees)
    - [16.2.5 随机森林](#1625-随机森林)
    - [16.2.6 CART compared to hierarchical mixture of experts*](#1626-cart-compared-to-hierarchical-mixture-of-experts)
  - [16.3 广义加性模型(Generative Additive Models)](#163-广义加性模型generative-additive-models)
  - [16.4 提升模型(Boosting)](#164-提升模型boosting)
    - [16.4.1](#1641)
    - [16.4.2 L2boosting](#1642-l2boosting)
    - [16.4.3 AdaBoost](#1643-adaboost)
    - [16.4.5 Boosting as functional gradient descent](#1645-boosting-as-functional-gradient-descent)
    - [16.4.6 稀疏Boosting](#1646-稀疏boosting)
    - [16.4.8 为什么Boosting如此有效](#1648-为什么boosting如此有效)
    - [16.4.9 贝叶斯观点](#1649-贝叶斯观点)
  - [16.5 前向神经网络(FeedForward Nueral Networks(MLP))](#165-前向神经网络feedforward-nueral-networksmlp)
    - [16.5.1 卷积神经网络](#1651-卷积神经网络)
    - [16.5.2 其他类型的神经网络](#1652-其他类型的神经网络)
    - [16.5.4 反向传播算法](#1654-反向传播算法)

<!-- /TOC -->
# 16 自适应基函数模型(Adaptive basis function models)

## 16.1 引言(Introduction)


在第14与15章中，我们讨论了核函数，为回归与分类提供了创建非线性模型的强有力方式。预测的形式是$f(x)=\mathbf{w}^{T}\phi(\mathbf{x})$，我们定义
$$
\phi(\mathbf{x})=[\kappa(\mathbf{x},\mathbb{\mu}_{1}),\mathellipsis,\kappa(\mathbf{x},\mathbb{\mu}_{k})] \tag{16.1}
$$
其中，$\mu_{k}$既可以是训练数据也可以是一些子集。这种形式的模型本质上是执行了一个***模板匹配***，从而对比输入$\mathbf{x}$与存储原型$\mu_{k}$。

尽管这个很有效，但是依赖于有一个好的核函数来测量数据向量之间的相似度。同时遇到一个好的核函数是很难得。例如，如果定义两个图片之间的相似度？

在15.2.4节中，我们讨论了一种学习核函数的参数的方式，即最大化边缘似然。例如，如果我们使用ARD核
$$
\kappa(\mathbf{x},\mathbf{x}^{'}) = \theta_{0}\mathbf{exp}\left(
    \frac{1}{2}\sum_{j=1}^{D}\theta_{i}(x_{j}-x_{j}^{'})^2
    \right) \tag{16.2}
$$
我们可以估计$\theta_{j}$，然后执行一个**非线性特征选择**。然而，这种方法计算开销很大。另一种方法，已知我多核学习，使用一个基核的凸组合$\kappa(\mathbf{x},\mathbf{x}^{'})=\sum_{j}w_{j}\kappa_{j}(\mathbf{x},\mathbf{x}^{'})$，然后估计混合权重$w_{j}$。但是这依赖于有一个很好地基核。

另一种方法是直接放弃核，并尝试直接从输入数据中学习有效的特征$\phi(\mathbf{x})$。我们将创建一种形式如下的模型，称为自适应基函数[adaptive basis-function model(**ABM**)]
$$
f(\mathbf{x})=w_{0}+\sum_{m=1}^{M}w_{m}\phi_{x}(\mathbf{x})
$$
其中$\phi_{x}(\mathbf{x})$是第$m$个基函数，是从数据中学习得到的。

一般基函数是参数化的，所以我们可以写$\phi_{m}(\mathbf{x})=\phi(\mathbf{x};\mathbf{v}_{m})$，其中$\mathbf{v}_{m}$是基函数的参数。我们将使用$\mathbb{\theta}=(w_{0},\mathbf{w}_{1:M},\{\mathbf{v}_{m}\}_{1}^{M})$来表示全部参数集。得到的模型不再是关于参数线性的，所以我们只能计算参数$\mathbb{\theta}$的一个局部最优MLE或MAP。至少，这种模型通常由于线性模型，目前是这样。

## 16.2 分类与回归树(CART)

分类与回归树(CART)模型，也称为决策树（不要与决策论中的决策树混淆），是通过对输入空间进行回溯分割得到的，并在每个输入空间得到的区域中定义一个局部模型。可以被展示为一个树，其中每个区域一个叶子，解释如下。

### 16.2.1 基础
为了解释CART方法，考虑图中的树。第一个节点问$x_{1}$是否小于阈值$t_{1}$。如果是，则询问$x_{2}$是否小于阈值$t_{2}$。如果是，我们在空间的左下象限$R_{1}$。如果不是，我们询问$x_{1}$是否小于阈值$t_{3}$。等等。这些坐标轴平行分割的结果是将二维空间分割为5个区域，如图中所示。现在，我们可以将**平均响应**与这些区域中的每个区域相关联，从而得到如图16.1(c)所示的分段恒定表面。

我们可以将模型写为如下
$$
f(x)=\mathbb{E}[y|x]=\sum_{m=1}^{M}w_{m}\mathbb{I}(x\in R_{m})=\sum_{m=1}^{M}w_{m}\phi(\mathbf{x};\mathbf{v}_{m}) \tag{16.4}
$$
其中$R_{m}$是第$m$个区域，$w_{m}$是这个区域的平均响应，$\mathbf{v}_{m}$编码了从根到第$m$个叶子节点的路径上变量的分割的选择以及阈值。可以很清晰的看出，一个CART模型就是一个自适应基函数模型，其中基函数定义了区域，权重指定了每个区域上的响应值。下面我们讨论如何发现这些基函数。

我们可以将叶子节点上的类标签的分布进行存储，而不是均值响应，从而将问题推广到分类背景中。如图16.2所示。这个模型可以用图1.1中的数据进行分类。例如，我们首先检查物体的颜色，如果是蓝色，我们跟随左支，然后以一个标签为"4.0"的叶子结束，意思是我们有匹配准则的4个正例子0个负例子。因此我们预测如果$\mathbf{x}$是蓝色，则$p(y=1|x)=4/4$。如果是红色，我们检查形状；如果是一个椭圆，我们以一个标签为"1,1"的叶子结束，所以我们预测$p(y=1|x)=1/2$。如果是红色但不是椭圆，我们预测$p(y=1|x)=0/2$。如果是其他颜色，我们检查大小；如果小于10，我们预测$p(y=1|x)=4/4$，要不然$p(y=1|x)=0/5$。这些概率只是满足特征值每个连接的正例子的经验分割，定义了从根节点到叶子的路径。

### 16.2.2 生成一棵树(Growing a tree)

寻找数据的最优分割是一个NP完全问题，因此是通常使用如算法6中的贪婪算法来实现，计算了一个局部最优MLE。CART、C4.5与ID3都使用了这个方法。

分割函数选择最好的特征，以及特征的最优值，如下
$$
(j^{*}, t^{*})=\arg\min_{j\in\{1,\dots,D\}}\min_{t\in\mathcal{T}_{j}} \text{cost}(\{\mathbf{x}_{i},y_{i} :x_{ij} \leq t\}) + \text{cost}(\{\mathbf{x}_{i},y_{i} :x_{ij} \gt t\})
$$
其中一个给定数据集合的成本函数定义如下。处于符号简化，我们假设所有的输入是实数值或是ordinal，所以对比一个特征$x_{j}$与实数值$t$是很有意义的。特征j的可能的阈值集合$\mathcal{T}_{j}$可以通过排序$x_{ij}$的唯一值来获得。例如，如果特征1有值$\{4.5,-12,72,-12\}$，那么我们设定$\mathcal{T}_{j}=\{-12,4.5,72\}$。这种分类输入的情况下，最常用的方法是对每个可能的类标签$c_{j}$考虑形式如$x_{ij}=c_{k},x_{ij}\neq c_{k}$的分割。尽管也允许使用多路分割(生成非二叉树)，这将导致数据碎片化，意味着只有很少部分的数据落入每个子树中，导致过拟合。

检查一个结点是否值得分割的函数可以使用几种停止启发，例如如下的：
- 成本的降低是否过小？一般，我们将使用一个特征的增益定义为成本降低的归一化测量：
$$
\Delta = \text{cost}(\mathcal{D}) - \left( \frac{|\mathcal{D}_{L}|}{|\mathcal{D}|} \text{cost}(\mathcal{D}_{L}) + \frac{|\mathcal{D}_{R}|}{|\mathcal{D}|}\text{cost}(\mathcal{D}_{R})\right) \tag{16.6}
$$
- 树是否已经超过最大的期望深度
- 在$\mathcal{D}_{L}$或$\mathcal{D}_{R}$中的响应分布是否充分同质(也就是说所有的标签都相同，所以分布是纯化的)?
- $\mathcal{D}_{L}$或$\mathcal{D}_{R}$中的样例数目是否过小？

所有的方法都是指定用来评估一个提议分割质量的成本测量方法。这依赖于我们的目标是回归还是分类。


#### 16.2.2.1 回归树

在回归背景中，我们将成本定义如下：
$$
\text{cost}(\mathcal{D})=\sum_{i\in\mathcal{D}}(y_{i}-\bar{y})^2 \tag{16.7}
$$
其中$\bar{y}=\frac{1}{|\mathcal{D}|}\sum_{i\in\mathcal{D}}y_{i}$是指定数据集合中的响应变量的均值。另外，我们可以为每个叶子节点拟合一个线性回归，使用从根节点路径上的特征作为输入，然后测量残差。

#### 16.2.2.2 分类树

在分类背景中，有几种测量一个分割质量的方法。首先，我们通过估计形如
$$
\hat{\pi}_{c}=\frac{1}{|\mathcal{D}|}\sum_{\in\mathcal{D}}\mathbb{I}(y_{i}=c) \tag{16.8}
$$
的类条件概率估计，将叶子中满足测试$X_{j}<t$的数据拟合为一个multinouli模型，其中$\mathcal{D}$是叶子中的数据。给定这个前提，有几种计算一个提议分割的常用误差测量方式。

- **误分类率**。我们定义概率最高的类标签为$\hat{y}_{c}=\argmax_{c}\hat{y}_{c}$。对应的误差率为
$$
\frac{1}{|\mathcal{D}|}\sum_{i\in\mathcal{D}}\mathbb{I}(y_{i}\neq\hat{y})=1-\hat{\pi}_{\hat{y}} \tag{16.9}
$$
- **熵**或**偏离**：
$$
\mathbb{H}(\hat{\pi})=-\sum_{c=1}^{C}\hat{\pi}_{c}\log\hat{\pi}_{c} \tag{16.10} 
$$
$f(x)=x\log(x)$是一个凸函数。注意到，最小化熵等效于最大化测试$X_{j}<t$与类标签$Y$之间的**信息增益**，信息增益定义为
$$
\begin{aligned}
    \text{InfoGain}(X_{j}\lt t,Y)   &\triangleq  \mathbb{H}(Y) - \mathbb{H}(Y|X_{j} \lt t) \\
    &= \left( -\sum_{c}p(y=c)\log p(y=c) \right) \\
    & + \left( \sum_{c} p(y=c|X_{j} \lt t)\log p(y=c|X_{j} \right)
\end{aligned}
$$
由于$\hat{\pi}_{c}$是分布$p(c|X_{j}\lt t)$的MLE(最大似然估计)
- **基尼系数**
$$
\sum_{c=1}^{C}\hat{\pi}_{c}(1-\hat{\pi}_{c})=\sum_{c}\hat{\pi}_{c}-\sum_{c}\hat{\pi}_{c}^{2}=1-\sum_{c}\hat{\pi}_{c}^{2} \tag{16.14}
$$
这是期望误差率。记为$\hat{\pi}$为属于类c的叶子中的任意一项的概率，$1-\hat{\pi}$为误分类的概率。

在2分类的情况中，其中$p=\pi_{m}(1)$，误分类率是$1-\max(p,1-p)$，熵为$\mathbb{H}_{2}(p)$，基尼系数为$2p(1-p)$。如图16.3。我们看到交叉熵与基尼系数非常类似，对类概率的变化比对误分类更加敏感。例如，考虑一个有400个情况的二分类问题。假设一个分割创建点(100, 300)或(300, 100),而其他的创建节点(200, 0)与(400, 200)。两个分割产生一个0.25的误分类。然而，后者更喜欢，由于其中一个节点是纯化的；也即是只包含一类。交叉熵与基尼测量将会有助于后者的选择。

### 16.2.3 剪枝树

如果误差下降不足以充分证明添加额外子树造成的复杂度增加，为了防止过拟合，我们可以停止生成树。然而这往往倾向于过于短视。例如，在xor数据，可能永远无法进行分割，因为每个特征本身很少有预测能力。

因此标准方法是生成一棵"完全"树，然后执行剪枝。这可以通过使用一种剪枝方案来实现，该方案使误差增加最小。

为了确定剪枝的程度，我们可以评估每个子树上的交叉验证误差，然后挑选CV误差在1个最小标准差以内的树。如图所示。有最小CV误差的点对应于最简单的树。

### 16.2.4 树的优缺点(Pros and cons of trees)

CART模型流行的几个主要原因是：它们易于解释，可以处理混合了离散与连续的输入，对输入的单调变换不是很敏感(因为分割点是基于数据点的排序)，它们自动执行变量选择，对异常值相对鲁棒，对大的数据集合有很好的缩放，可以修改为处理缺失输入。

然而，CART模型也有一些缺点。**第一个就是相对其他模型很难准确预测。部分原因是因为树构建算法的贪婪本质造成的。一个相关问题就是树不稳定：输入数据一个很小的改变都会对树结构有很大的影响，这是因为生成树过程的层次本性造成的，导致在顶部的误差对后续的树的影响比较大。在频率方面，我们认为树是高方差估计器。**

### 16.2.5 随机森林

一种减少估计方差的方式是将许多估计器平均起来。例如，我们可以在不同的数据子集上训练M个不同的树，然后有放回的随机选择，计算集成
$$
f(\mathbf{x})=\sum_{m=1}^{M}\frac{1}{M}f_{m}(x)  \tag{16.15}
$$
其中$f_{m}$是第$m$棵树。这个技术称为**bagging**，意思是"bootstrap aggregating"。

不幸的是，在数据的不同子集上简单的重复运行相同的学习算法会导致高相关的预测器，限制了方差可能下降的幅度。这个技术称为**随机森林**，通过尝试随机选择输入变量的子集以及随机选择数据的子集来学习树，从而抵消基学习器的相关性。这样的模型通常具有较好的预测精度，在许多应用中有广泛的使用。

Bagging是一个频率概念。也可能采用一种贝叶斯方法来学习树。尤其MCMC对树空间执行近似推理。这减少了预测器的方差。我们还可以在树的集合空间上执行Bayesian推断，这样往往效果更好。这称为Bayesian自适应回归树或BART。注意到，这种基于采样的Bayesian方法与基于抽样的随机森林的方法相当。但是这两种方法都很缓慢，但是可以产生高质量的分类器。

不幸的是，使用多棵树的方法失去了良好的可解释性。

### 16.2.6 CART compared to hierarchical mixture of experts*

CART一个有趣的替选是一种称为层次混合专家的模型。图11.7(b)解释了哪里需要两层专家。这可以被认为是一个深度为2的概率决策树，因为回溯的分割空间，对每个分割应用一个不同的专家。Hastie写到"HME方法是CART树的一个有力竞争者"。一些优势为：
- 模型可以使用嵌套线性决策边界的任何子集来分割输入空间。相反的，标准决策树受限于使用平行于坐标轴的分割。
- 模型通过平均所有专家来做出预测。相反的，在一个标准决策树，预测只是基于对应叶子的模型来做出预测。因为叶子通常包含很少的训练例子，这将导致过拟合。
- 拟合一个HME涉及到求解一个平滑的连续最优化问题，与用于拟合决策树的标准贪婪离散优化方法相比，该问题可能不太容易出现局部最优解。


## 16.3 广义加性模型(Generative Additive Models)

一种创建多输入的非线性模型的简单方法是使用一种称为广义加性模型(GLM),模型形式如下
$$
\begin{aligned}
    f(\mathbf{x})=\alpha + f_{1}(x_{1}) + \cdots + f_{D}(x_{D})
\end{aligned}
$$
这里每个$f_{i}$可以使用一些散点平滑器建模，$f(\mathbf{x})$可以使用一个链接函数映射到$p(y|\mathbf{x})$。

如果让$f_{j}$使用回归线条(或是其他的固定基函数扩展方法)，那么每个$f_{j}(x_{j})$可写为$\mathbf{\beta}_{j}^{T}\phi_{j}(x_{j})$，所以整个模型可以写成$f(\mathbf{x})=\mathbf{\beta}^{T}\phi(\mathbf{x})$，其中$\phi(\mathbf{x})=[1, \phi_{1}(x_{1}), \cdots, \phi_{D}(x_{D})]$。然而，通常为$f_{j}$使用平滑线条。这种情况下，目标函数变成了
$$
\begin{aligned}
    J(\alpha_{1}, f_{1}, \cdots, f_{D}) = \sum_{i=1}^{N}\left( y_{i} -\alpha -\sum_{j=1}^{D}f_{j}(x_{ij})  \right)^2 + \sum_{j=1}^{D}\lambda_{j}\int f_{j}^{''}(t_{j})^2dt_{j} \tag{16.17}
\end{aligned}
$$
其中$\lambda_{j}$是$f_{j}$的正则器的强度。


## 16.4 提升模型(Boosting)

Boosting是一种贪婪算法，用来拟合形式如(16.3)的自适应基函数模型，其中$\phi_{m}$是由一种称为弱学习器的算法产生的。这种算法通过顺序应用弱学习到加权版的数据来工作，其中对先前几轮错误的分类示例给予更高的权重。

这个弱学习器可以是任意的回归或分类算，但是通常使用一个CART模型。1998年，Leo Breiman称Boosting是世界上最好的现成分类器，其弱学习器是一个肤浅的决策树。这得到了对10种不同的分类器的广泛经验比较的支持，后者表明Boosting决策树是在误分类误差以及产生良好的校准概率方面都是最好的，正如ROC曲线所判断的。相比之下，但决策树表现非常差。

Boosting源自计算学习理论，主要关注二分类。在这些文章中，证明可以任意高的提高弱学习器的表现，前提是弱学习器表现总是优于偶然表现。


鉴于其惊人的经验成功，统计学家开始对这种方法感兴趣。Breimain证明Boosting可以被解释为函数空间中的一个梯度下降的方式。这一观点随后被扩展，展示了如果将boosting扩展到处理各种损失函数，包括回归、稳健回归以及泊松回归等等。本节中我们展示了Boosting的统计解释。

### 16.4.1

Boosting的目标是求解如下的优化问题
$$
\min_{f}\sum_{i=1}^{N}L(y_{i},f(\mathbf{x}_{i}))
$$
其中$L(y,\hat{y})$是一些损失函数，$f$假设是一个ABM模型。损失函数的常用选择通常如表16.1中所示。

如果我们使用平方误差损失，则最优估计给定为
$$
f^{*}(\mathbf{x})=\argmax_{f(\mathbf{x})}=\mathbb{E}_{y|\mathbf{x}}\left[ (Y-f(\mathbf{x})) \right] = \mathbb{E}[Y|\mathbf{x}] \tag{16.26}
$$

Name|Loss|Derivative| f* | Algorithm
-|-|-|-|-
Squared Error | $\frac{1}{2}(y_{i} - f(\mathbf{x}_{i}))^2$ | $y_{i} - f(\mathbf{x}_{i})$ | $\mathbb{E}[y\|\mathbf{x}_{i}]$ | L2boosting
Absolute Error | $\text{abs}(y_{i}-f(\mathbf{x}_{i}))$ | $\text{sgn}(y_{i}-f(\mathbf{x}_{i}))$ | $\text{median}(y\|\mathbf{x}_{i})$ | Gradient Boosting
Exponential loss | $\text{exp}(-\tilde{y}_{i}f(\mathbf{x}_{i}))$ | 


当然，这在现实中是不能计算的，由于其需要知道真实的条件概率$p(y|x)$。这有时称为样本最小化，因为期望可以以概率的概念解释。下面，我们将看到Boosting将尝试逼近这个条件概率。

对于二分类，上述损失是0-1损失，但是这是不可微的。相反的是，通常可使用对数损失，这将是一个凸上边界的0-1损失，正如我们在6.5.5节中显示的。这种情况下，可以看到最优估计给定为
$$
\begin{aligned}
    f^{*}(\mathbf{x}) = \frac{1}{2}\log\frac{p(\tilde{y}=1)|\mathbf{x}}{p(\tilde{y}=-1)|\mathbf{x}}
\end{aligned} \tag{16.27}
$$
其中$\tilde{y}\in\{-1,+1\}$。可以将这个框架推广到多类情况。

另一种凸上边界是**指数损失**，定义为
$$
\begin{aligned}
    L(\tilde{y},f) = \text{exp}(-\tilde{y}f)
\end{aligned}
$$
看图16.9的一个图。这在对数损失上面有一定的优势。证明这个损失的最优估计也是$f^{*}(\mathbf{x})=\frac{1}{2}\log\frac{p(\tilde{y}=1)|\mathbf{x}}{p(\tilde{y}=-1)|\mathbf{x}}$。为了证明这个，我们可以设期望损失的导数为0：
$$
\begin{aligned}
    \frac{\partial}{\partial f(\mathbf{x})}\mathbb{E}\left[ e^{-\tilde{y}f(\mathbf{x})}|\mathbf{x} \right] &= \frac{\partial}{\partial f(\mathbf{x})} \left[ p(\tilde{y}=1|\mathbf{x})e^{-f(\mathbf{x})} + p(\tilde{y}=-1|\mathbf{x})e^{f(\mathbf{x})} \right] \\
    &=  -p(\tilde{y}=1|\mathbf{x})e^{-f(\mathbf{x})} + p(\tilde{y}=-1|\mathbf{x})e^{f(\mathbf{x})} \\
    &= 0 \Rightarrow \frac{p(\tilde{y}=1)|\mathbf{x}}{p(\tilde{y}=-1)|\mathbf{x}} = e^{2f(\mathbf{x})}
\end{aligned}
$$

在这两种情况中，我们可以看到Boosting方法尝试逼近对数几率比例。

因为发现最优的$f$是很难得到的，我们将顺序处理之。我们通过定义
$$
f_{0}(\mathbf{x}) = \argmin_{\gamma}\sum_{i=1}^{N}L(y_{i},f(\mathbf{x}_{i};\gamma))
$$
例如，如果我们使用平方误差，我们可以设置$f_{0}(\mathbf{x})=\bar{y}$，如果我们使用对数损失或是指数损失，我们可以设置$f_{0}(\mathbf{x})=\frac{1}{2}\log\frac{\hat{\pi}}{1-\hat{\pi}}$，其中$\hat{\pi}=\frac{1}{N}\sum_{i=1}^{N}\mathbb{I}(y_{i}=1)$。我们也可以为我们的基线使用一个更加强有力的模型，例如一个GLM。

那么在第$m$次迭代时，我们计算
$$
(\beta_{m},\gamma_{m})=\argmin_{\beta,\gamma}\sum_{i=1}^{N}L(y_{i},f_{m-1}\mathbf{x}_{i}+\beta\phi(\mathbf{x}_{i};\gamma)) \tag{16.33}
$$
那么我们设置
$$
f_{m}(\mathbf{x}) = f_{m-1}(\mathbf{x}) + \beta_{m}\phi(\mathbf{x};\gamma_{m}) \tag{16.34}
$$
关键点是我们不会回去调整之前的参数。这个方法是前向逐段加性模型。我们在固定迭代次数M下继续这样做，实际上M是该方法的主要调参。通常，我们通过在一个单独的实验事件总监控性能，然后在性能开始下降时停止；这也称为**早停(early stopping)**。另外，我们也可以使用例如AIC或BIC这样的模型选择标准。

实际中，可以通过执行"部分更新"获得更好的(测试集)业绩，形式为
$$
\begin{aligned}
    f_{m}(\mathbf{x}) = f_{m-1}(\mathbf{x}) + \nu \beta_{m}\phi(\mathbf{x};\gamma_{m}) \tag{16.35}
\end{aligned}
$$
这里$1 \lt \nu \leq 1$是一个步大小参数。实际中，通常例如0.01的小值。这称为**Shrinkage(收缩)**。

下面我们讨论如何求解这个子问题。这依赖于损失函数的形式。然而，这独立于弱学习器的形式。

### 16.4.2 L2boosting

假设我们使用平方误差损失。我们在$m$步的形式为
$$
L(y_{i},f_{m-1}(\mathbf{x}_{i}) + \beta\phi(\mathbf{x}_{i};\gamma)) = (r_{m} - \phi(\mathbf{x}_{i};\gamma))^{2} \tag{16.36}
$$
其中$r_{im}\triangleq y_{i}-f_{m-1}(\mathbf{x}_{i})$是当前的残差，其中不失一般性的我们设定为$\beta=1$。因此我们可以利用弱学习器预测$r_{m}$来寻找新的基函数。这称为**L2boosting**，或是**least square boosting**。在16.4.6节中，将看到这种方法，在适当选择弱学习者的情况下，可以得到与LARS相同的结果，后者可以用于执行变量选择。

### 16.4.3 AdaBoost

考虑一个指数损失的二分类问题。在$m$步我们最小化
$$
L_{m}(\phi)=\sum_{i=1}^{N}\exp[-\tilde{y}_{i}(f_{m-1}(\mathbf{x}_{i})+\beta\phi(\mathbf{x}_{i}))] = \sum_{i=1}^{N}w_{i,m}\exp(-\beta\tilde{y}_{i}\phi(\mathbf{x}_{i})) \tag{16.37}
$$
其中$w_{i,m}\triangleq \exp(-\tilde{y}_{i}(f_{m-1}(\mathbf{x}_{i}))$是一个应用到数据i的权重，且$y_{i}\in \{-1, +1\}$。我们将这个目标函数重写为
$$
\begin{aligned}
    L_{m} &= e^{-\beta}\sum_{\bar{y}_{i}=\phi(\mathbf{x}_{i})}w_{i,m} + e^{\beta} \sum_{\bar{y}_{i}\neq \phi(\mathbf{x}_{i})}w_{i,m} \\
    & = (e^{\beta} - e^{-\beta})\sum_{i=1}^{N}w_{i,m}\mathbb{I}(\tilde{y}_{i}\neq \phi(\mathbf{x}_{i})) + e^{-\beta}\sum_{i=1}^{N}w_{i,m} \tag{16.39}
\end{aligned}
$$
因此，要添加的最优函数
$$
\phi_{m}=\argmin_{\phi}w_{i,m}\mathbb{I}(\tilde{y}_{i}\neq \phi(\mathbf{x}_{i})) \tag{16.40}
$$
这个可以通过对数据集的加权版本应用一个弱学习器发现，权重为$w_{i,m}$。将$\phi_{m}$提交到$L_{m}$，求解$\beta$我们发现
$$
\beta_{m}=\frac{1}{2}\log\frac{1-\text{err}_{m}}{\text{err}_{m}}
$$
其中
$$
\text{err}_{m} = \frac{\sum_{i=1}^{N}w_{i}\mathbb{I}(\tilde{y}\neq\phi_{m}(\mathbb{x}_{i}))}{\sum_{i=1}^{N}w_{i,m}}
$$
总体的更新变成了
$$
\begin{aligned}
    f_{m}(\mathbf{x})=f_{m-1}(\mathbb{x})+\beta_{m}\phi_{m}(\mathbf{x})  \tag{16.43}
\end{aligned}
$$
有了这个下一次权重的迭代变为了
$$
\begin{aligned}
    w_{i,m+1}&=w_{i,m}e^{-\beta_{m}\tilde{y}_{i}\phi_{m}\mathbf{x}_{i}} \\
    &= w_{i,m}e^{\beta_{m}(2\mathbb{I}(\tilde{y}_{i}\neq\phi_{m}\mathbf{x}_{i}) - 1)} \\
    & = w_{i,m}e^{2\beta_{m}\mathbb{I}(\tilde{y}_{i}\neq\phi_{m}\mathbf{x}_{i})}e^{-\beta_{m}}
\end{aligned}
$$
其中我们利用了事实如果$\tilde{y}_{i}=\phi_{m}(\mathbf{x}_{i})$那么$-\tilde{y}_{i}\phi_{m}(\mathbf{x}_{i})=-1$，否则$-\tilde{y}_{i}\phi_{m}(\mathbf{x}_{i})=+1$。因为$e^{-\beta_{m}}$将会抵消归一化步骤，我们可以将其丢弃。

这个算法的一个真实例子如图16.10中所示，使用决策桩作为弱学习器。我们看到在许多步的迭代后，我们可以“划出”一个非常复杂的决策边界。令人惊讶的是，AbaBoost非常缓慢的过度拟合，，如果图16.8。


### 16.4.5 Boosting as functional gradient descent

没必要为每个不同的损失函数开发一个新的版本的boosting，可以推导一个泛型的版本，称为**梯度提升Gradient Boosting**。为了解释这个，想象最小化
$$
\hat{\mathbf{f}} = \argmin_{f} L(\mathbf{f})
$$
其中$\mathbf{f}=(f_{\mathbf{x}_{1}}, \cdots, f_{\mathbf{x}_{N}})$是参数。我们使用梯度下降的方法逐段求解这个问题。在步骤$m$，令$g_{m}$是$L(\mathbf{f})$在$\mathbf{f}=\mathbf{f}_{m-1}$计算的梯度：
$$
g_{im}=\left[ \frac{\partial L(y_{i},f(\mathbf{x}_{i}))}{\partial f(\mathbf{x}_{i})} \right] _{f=f_{m-1}}
$$
一些常用损失函数的梯度在表16.1中给定。我们令更新为
$$
\mathbf{f}_{m} = \mathbf{f}_{m-1}-\rho_{m}\mathbf{g}_{m}
$$
其中$\rho_{m}$是步长，
$$
\rho_{m}=\argmin_{\rho} L(\mathbf{f}_{m-1}-\rho \mathbf{g}_{m})
$$
称为函数梯度下降。

由于当前形式下，只在固定的N个点集合优化f，很少使用，所以我们不是学习一个泛化的函数。然而，我们可以通过拟合一个弱学习来修改算法去近似负梯度信号。我们使用更新
$$
\gamma_{m} = \argmin_{\gamma} \sum_{i=1}^{N}(-g_{im}-\phi(\mathbf{x}_{i};\gamma))^{2}
$$


如果在算法中使用平方损失，我们会以L2Boosting。如果我们将算法应用到对数损失，我们得到一种称为BinomialBoost的算法。与LogitBoost相比这种方法的优点是不需要进行加权拟合：只需要将任何黑盒回归模型应用于梯度向量。

### 16.4.6 稀疏Boosting

假设我们使用弱学习执行如下算法：搜索所有可能为1的变量$j=1:D$，挑选一个可以最好预测残差向量的$j(m)$
$$
\begin{aligned}
    j(m)&=\argmin_{j}\sum_{i=1}^{N}(r_{im}-\hat{\beta}_{jm}x_{ij})^{2} \\
    \bar{\beta}_{jm} &= \frac{\sum_{i=1}^{N}x_{ij}r_{im}}{\sum_{i=1}^{N}x_{ij}^{2}} \\
     \phi_{m}(\mathbf{x}) &= \bar{\beta}_{j(m),m}x_{j(m)}
\end{aligned}
$$
这种方法称为稀疏提升，与匹配追踪算法相同。

很明显，我们将得到一个稀疏估计咋，如果M很小至少是这样的。为了看这个，我们将更新重写为如下
$$
\begin{aligned}
    \mathbb{\beta}_{m} := \mathbf{\beta} _{m-1} + \nu(0,\cdots, 0,\bar{\beta}_{j(m),m},0,\cdots,0)
\end{aligned}
$$
其中非零项发生在$j(m)$。这称为前向分段线性回归，与LAR等效。

### 16.4.8 为什么Boosting如此有效

主要有两个原因。首先，可以被看作一个$\mathscr{l}_{1}$正则化，可以通过删除不相关特征来防止过拟合。为了观察这个，想象提前计算所有的弱学习器，顶一个形式如下的特征向量$\phi_{\mathbf{x}}=[\phi_{1}(\mathbf{x}),\cdots,\phi_{K}(\mathbf{x})]$，我们将使用$\mathscr{l}_{1}$正则化选择一个子集。另外，我们在每步使用Boosting，弱学习器会迅速创建一个新的$\phi_{k}$。可能将Boosting与$\mathscr{l}_{1}$正则化进行组合得到一个称为**L1-Adaboost**的算法。本质上，这种方法使用Boosting贪婪的添加最优特征，然后使用$\mathscr{l}_{1}$正则化剪枝不相关的特征。

另一个解释是，与margin的概念有关。证明AdaBoost在训练数据上最大化了裕度。

### 16.4.9 贝叶斯观点

到目前为止，我们展示的Boosting都是频率派的，因为其主要关注贪婪的最小化损失函数。这个算法的一个似然解释的主要思想是考虑一个形式为
$$
p(y|\mathbf{x},\mathbf{\theta}) = \sum_{m=1}^{M}\pi_{m}p(y|\mathbf{x},\mathbf{\gamma}_{m})
$$
的专家混合模型，其中$p(y|\mathbf{x},\mathbf{\gamma}_{m})$是一个弱学习器。我们通常使用EM算法一次性拟合所有的M个专家，但是我们想象一个顺序方案，因此我们每次只能更新一个专家。在E步，后验概率将反应现存专家如果解释一个给定数据点。如果是一个差拟合的话，这些数据对下一个要拟合的专家的影响更大。(这个观点自然建议为非监督学习使用一个类boost的算法：我们简单的顺序拟合混合模型，而不是专家混合)。

注意到，这其实是一个残损MLE方法，因为其永远不会反向回去更新一个弱学习器，这么做的唯一方式是以一个新的权重添加一个弱学习器。

## 16.5 前向神经网络(FeedForward Nueral Networks(MLP))

一个前向神经网络是一系列相互叠加的logistics回归模型，最后一层是另一个logistics回归模型或是一个线性回归模型，依赖于问题是求解一个分类还是回归问题。例如，如果我们有两层，我们求解一个回归问题，模型的形式为
$$
\begin{aligned}
    p(y|\mathbf{x},\mathbf{\theta}) & = \mathcal{N}(y|\mathbf{w}^{T}\mathbf{z}(\mathbf{x}), \sigma^{2}) \\
    \mathbf{z}(\mathbf{x}) &= g(\mathbf{V}\mathbf{x}) = [g(\mathbf{v}_{1}^{T}\mathbf{x}, \cdots, \mathbf{v}_{H}^{T}\mathbf{x})]
\end{aligned}
$$
其中$g(\cdot)$是非线性激活函数或是转换函数，$\mathbf{z}(\mathbf{x})=\phi(\mathbf{x},\mathbf{V})$是隐层。H是隐单元的数目，$\mathbf{V}$是从输入到隐单元的加权矩阵，且$\mathbf{w}$是从隐单元到输出的加权向量。g是非线性是非常重要的，要不然模型将会陷入一个形如$y=\mathbf{w}^{T}(\mathbf{V}\mathbf{x})$。可以证明MLP是一个通用逼近器，给定足够的隐层以后可以以任意的精度拟合任何的平滑函数。

为了处理2分类情况，我们将输入传递给一个sigmoid函数，例如在一个GLM中
$$
p(y|\mathbf{x,\theta}) = \text{Ber}(y|\text{sigm}(\mathbf{w}^{T}\mathbf{z}(\mathbf{x})))
$$
我们可以将这个可以扩展到MLP来预测多输出。例如，在一个回归情况中我们有
$$
p(\mathbf{y}|\mathbf{x,\theta}) = \mathcal{N}(\mathbf{y}|\mathbf{W}\phi(\mathbf{x},\mathbf{V}),\sigma^2\mathbf{I})
$$

### 16.5.1 卷积神经网络

隐单元的目标是学习原始输入的线性组合；这称为**特征提取**或是**特征构建**。这些隐单特征然后传入最终的GLM。**这种方法特别适用于原始输入特性不是很具体的问题。**相反，类似于使用一包单词处理文本分类等问题，每个特征是关于自身的有信息的，所以 抽取更高阶的特征不是那么重要。

### 16.5.2 其他类型的神经网络

除了上述拓扑之外，也可能存在其他网络拓扑。例如，我们可以跳过直接从输入到输出的圆弧，跳过隐层层；我们可以在层之间有稀疏连接。等等。然而，MLP总是要求权重形成有向无环图。如果我们允许反馈连接，模型称为RNN；这个定义另了另一种非线性动态系统，但是没有一个简单的概率解释。像RNN这样的模型是目前最好的处理语言的模型。

### 16.5.4 反向传播算法

不像GLM，MLP的NLL是一个参数的非凸函数。尽管如此，我们仍然可以使用标准的基于梯度的优化方法找到一个局部的最优ML或一个MAP估计。由于MLPs有大量的参数，经常需要大量的数据来训练。因此，通常使用一阶在线方法，如随机梯度下降，而GLM通常使用IRLS，这是一个二阶的离散方法。

我们现在讨论如何应用积分的链式法则来计算NLL的梯度向量。得到的算法称为反向传播。

出于符号简化，我们假设模型只有一层隐层。区分一个神经元的突触前值与后值是很有用的，也就是说，在应用非线性之前与之后。令$\mathbf{x}_{n}$是第$n$个输入，$\mathbf{a}_{n} = \mathbf{V}\mathbf{x}_{n}$是前突触隐层，$\mathbf{z}_{n}=g(\mathbf{a}_{n})$好后突触隐层，令$g$为转换函数。我一般令$g(a)=\text{sigm}(a)$，也可令$g(a)=\text{tanh}(a)$。

我们现在将隐层转换到输出层。令$\mathbf{b}_{n} = \mathbf{W}\mathbf{z}_{n}$是前突输出层，且$\hat{\mathbf{y}}_{n} = h(\mathbf{b}_{n})$是后突输出层，其中$h$是另一个非线性变换，对应于GLM的标准链接。对于一个回归模型，我们使用$h(\mathbf{b}) = \mathbf{b}$；对于二分类问题，我们使用$h(\mathbf{b}) = [\text{sigm}(b_{1}), \cdots, \text{sigm}(b_{c})]$；对于多分类问题，我们使用$h(\mathbf{b}) = \mathcal{S}(\mathbf{b})$。

我们将所有的模型重写为如下：
$$
\mathbf{x}_{n} \stackrel{\mathbf{V}}{\longrightarrow} \mathbf{a}_{n} \stackrel{g}{\longrightarrow} \mathbf{z}_{n} \stackrel{\mathbf{\mathbf{W}}}{\longrightarrow} \mathbf{b}_{n} \stackrel{h}{\longrightarrow} \hat{\mathbf{y}}_{n}
$$
模型参数为$\boldsymbol{\theta}=(\mathbf{V}, \mathbf{W})$，第一阶与第二阶权重矩阵。偏离或是偏差项可以将$\mathbf{x}_{n}$以及$\mathbf{z}_{n}$固定为1来求解。

回归问题中，有$K$个输出，$NLL$由平方误差类给定
$$
J(\boldsymbol{\theta}) = \sum_{n}\sum_{k}(\hat{y}_{nk}(\boldsymbol{\theta})-y_{nk})^{2}
$$
在分类问题中，有$K$个分类时，$NLL$由交叉熵给定
$$
J(\boldsymbol{\theta}) = \sum_{n}\sum_{k}y_{nk}\log \hat{y}_{nk}(\boldsymbol{\theta})
$$

我们的任务是计算$\nabla_{\theta}J$。我们为每个$n$单独求解这个；总体的梯度是由将$n$个加和得到的，尽管我们只使用mini-batch。

我们开始考虑输出层权重。我们有
$$
\nabla_{\mathbf{w}_{k}}J_{n} = \frac{\partial J_{n}}{\partial b_{nk}} \nabla_{\mathbf{w}_{k}}b_{nk} = \frac{\partial J_{n}}{\partial b_{nk}} \nabla_{\mathbf{w}_{k}}\mathbf{z}_{n}
$$