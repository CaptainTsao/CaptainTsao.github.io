<!-- TOC -->

- [15 高斯过程(Gaussian Process)](#15-高斯过程gaussian-process)
  - [15.1 引言(Introduction)](#151-引言introduction)
  - [15.2 回归的GPs(GPs for regression)](#152-回归的gpsgps-for-regression)
    - [15.2.1 使用无噪声观测来进行预测(Predictions using noise-free observations)](#1521-使用无噪声观测来进行预测predictions-using-noise-free-observations)
    - [15.2.2 使用噪声观测的预测(Predictions using noisy observations)](#1522-使用噪声观测的预测predictions-using-noisy-observations)
    - [15.2.3 核参数的影响(Effect of the kernel parameters)](#1523-核参数的影响effect-of-the-kernel-parameters)
    - [15.2.4 估计核参数(Estimating the kernel parameters)](#1524-估计核参数estimating-the-kernel-parameters)
      - [15.2.4.1 Example](#15241-example)
      - [15.2.4.2 超参的贝叶斯推断(Bayesian inference for the hyper-parameters)](#15242-超参的贝叶斯推断bayesian-inference-for-the-hyper-parameters)
  - [15.3 GPs meet GLMs](#153-gps-meet-glms)
    - [15.3.1 二分类(Binary classification)](#1531-二分类binary-classification)
      - [15.3.1.1 计算后验(Computing the posterior)](#15311-计算后验computing-the-posterior)
      - [15.3.1.2 计算后验预测(Computing the posterior predictive)](#15312-计算后验预测computing-the-posterior-predictive)
      - [15.3.1.3 计算边缘似然(Computing the marginal likelihood)](#15313-计算边缘似然computing-the-marginal-likelihood)
      - [15.3.1.4 数值稳定计算(Numerically stable computation*)](#15314-数值稳定计算numerically-stable-computation)
    - [15.3.3 泊松回归的GPs(GPs for Poisson regression)](#1533-泊松回归的gpsgps-for-poisson-regression)
  - [15.4 与其他方法的联系(Connection with other methods)](#154-与其他方法的联系connection-with-other-methods)
    - [15.4.1 Linear models compared to GPs](#1541-linear-models-compared-to-gps)
    - [15.4.2 GPs与线性平滑器的比较(Linear smoothers compared to GPs)](#1542-gps与线性平滑器的比较linear-smoothers-compared-to-gps)
      - [15.4.2.1 线性平滑器的自由度(Degrees of freedom of linear smoothers)](#15421-线性平滑器的自由度degrees-of-freedom-of-linear-smoothers)
    - [15.4.3 SVMs与GPs的比较(SVMs compared to GPs)](#1543-svms与gps的比较svms-compared-to-gps)
    - [15.4.4 L1VM与RVMs对GPs的对比(L1VM and RVMs compared to GPs)](#1544-l1vm与rvms对gps的对比l1vm-and-rvms-compared-to-gps)
    - [15.4.5 神经网络与GPs比较(Neural networks compared to GPs)](#1545-神经网络与gps比较neural-networks-compared-to-gps)
    - [15.4.6 平滑线条与GPs的比较(Smoothing splines compared to GPs *)](#1546-平滑线条与gps的比较smoothing-splines-compared-to-gps-)
      - [15.4.6.1 单元线条(Univariate splines)](#15461-单元线条univariate-splines)
      - [15.4.6.2 回归样条(Regression splines)](#15462-回归样条regression-splines)
      - [15.4.6.3 与GPs的联系(The connection with GPs)](#15463-与gps的联系the-connection-with-gps)
      - [15.4.6.4 二维输入(2d input (thin-plate splines))](#15464-二维输入2d-input-thin-plate-splines)
      - [15.4.6.5 更高维的输入(Higher-dimensional inputs)](#15465-更高维的输入higher-dimensional-inputs)
    - [15.4.7 RKHS methods compared to GPs *](#1547-rkhs-methods-compared-to-gps-)
  - [15.5 GP latent variable model](#155-gp-latent-variable-model)

<!-- /TOC -->

# 15 高斯过程(Gaussian Process)

## 15.1 引言(Introduction)

在监督学习中，我们观测到一些输入$\mathbf{x}_{i}$或输出$y_{i}$。对于一些未知的函数$f$，我假设$y_{i} = f(\mathbf{x}_{i})$可能会被一些噪声所污染。最好的方法是在给定数据的情况下推断函数分布，$p(f\vert \mathbf{X,y})$，然后使用这个来对新的输入进行预测，即计算
$$
p(y_{*}|\mathbf{x}_{*},\mathbf{X}，\mathbf{y}) = \int p(y_{*}|f, \mathbf{x}_{*})p(f|\mathbf{X}, \mathbf{y})df       \tag{15.1}
$$
直到现在，我们都在关注函数$f$的参数化表示，因此不必推断$p(f|\mathcal{D})$，而是推断$p(\boldsymbol{\theta}|\mathcal{D})$。本章中，我们讨论一种方式在函数本身上执行贝叶斯推断的一种方式。

我们的方法是基于**高斯过程(Gaussian Processes)**或是**GPs**的。一个GP定义了函数的一个先验，有了该先验，我们看到一些数据以后就可以将其转化为后验。尽管可能表示一个函数上的分布是很难的，但是事实证明，我们只需要能在一组有限但任意点$\mathbf{x}_{1}, \cdots, \mathbf{x}_{N}$上定义函数值的分布即可。**一个`GP`假设$p(f(\mathbf{x}_{1}), \cdots, f(\mathbf{x}_{N}))$是联合高斯分布的，均值是$\boldsymbol{\mu}(\mathbf{x})$，协方差$\boldsymbol{\Sigma}$给定为$\sum_{ij}=\kappa(\mathbf{x}_{i}, \mathbf{x}_{j})$，其中$\kappa$为正定的核函数(14.2节中有关于核的信息)**。关键思想是如果$\mathbf{x}_{i}$与$\mathbf{y}_{j}$是被内核认为是相似的，然后我们期望函数在这些点的输出也是相似的。可以看图15.1。

![image](Figure15.1.png)
> 图15.1 具有2个训练点与1个测试点的高斯过程，表示为了一个混合有向与无向概率图模型，

**可以证明在回归背景中，所有这些计算是可以解析计算的，时间消耗为$O(N^{3})$。(我们将在15.6节中讨论更快的近似)在分类背景中，我们必须使用近似，例如高斯近似，因为后验不再是准确的高斯。**

`GPs`可以被看作是我们在第14章中讨论的核方法(包括`L1VM`，`RVM`以及`SVM`)的一种贝叶斯替代。尽管这些方法较稀疏，因此速度较快，但它们并未提供经过良好校准的概率输出(15.4.4节给出深入讨论)。有确定的**调节概率输出(tuned probabilistic output)** 在确定的应用中是很重要的，例如在线视觉追踪，机器人，**强化学习以及最优控制**，非凸函数的全局优化，实验设计。


## 15.2 回归的GPs(GPs for regression)

本节中，我们讨论回归的`GPs`。令在回归函数上的**先验**是一个GP，记为
$$
f(\mathbf{x}) \sim GP(m(\mathbf{x}), \kappa(\mathbf{x}, \mathbf{x}^{\prime}))  \tag{15.2}
$$
其中$m(\mathbf{x})$是均值函数，$\kappa(\mathbf{x}, \mathbf{x}^{\prime})$是**核函数或协方差函数**，也就是：
$$
\begin{aligned}
    m(\mathbf{x}) &= \mathbb{E}[f(\mathbf{x})]  \\ 
    \kappa(\mathbf{x}, \mathbf{x}^{\prime}) & = \mathbb{E}[(f(\mathbf{x}) - m(\mathbf{x}))(f(\mathbf{x}^{\prime})-m(\mathbf{x}^{\prime}))] \tag{15.4}
\end{aligned}
$$

很明显，我们要求$\kappa(\mathbf{x}, \mathbf{x}^{\prime})$是一个正定的核。对于任何有限的点集，这个过程称为一个联合高斯
$$
p(\mathbf{f}|\mathbf{X}) = \mathcal{N}(\mathbf{f} | \boldsymbol{\mu}, \mathbf{K}) \tag{15.5}
$$
其中$K_{ij}=\kappa(\mathbf{x}_{i}, \mathbf{x}_{j})$且$\boldsymbol{\mu}=(m(\mathbf{x}_{1}),\cdots,m(\mathbf{x}_{N}))$。

注意到，通常使用一个$m(\mathbf{x})=0$的均值函数，因为`GP`足够灵活来任意建模均值。**然而，在15.2.6节中我们将考虑一个均值函数的参数模型，因此`GP`只需要对残差进行建模**。**这个半参数模型将参数模型的可解释性与非参数模型的准确性结合了起来**。

### 15.2.1 使用无噪声观测来进行预测(Predictions using noise-free observations)

假设我们观测到一个训练集$\mathcal{D}=\{(\mathbf{x}_{i}, f_{i}), i=1:N\}$，其中$f_{i} = f(\mathbf{x}_{i})$是在点$\mathbf{x}_{i}$处计算的函数的**无噪声**观测值。给定一个大小为$N_{*}\times D$测试集$\mathbf{X}_{*}$，我们想预测函数输出$\mathbf{f}_{*}$。

如果我们要求`GP`为已观测的一个值$\mathbf{x}$预测$f(\mathbf{x})$，我们希望`GP`返回确定的答案$f(\mathbf{x})$。换而言之，**它应该作为训练数据的插值器**。只有当我们假设观测是无噪声的时候才会发生。我们考虑无噪声观测的情况。

我们现在返回到预测问题。通过定义`GP`，联合分布的形式如下：
$$
\begin{pmatrix}
    \mathbf{f} \\ \mathbf{f}_{*} 
\end{pmatrix} \sim
\begin{pmatrix}
    \begin{pmatrix}
        \boldsymbol{\mu} \\ \boldsymbol{\mu}_{*}
    \end{pmatrix} , 
    \begin{pmatrix}
        \mathbf{K} & \mathbf{K}_{*} \\
        \mathbf{K}_{*}^{T} & \mathbf{K}_{**}
    \end{pmatrix}
\end{pmatrix} \tag{15.6}
$$
其中$\mathbf{K}=\kappa(\mathbf{X}, \mathbf{X})$是一个$N\times N$的矩阵，$\mathbf{K}_{*} = \kappa(\mathbf{X}, \mathbf{X}_{*})$是一个$N\times N_{*}$的矩阵，$\mathbf{K}_{**}$是$N_{*}\times N_{*}$。通过条件高斯的标准准则，后验有如下形式
$$
\begin{aligned}
    p(\mathbf{f}_{*} | \mathbf{X}_{*}, \mathbf{X}, \mathbf{f}) &= \mathcal{N}(\mathcal{f}_{*}|\boldsymbol{\mu}_{*}, \boldsymbol{\Sigma}_{*}) \\
    \boldsymbol{\mu}_{*} &= \boldsymbol{\mu}(\mathbf{X}_{*}) + \mathbf{K}_{*}^{T}\mathbf{K}^{-1}(\mathbf{f}-\boldsymbol{\mu}(\mathbf{X}))  \\
    \boldsymbol{\Sigma_{*}} &= \mathbf{K}_{**} - \mathbf{K}_{*}^{T}\mathbf{K}^{-1}\mathbf{K}_{*} \tag{15.7-15.9} 
\end{aligned}
$$

![image](Figure15.2.png)

这个过程是在图15.2中解释的。左边我们看到来自先验的采样$p(\mathbf{f\vert X})$，其中我们使用一个**平方指数核**，类似于`高斯核或RBF核`。在1维空间中，给定为
$$
\kappa(x,x^{\prime}) = \sigma^2_f\exp(-\frac{1}{2\ell^2}(x-x^{\prime})^2)       \tag{15.10}
$$
这里$\ell$控制函数变化的水平长度刻度，$\sigma^2_f$控制了垂直变化。左边我们显示了来自后验的采样，$p(\mathbf{f_*\vert X_*,X,f})$。我们看到模型完美地插值了训练数据，并且随着我们离观测数据的距离越来越远，预测不确定性也会增加。

无噪声GP回归的一个应用是作为复杂模拟器（如天气预报程序）行为的廉价计算代理。（如果模拟器是随机的，我们可以将f定义为其平均输出；请注意，仍然没有观测噪声。）然后，可以通过检查模拟器参数对GP预测的影响来估计其影响，而不必多次运行模拟器，这可能会慢得让人望而却步。这种策略被称为DACE，它代表**计算机实验的设计和分析**（Santner等人，2003）

### 15.2.2 使用噪声观测的预测(Predictions using noisy observations)

我们现在考虑我们观测的是一个基本函数的噪声版本$y=f(\mathbf{x})+\epsilon$，其中$\epsilon\sim\mathcal{N}(0,\sigma_y^2)$。这种情况下，不需要模型对数据进行插值，但是模型必须与观测数据接近。观测噪声响应的协方差为
$$
\text{cov}[y_p,y_q]=\kappa(\mathbf{x}_p,\mathbf{x}_q)+\sigma_y^2\delta_{pq}     \tag{15.11}
$$
其中$\delta_{pq}=\mathbb{I}(p=q)$。换而言之，
$$
\text{cov}[\mathbf{y\vert X}]=\mathbf{K}+ \sigma_y^2 \mathbf{I}_N \triangleq \mathbf{K}_y   \tag{15.12}
$$
**第二个矩阵为对角阵，因为我们假设噪声项是独立添加到每个观测值**。

**观测数据**以及在测试点的**隐无噪声函数**的联合密度由下式给出
$$
\begin{aligned}
    \begin{pmatrix}
        \mathbf{y} \\
        \mathbf{f}_* 
    \end{pmatrix} \sim \begin{pmatrix}
        \mathbf{0}, \begin{pmatrix}
            \mathbf{K}_y & \mathbf{K}_* \\
            \mathbf{K}_*^T & \mathbf{K}_{**}
        \end{pmatrix}
    \end{pmatrix}
\end{aligned}\tag{15.13}
$$
出于符号简化，其中我假设均值为零。因此**后验预测密度**为
$$
\begin{aligned}
    p(\mathbf{f_*\vert X_*,X,y}) &= \mathcal{N}(\mathbf{f_*}\vert \boldsymbol{\mu}_*,\mathbf{\Sigma}_*) \\
    \boldsymbol{\mu}_* &= \mathbf{K}_*^T \mathbf{K}_y^{-1} \mathbf{y} \\
    \mathbf{\Sigma}_* &= \mathbf{K}_{**} - \mathbf{K}_*^T \mathbf{K}_y^{-1} \mathbf{K}_* 
\end{aligned}
$$
在单个测试输入时，可以简化为
$$
p(f_*\vert\mathbf{ x_*,X,y}) = \mathcal{N}(f_* \vert \mathbf{k}_*^T \mathbf{K}_y^{-1} \mathbf{y},  k_{**} - \mathbf{k}_*^T \mathbf{K}_y^{-1} \mathbf{k}_* )
$$
其中$\mathbf{k}_*=[\kappa(\mathbf{x}_*,\mathbf{x}_1),\cdots,\kappa(\mathbf{x}_*,\mathbf{x}_N)]$且$k_{**}=\kappa(\mathbf{x}_*,\mathbf{x}_*)$。**后验均值**的另一种方法如下
$$
\bar{f}_* = \mathbf{k}_*^T \mathbf{K}_y^{-1} \mathbf{y} = \sum_{i=1}^N\alpha_i\kappa(\mathbf{x}_i,\mathbf{x}_*)     \tag{15.18}
$$
其中$\boldsymbol{\alpha}=\mathbf{K}_y^{-1}\mathbf{y}$。

![image](Figure15.3.png)
> 图15.3: 一些有SE核但是不同超参的的GPs对20个噪声观测的拟合。超参$(\ell, \sigma_f, \sigma_y)$如下(a):(1,1,0.1),(b)(0.3, 0.1, 0.8,),(c)(3.0, 1.16, 0.89)

### 15.2.3 核参数的影响(Effect of the kernel parameters)

GPs的预测性能完全取决于所选内核的适用性。假设，我们为噪声观测选择如下的平方指数核
$$
\kappa_{y}(x_p,x_q) = \sigma_f^2 \exp\left( -\frac{1}{2\ell^2} \left(x_p-x_q \right )^2 \right) + \sigma_y^2\delta_{pq}     \tag{15.19}
$$
这里$\ell$是函数变化的水平比例，$\sigma^2_f$控制了函数的垂直比例，$\sigma_y^2$是噪声方差。图15.3解释了改变这些参数的影响。我们使用$(\ell,\sigma_f,\sigma_y)=(1,1,0.1)$从SE核中采样了20个噪声数据点，然后以这些数据为条件使用各种参数做出预测。在图15.3(a)中，我们使用$(\ell,\sigma_f,\sigma_y)=(1,1,0.1)$，且结果为一个很好的拟合。图15.3(b)中，我们将长度缩放减少至0.3(其他参数通过最大化似然来优化)；现在函数看起来更加"wiggly"。而且，由于距训练点的有效距离增加得更快，不确定性上升得更快。在图15.3(c)中，我们将长度比例增加到$\ell = 3$；现在该功能看起来更加平滑。

我们可以经SE核扩展到多维
$$
\kappa_y(\mathbf{x}_p,\mathbf{x}_q) = \sigma_f^2\exp\left( -\frac{1}{2}(\mathbf{x}_p-\mathbf{x}_q)^T\mathbf{M}(\mathbf{x}_p-\mathbf{x}_q) \right) + \sigma_y^2\delta_{pq}       \tag{15.20}
$$

![image](Figure15.4.png)
我们可以使用几种方式定义矩阵$\mathbf{M}$。最简单的方式就是使用一个同向矩阵$\mathbf{M}_1=\ell^{-2}\mathbf{I}$。如图15.4(a)作为一个例子。我们还可以赋予每个维度自己的特征长度比例，$\mathbf{M}_2=\text{diag}(\boldsymbol{\ell})^{-2}$。如果这些长度缩放中的任何一个变大，相应的特征尺寸将被视为“无关紧要”，与ARD中一样(第13.7节)。如图15.4(b)中，我们使用$\mathbf{M}=\mathbf{M}_2,\ell=(1,3)$，所以函数在沿着方向$x_1$比沿着方向$x_2$变化的更快。我们也可以创建一个形式如$\mathbf{M}_3 = \boldsymbol{\Lambda\Lambda}^T + \text{diag}(\boldsymbol{\ell})^{-2}$的矩阵，其中$\boldsymbol{\Lambda}$是一个$D\times K$的矩阵，其中$K\lt D$。(Rasmussen and Williams 2006, p107)之所以称其为**因子分析距离函数**，是因为因子分析(第12.1节)将协方差矩阵近似为低秩矩阵和对角矩阵。$\boldsymbol{\Lambda}$的列对应着输入空间的相关方向。在图15.4(c)中，我们使用$\boldsymbol{\ell}=(6;6)$与$\boldsymbol{\Lambda}=(1;-1)$，所以函数改变最快的方向是与$(1,1)$垂直的。

### 15.2.4 估计核参数(Estimating the kernel parameters)

为了估计核参数，我们可以在值的离散网格上使用穷举搜索，并以验证损失为目标，但这可能会很慢。(这个方法用来调节SVMs的参数)这里我们考虑一个经验贝叶斯方法，将允许我们使用连续的优化方法，这个会更快一点。特别的是，我们将最大化边缘似然[^1]
$$
p(\mathbf{y\vert X}) = \int p(\mathbf{y\vert f,X})p(\mathbf{f\vert X})d\mathbf{f}       \tag{15.21}
$$
因为$p(\mathbf{f\vert X})=\mathcal{N}(\mathbf{f,\vert 0,K})$且$p(\mathbf{y\vert f})=\prod_{i}\mathcal{N}(y_i\vert f_i, \sigma_y^2)$，边缘似然给定为
$$
\log p(\mathbf{y\vert X}) = \log \mathcal{N}(\mathbf{y\vert 0,K}_y) =-\frac{1}{2}\mathbf{yK}_y^{-1}\mathbf{y} - \frac{1}{2}\log\lvert \mathbf{K}_y\rvert - \frac{N}{2}\log(2\pi)    \tag{15.22}
$$
第一项是数据拟合项，第二项是模型复杂度项，第三项只是一个常数。为了理解前两项之间的平衡，考虑一个$1$维中的SE核，因为我们保持$\sigma_y^2$不变而变化长度比例$\ell$。令$J(\ell)=-\log p(\mathbf{y\vert X,\ell})$。对于短的长度比例，拟合会好点，所以$\mathbf{y}^T\mathbf{K}_y^{-1}\mathbf{y}$将会很小。然而，模型复杂度将会很高：
$\mathbf{K}$将会是对角的，因为大多数点不会被认为是“接近”其他点的，所以$\log\lvert\mathbf{K}_y\rvert$将会很大。对于长的尺度比例，拟合不会太好但是模型复杂度将会较低：$\mathbf{K}$将会是全$1$，所以$\log\lvert\mathbf{K}_y\rvert$将会很小。

我们现在讨论如何最大化边际似然。令核参数(也称为超参数)记为$\boldsymbol{\theta}$。可以证明
$$
\begin{aligned}
    \frac{\partial}{\partial\theta_j}\log p(\mathbf{y\vert X})&= \frac{1}{2}\mathbf{y}^T\mathbf{K}_y^{-1}\frac{\partial\mathbf{K}_y}{\partial\theta_j}\mathbf{K}_y^{-1}\mathbf{y} - \frac{1}{2}\text{tr}\left( \mathbf{K}_y^{-1}\frac{\partial\mathbf{K}_y}{\partial\theta_j} \right) \\
    &= \frac{1}{2} \text{tr}\left( (\boldsymbol{\alpha\alpha}^T-\mathbf{K}_y^{-1}) \frac{\partial\mathbf{K}_y}{\partial\theta_j} \right)        \tag{15.23-15.24}
\end{aligned}
$$
其中$\boldsymbol{\alpha}=\mathbf{K}_y^{-1}\mathbf{y}$。其消耗$O(N^3)$的时间来计算$\mathbf{K}_y^{-1}$，如何使用$O(N^2)$时间来为每个超参计算梯度。

$\frac{\partial\mathbf{K}_y}{\partial\theta_j}$的形式依赖于核的形式，以及我们相对哪些参数求偏导。通常我们对于超参是有约束的，例如$\sigma_y^2\gt0$。在这种情况下，我们可以定义$\theta=\log(\sigma_y^2)$，如何使用链式法则。

给定一个对数似然的表达式及其导数，我们可以使用标准的基于梯度的优化器来计算核参数。然而，因为目标函数不是凸的，局部最小值可能是一个问题。

![image](Figure15.5.png)
> 图15.5 解释了边缘似然面的局部最小值。(a). 我们在固定$\sigma_f^2=1$的情况下，使用7个数据点画了边缘似然相对于$\sigma_f^2$与$\ell$。(b). 函数对应左边局部最小值,$(\ell, \sigma_n^2)\approx (1, 0.2)$，这个确实很宽松，噪声更低。(c)函数对应右上角的局部最小点，$(\ell, \sigma_n^2)\approx (10, 0.8)$。这个确实平滑但是有更多噪声。

#### 15.2.4.1 Example

考虑图15.5。我们使用方程15.19中的SE核，$\sigma_f^2=1$，并画出随着$\ell,\sigma_y^2$改变的$\log p(\mathbf{y\vert X,\ell,\sigma_y^2})$。两个局部的最优值表示为$+$。左下方的最佳值对应于一个低噪声，短长度比例的解决方案(显示在面板b中)。右上方的最佳值对应于高噪声，长标度的解决方案(显示在面板c中)。仅使用7个数据点，尽管更复杂的模型(面板b)的边际可能性比简单模型(面板c)高约60％，但没有足够的证据来自信地确定哪个更合理。有了更多数据，MAP估计值将占主导地位。

#### 15.2.4.2 超参的贝叶斯推断(Bayesian inference for the hyper-parameters)

计算一个超参数的点估计的另一种方式是计算它们的后验。令$\boldsymbol{\theta}$代表所有的核参数，以及$\sigma_y^2$。如果$\boldsymbol{\theta}$的维度很小，我们可以以$\hat{\boldsymbol{\theta}}$的MAP估计为中心，计算可能值的一个离散网格。我们然后可以使用近似隐变量的后验
$$
p(\mathbf{f\vert\mathcal{D}}) \propto \sum_{s=1}^{\mathcal{S}} p(\mathbf{f\vert\mathcal{s},\boldsymbol{\theta}_{\mathcal{s}}}) p(\boldsymbol{\theta}_{\mathcal{s}}\vert\mathcal{D})\delta_{\mathcal{s}}         \tag{15.25}
$$
其中$\delta_{\mathcal{s}}$代表网格点$\mathcal{s}$的权重。

## 15.3 GPs meet GLMs

本节中，我们将GPs扩展到GLM背景下，集中在分类情况下。例如贝叶斯logistics回归，主要难度是高斯先验与bernoulli/multinoulli似然不共轭。这里可以采用几种近似：高斯近似(8.4.3节)、期望传播，变分、MCMC等等。这里我们主要关注高斯近似，因为其是最简单的最快的。

### 15.3.1 二分类(Binary classification)

在二分类情况中，我们定义模型为$p(y_i\vert x_i)=\sigma(y_if(\mathbf{x}_i))$，其中我们假设$y_i\in\{-1,+1\}$，且我们令$\sigma(z)=\text{sigm}(z)$(logistics回归)或$\sigma(z)=\Phi(z)$(probit回归)。对于GP回归，我们假设$f\sim GP(0,\kappa)$。

#### 15.3.1.1 计算后验(Computing the posterior)

定义未归一化的后验的对数如下
$$
\begin{aligned}
    \ell(\mathbf{f}) &= \log p(\mathbf{y\vert f}) + \log p(\mathbf{f\vert X})       \\
    &= \log p(\mathbf{y\vert f}) - \frac{1}{2}\mathbf{f}^T\mathbf{K}^{-1}\mathbf{f} - \frac{1}{2}\log\lvert \mathbf{K}\rvert - \frac{N}{2}\log(2\pi)        \tag{15.33}
\end{aligned}
$$
令$J(f)\triangleq-\ell(f)$是我们想要最小化的函数。其梯度与Hessian未
$$
\begin{aligned}
    \mathbf{g} &= -\nabla\log p(\mathbf{y\vert f}) + \mathbf{K}^{-1}\mathbf{f} \\
    \mathbf{H} &= -\nabla\nabla \log p(\mathbf{y\vert f}) + \mathbf{K}^{-1} = \mathbf{W+K}^{-1} \tag{15.35}
\end{aligned}
$$
注意到$\mathbf{W}\triangleq \nabla\nabla\log p(\mathbf{y\vert f})$是一个对角矩阵，因为数据是iid的。第8.3.1节和第9.4.1节给出了logit和probit情况下对数似然的梯度和Hessian表达式，并在表15.1中进行了总结。

我们可以使用IRLS来计算MAP估计。更新的形式为
$$
\begin{aligned}
    \mathbf{f}^{new} &= \mathbf{f} - \mathbf{H}^{-1}\mathbf{g} = \mathbf{f} + (\mathbf{W+K}^{-1})^{-1}(-\nabla\log p(\mathbf{y\vert f}) + \mathbf{K}^{-1}\mathbf{f})  \\
    &= (\mathbf{W+K}^{-1})^{-1}(\mathbf{Wf}+\nabla \log p(\mathbf{y\vert f}))       \tag{15.37}
\end{aligned}
$$

收敛时，后验函数的高斯近似形式如下：
$$
p(\mathbf{f\vert X,y}) \approx \mathcal{N}(\hat{\mathbf{f}}, (\mathbf{W+K}^{-1})^{-1})      \tag{15.38}
$$

#### 15.3.1.2 计算后验预测(Computing the posterior predictive)

我们现在计算后验预测。首先，我们在检测情况$\mathbf{x}_*$下预测隐函数。对于均值，我们有
$$
\begin{aligned}
    \mathbb{E}[f_*\vert \mathbf{x}_*,\mathbf{X,y}] &= \int \mathbb{E}[f_*\vert \mathbf{f,x}_*,\mathbf{X,y}]p(\mathbf{f\vert X,y})d\mathbf{f} \\
    &= \int \mathbf{k}_*^T\mathbf{K}^{-1}\mathbf{f} p(\mathbf{y\vert X,y})d\mathbf{f} \\ 
    &= \mathbf{k}_*^T\mathbf{K}^{-1}\mathbb{E}[\mathbf{f\vert X,y}] \approx \mathbf{k}_*^T\mathbf{K}^{-1}\hat{\mathbf{f}}       \tag{15.41}
\end{aligned}
$$
我们使用方程15.8来得到给定无噪声$\mathbf{f}$的均值$f_*$.

为了计算，预测方差，我们使用迭代方差的的原则
$$
\text{var}[f_*] = \mathbb{E}[\text{var}[f_*\vert\mathbf{f}]] + \text{var}[\mathbb{E}[f_*\vert\mathbf{f}]]       \tag{15.42}
$$
其中所有的概率是以$\mathbf{x}_*，\mathbf{X,y}$为条件。根据方程15.9，我们有
$$
\mathbb{E}[\text{var}[f_*\vert\mathbf{f}]]=\mathbb{E}[k_{**}-\mathbf{k}^T_*\mathbf{K}^{-1}\mathbf{k}_*] = k_{**}-\mathbf{k}_*^T\mathbf{K}^{-1}\mathbf{k}_*      \tag{15.43}
$$
$$
\text{var}[\mathbb{E}[f_*\vert\mathbf{f}]] = \text{var}[\mathbf{k}_*\mathbf{K}^{-1}\mathbf{f}] = \mathbf{k}_*^T\mathbf{K}^{-1}\text{cov}[\mathbf{f}]\mathbf{K}^{-1}\mathbf{k}_*     \tag{15.44}
$$
将这些结合起来有
$$
\text{var}[f_*] = k_{**} - \mathbf{k}_*^T(\mathbf{K}^{-1} - \mathbf{K}^{-1}\text{cov}[\mathbf{f}]\mathbf{K}^{-1})\mathbf{k}_*       \tag{15.45}
$$
从方程15.38，我们有$\text{cov}[f]\approx (\mathbf{W+K}^{-1})^{-1}$。使用矩阵可逆理论，我们得到
$$
\begin{aligned}
    \text{var}[f_*] &\approx k_{**} - \mathbf{k}_*^T(\mathbf{K}^{-1} - \mathbf{K}^{-1}(\mathbf{W+K}^{-1})^{-1} \mathbf{K}^{-1})\mathbf{k}_*  \\
    & = k_{**} - \mathbf{k}_*^T (\mathbf{W} + \mathbf{K}^{-1})^{-1} \mathbf{k}_*
\end{aligned}
$$
总之，我们有
$$
p(f_*\vert \mathbf{x}_*,\mathbf{X,y}) = \mathcal{N}(\mathbb{E}[f_*],\text{var}[f_*])
$$
为了将转换为二元响应的预测分布，我们使用
$$
\pi_* = p(y_*=1\vert \mathbf{x}_*,\mathbf{X,y}) \approx \int \sigma(f_*)p(f_* \vert \mathbf{x}_*,\mathbf{X,y}) df_*     \tag{15.49}
$$
可以使用第8.4.4节讨论贝叶斯逻辑回归的任何方法来近似。例如，使用8.4.4.2节的概率近似，我们有$\pi_*\approx\text{sigm}(\kappa(v))\mathbb{E}[f_*]$，其中$v=\text{var}[f_*]$。

#### 15.3.1.3 计算边缘似然(Computing the marginal likelihood)

我们需要边缘似然而优化核参数。使用方程8.54中的Laplace近似，我们有
$$
\log p(\mathbf{y\vert X}) \approx \ell(\hat{\mathbf{f}}) - \frac{1}{2}\log\lvert\mathbf{H}\rvert + \text{const} \tag{15.50}
$$
因此
$$
\log p(\mathbf{y\vert X}) \approx \log p(\mathbf{y}\vert \hat{\mathbf{f}}) - \frac{1}{2}\hat{\mathbf{f}}^T\mathbf{K}^{-1}\hat{\mathbf{f}} - \frac{1}{2}\log\lvert\mathbf{K}\rvert  - \frac{1}{2}\log\lvert\mathbf{K}^{-1} + \mathbf{W} \rvert \tag{15.51}
$$
计算导数$\frac{\partial p(\mathbf{t\vert X},\boldsymbol{\theta})}{\partial \theta_j}$比在回归中更加复杂，因为$\hat{\mathbf{f}}，\mathbf{W}$以及$\mathbf{K}$依赖于$\boldsymbol{\theta}$。

#### 15.3.1.4 数值稳定计算(Numerically stable computation*)

为了以一种数值稳定的方式应用上述方程，最好避免求逆$\mathbf{K}$或$\mathbf{W}$。(Rasmussen and Williams 2006, p45)证明定义
$$
\mathbf{B}=\mathbf{I}_N + \mathbf{W^{\frac{1}{2}}KW^{\frac{1}{2}}}\tag{15.52}
$$
其特征值边界低于1(因为有$\mathbf{I}$)，且高于$1+\frac{N}{4}\max_{ij}K_{ij}$，因此可以安全的求逆。

可以使用矩阵逆定理
$$
(\mathbf{K}^{-1}+\mathbf{W})^{-1} = \mathbf{K}-\mathbf{KW}^{\frac{1}{2}}\mathbf{B}^{-1}\mathbf{K}^{\frac{1}{2}} \mathbf{K}     \tag{15.53}
$$
因此IRILS更新变为
$$
\begin{aligned}
    \mathbf{f}^{new} &= (\mathbf{K}^{-1}+\mathbf{W})^{-1}\underbrace{(\mathbf{Wf}+\nabla\log p(\mathbf{y\vert f}))}_{\mathbf{b}} \\
    &= \mathbf{K}(\mathbf{I}-\mathbf{W}^{\frac{1}{2}}\mathbf{B}^{-1}\mathbf{W}^{\frac{1}{2}} \mathbf{K})\mathbf{b}  \\
    &= \mathbf{K}\underbrace{(\mathbf{b}-\mathbf{W}^{\frac{1}{2}}\mathbf{L}^T \setminus (\mathbf{L} \setminus (\mathbf{W}^{\frac{1}{2}} \mathbf{K} \mathbf{b} )))}_{\mathbf{a}}
\end{aligned}\tag{15.56}
$$
其中$\mathbf{B}=\mathbf{LL}^T$是$\mathbf{B}$的契比雪夫分解。拟合算法消耗时间为$O(TN^3)$空间消耗为$O(N^2)$，其中$T$是牛顿迭代次数。

在收敛处，我们有$\mathbf{a=K}^{-1}\hat{\mathbf{f}}$，所以我们可以使用
$$
\log p(\mathbf{y\vert X}) = \log p(\mathbf{y}\vert\hat{\mathbf{f}})-\frac{1}{2}\mathbf{a}^T\hat{\mathbf{f}} - \sum_{i}\log L_{ii}   \tag{15.57}
$$
计算边缘似然，其中利用的事实为
$$
\lvert\mathbf{B}\rvert = \lvert\mathbf{K}\rvert\lvert\mathbf{K}^{-1}+\mathbf{W}\rvert = \lvert\mathbf{I}_{N} + \mathbf{W}^{1/2}\mathbf{K}\mathbf{W}^{1/2}\rvert     \tag{15.58}
$$

### 15.3.3 泊松回归的GPs(GPs for Poisson regression)

本节，我们将解释泊松回归的GPs。该方法的一个有趣的应用就是空间疾病映射。例如，讨论了建模在Finland不同区域心脏病的突发的相对风险的问题。数据由Finland从1996-2000的心脏病发生的数据。模型如下
$$
y_i \sim \text{Poi}(e_ir_i)     \tag{15.74}
$$

## 15.4 与其他方法的联系(Connection with other methods)

在统计学以及机器学习中，有大量的方法与GP回归/分类相关。我们下面给出一些简单的解释。

### 15.4.1 Linear models compared to GPs

考虑D-维特征的贝叶斯线性回归，其中权重的先验为$p(\mathbf{w})=\mathcal{N}(\mathbf{0,\Sigma})$。后验预测分布给定为如下
$$
\begin{aligned}
    p(f_*\vert \mathbf{x}_*,\mathbf{X,y}) &= \mathcal{N}(\mu,\sigma^2)  \\
    \mu &= \frac{1}{\sigma_y^2}\mathbf{x}_*^T\mathbf{A}^{-1}\mathbf{X}^T\mathbf{y} \\
    \sigma^2 &= \mathbf{x}_*^T\mathbf{A}^{-1}\mathbf{x}_*
\end{aligned}   \tag{15.75-15.77}
$$
其中$\mathbf{A}=\sigma_y^{-2}\mathbf{X}^T\mathbf{X}+\mathbf{\Sigma}^{-1}$。可以证明我们可以将上述分布重写为
$$
\begin{aligned}
    \mu &= \mathbf{x}_*^T\mathbf{\Sigma X}^T(\mathbf{K}+\sigma_y^2\mathbf{I})^{-1}\mathbf{y} \\
    \sigma^2 &= \mathbf{x}_*^T\mathbf{\Sigma}\mathbf{x}_* - \mathbf{x}_*^T\mathbf{\Sigma X}^T(\mathbf{K}+\sigma^2\mathbf{I})^{-1}\mathbf{X\Sigma x}_*
\end{aligned}
$$
其中我们定义$\mathbf{K}=\mathbf{X\Sigma X}^T$，其大小为$N\times N$。因为特征只是以形式$\mathbf{X\Sigma X}^T,\mathbf{x}_*^T\mathbf{\Sigma X}^T$存在，我们可以通过定义$\kappa(\mathbf{x},\mathbf{x}^{\prime})=\mathbf{x}^T\mathbf{\Sigma}\mathbf{x}^{\prime}$核化上述表达式。

那么，我们看到**贝叶斯线性回归**等价于协方差函数为$\kappa(\mathbf{x},\mathbf{x}^{\prime})=\mathbf{x}^T\mathbf{\Sigma x}^{\prime}$的一个GP。然而，注意到这是一个**退化**协方差函数，因为其有最多$D$个非零特征值。直觉上，这反应了一个事实，模型只可以表示为有限个函数。这导致了欠拟合，因为模型不够灵活，无法捕获数据。更糟糕的是，它可能会导致过度自信，因为模型的先验信息非常贫乏，以至于它的后验函数会变得过于集中。所以模型不仅是错的，而且认为是对的！

### 15.4.2 GPs与线性平滑器的比较(Linear smoothers compared to GPs)

线性平滑器是一种回归函数，其是训练输出的一个线性函数
$$
\hat{f}(\mathbf{x}_*) = \sum_i w_i(\mathbf{x}_*)y_i     \tag{15.80}
$$
其中$w_i(\mathbf{x}_*)$称为权重函数。

这里有大量的线性平滑器，例如**核回归(kernal regression)、局部加权回归(locally weighted regression)、平滑线条(smoothing splines)以及GP回归(GP regression)**。为了了解GP回归是一个线性平滑器，注意到一个GP的后验预测分布的均值给定为
$$
\bar{f}(\mathbf{x}_*) = \mathbf{k}_*^T(\mathbf{K}+\sigma_y^2\mathbf{I}_N)^{-1}\mathbf{y}=\sum_{i=1}^N y_iw_i(\mathbf{x}_*)      \tag{15.81}
$$
其中$w_i(\mathbf{x}_*)=[(\mathbf{K}+\sigma_y^2\mathbf{I}_N)^{-1}\mathbf{k}_*]_i$。

在核回归中，我们从一个平滑核而不是Mercer核中得到权重，所以很明显权重函数将会有本地支持。在GP中，事情是不清楚的，因为权重函数依赖于$\mathbf{K}$的逆。对于确定的GP核函数，我们可以解析的得到$w_i(\mathbf{x})$的形式；这称为**等效核(equivalent kernal)**。可以证明$\sum_{i=1}^Nw_i(\mathbf{x}_*)=1$，尽管我们可以有$w_i(\mathbf{x})\lt 0$，因此我们计算的是$y_i$的一个线性组合而不是凸组合。更有趣的是，$w_i(\mathbf{x}_*)$是一个局部函数，即使GP之前使用的原始核不是局部的。此外，GP的等效内核的有效带宽随样本大小$N$的增加而自动减少，而在内核平滑中，需要手动设置带宽$h$以适应$N$。

#### 15.4.2.1 线性平滑器的自由度(Degrees of freedom of linear smoothers)

这个方法是线性的原因是很明显的，但是还不清楚为什么叫“平滑器”。这个最好用GPs的项来解释。考虑在训练集上的预测
$$
\bar{\mathbf{f}} = \mathbf{K}(\mathbf{K}+\sigma_y^2)^{-1}\mathbf{y}     \tag{15.82}
$$
令$\mathbf{K}$有特征分解$\mathbf{K}=\sum_{i=1}^{N}\gamma_i\mathbf{u}_i\mathbf{u}_i^T$。因为$\mathbf{K}$是一个实对称正定矩阵，因此特征值$\lambda_i$是实数且非负的，特征向量$\mathbf{u}_i$是正交的。令$\mathbf{y}=\sum_{i=1}^N\gamma_i\mathbf{u}_i$，其中$\gamma_i=\mathbf{u}_i^T\mathbf{y}$。那么，我们重写上述方程：
$$
\bar{\mathbf{f}} = \sum_{i=1}^N\frac{\gamma_i\lambda_i}{\lambda_i+\sigma_y^2}\mathbf{u}_i       \tag{15.83}
$$
这与方程7.47一样，区别是我们使用**Gram矩阵**$\mathbf{K}$的特征向量而不是数据矩阵$\mathbf{X}$的。在任何情况中，解释是类似的：如果$\frac{\lambda_i}{\lambda_i+\sigma_y^2}\ll 1$，那么对应的基函数$\mathbf{u}_i$将不会有太多的影响。因此，$\mathbf{y}$中的高频成分被平滑了。线性平滑器的自由度定义为
$$
\text{dof} \triangleq \text{tr}(\mathbf{K}(\mathbf{K}+\sigma_y^2\mathbf{I})^{-1})=\sum_{i=1}^N\frac{\lambda_i}{\lambda_i+\sigma_y^2}        \tag{15.84}
$$
这个指定了曲线的“摆动”程度。

### 15.4.3 SVMs与GPs的比较(SVMs compared to GPs)

我们看到在14.5.2节中，SVM对于二分类的目标函数给定为方程14.57
$$
J(\mathbf{w}) = \frac{1}{2}\lVert\mathbf{w}\rVert^2 + C\sum_{i=1}^N(1-y_if_i)_+     \tag{15.85}
$$
我们可以从方程14.59中，我们也可以看到最优解的形式为$\mathbf{w}=\sum_i\alpha_i\mathbf{x}_i$，所以$\lVert\mathbf{w}\rVert^2=\sum_{i,j}\alpha_i\alpha_j\mathbf{x}_i^T\mathbf{x}_j$。核化我们得到$\lVert\mathbf{w}\rVert^2=\boldsymbol{\alpha}^T\mathbf{K}\boldsymbol{\alpha}$。根据方程14.61，且$\hat{w}_0$项合并到一个核中，我们得到$\mathbf{f}=\mathbf{K}\boldsymbol{\alpha}$，所以$\lVert\mathbf{w}\rVert^2=\mathbf{f}^T\mathbf{K}^{-1}\mathbf{f}$。因此，SVM目标函数可以重写为
$$
J(\mathbf{f}) = \frac{1}{2}\mathbf{f}^T\mathbf{f} + C\sum_{i=1}^N(1-y_if_i)_+     \tag{15.86}
$$
这个对比GP分类器的MAP估计：
$$
J(\mathbf{f}) = \frac{1}{2}\mathbf{f}^T\mathbf{f} - \sum_{i=1}^N\log p(y_i\vert f_i)    \tag{15.87}
$$
我们很容易想到，我们可以通过计算等效于hinge损失的似然，将SVM“转换”为GP。然而，事实证明不存在这种似然(Sollich 2002)，尽管存在与SVM匹配的伪似然(见14.5.5节)。

![image](Figure6.7.png)
从图6.7可以看出，higne损失和logistic损失（以及概率损失）非常相似。主要区别在于，当误差大于1时，higne损失严格为0。这就产生了稀疏解。在14.3.2节中，我们讨论了导出稀疏核机器的其他方法。我们将在下面讨论这些方法与GPs之间的联系。

### 15.4.4 L1VM与RVMs对GPs的对比(L1VM and RVMs compared to GPs)

稀疏核机器只是一个形式为$[\kappa(\mathbf{x},\mathbf{x}_1),\cdots,\kappa(\mathbf{x},\mathbf{x}_N)]$的基函数扩展的线性模型。从15.4.1节中，我们知道这个等效于一个有如下核的GP：
$$
\kappa(\mathbf{x},\mathbf{x}^{\prime}) = \sum_{j=1}^D\frac{1}{\alpha_j}\phi_j(\mathbf{x})\phi_j(\mathbf{x}^{\prime})        \tag{15.88}
$$
其中$p(\mathbf{w})=\mathcal{N}(\mathbf{0},\text{diag}(\alpha_j^{-1}))$。这个核函数有两个有趣的性质。
- 首先，它是退化的，意味着其有最多$N$个非零特征值，所以联合分布$p(\mathbf{f,f}_*)$将会被高度约束。
- 第二，核依赖于训练数据。这可能导致模型在超出训练数据的外推时过于自信。

要看到这一点，请考虑数据凸包之外很远的一个点$\mathbf{x}_*$。所有基函数的值都将接近0，因此预测值将回到GP的平均值。更令人担忧的是，方差将退到噪声方差上。相比之下，当使用非退化核函数时，预测方差随着远离训练数据而增加。更多讨论见(Rasmussen和Quiñonero Candela 2005)。

### 15.4.5 神经网络与GPs比较(Neural networks compared to GPs)

在16.5节中，我们将讨论**神经网络，其是GLMs的一个非线性推广**。在二元分类情况中，神经网络由应用于`logistics`回归模型的`logistics`回归模型定义：
$$
p(y\vert\mathbf{x},\boldsymbol{\theta})=\text{Ber}(y\vert\text{sigm}(\mathbf{w}^T\text{sigm}(\mathbf{Vx})))     \tag{15.89}
$$
可以证明，在**神经网络与高斯过程之间存在一个有趣的联系**。

为了解释联系，我们使用如下表示。考虑一个用于回归的含有一个隐层的神经网络。形式为
$$
p(y\vert\mathbf{x},\boldsymbol{\theta})=\mathcal{N}(y\vert f(\mathbf{x};\boldsymbol{\theta}),\sigma^2)      \tag{15.90}
$$
其中
$$
f(\mathbf{x}) = b + \sum_{j=1}^H v_i g(\mathbf{x;u}_j)   \tag{15.91}
$$
其中$b$是偏差项的偏移，$v_j$是从隐藏单元$j$到响应$y$的**输出权重**，其中$\mathbf{u}_j$是从输入$\mathbf{x}$到隐藏单元$j$的**输入权重**，且$g()$是隐层单元的**激活函数**。这是一般是**sigmoid函数**或是**tanh函数**，但是可以是任意的平滑函数。

我们使用权重的如下先验：其中$b\sim\mathcal{N}(0,\sigma_b^2),\mathbf{v}\sim\prod_j\mathcal{N}(v_j\vert0,\sigma_w^2)$，其中对于一些为特定的$p(\mathbf{u}_j)$有$\mathbf{u}\sim\prod_jp(\mathbf{u}_j)$。将所有的权重记为$\boldsymbol{\theta}$，我们有
$$
\begin{aligned}
    \mathbb{E}_{\boldsymbol{\theta}}[f(\mathbf{x})] &= 0\\
    \mathbb{E}_{\boldsymbol{\theta}}[f(\mathbf{x})f(\mathbf{x}^{\prime})] &= \sigma_b^2 + \sum_j\sigma_v^2 \mathbb{E}_{\mathbf{v}}[g(\mathbf{x;u}_j)g(\mathbf{x}^{\prime};\mathbf{u}_j)] \\
    &= \sigma_b^2 + H \sigma_v^2 \mathbb{E}_{\mathbf{u}}[g(\mathbf{x;u})g(\mathbf{x}^{\prime};\mathbf{u})]
\end{aligned}       \tag{15.92-15.94}
$$
其中最后一个等式成立因为$H$隐单元是$iid$的。如果我们令$\sigma_v^2$缩放为$w^2/H$，那么最后一项变为$w^2\mathbb{E}_{\mathbf{u}}[g(\mathbf{x;u})g(\mathbf{x}^{\prime};\mathbf{u})]$。这个是一个在$H$个$iid$随机变量之和。假设$g$是有界的，我们可以应用大数定理。结果是随着$H\rightarrow\infty$，我们得到一个高斯过程。

![image](Figure15.9.png)
如果我们使用一个激活/转换函数$g(\mathbf{x;u})=\text{erf}(u_0 + \sum_{j=1}^Du_jx_j)$，其中$\text{erf}(z)=2/\sqrt{\pi}\int_0^ze^{-t^2}dt$，且我们选择$\mathbf{u}\sim\mathcal{N}(\mathbf{0,\Sigma})$，那么(Williams 1998)证明协方差核的形式为
$$
\kappa_{NN}(\mathbf{x},\mathbf{x}^{\prime}) = \frac{2}{\pi}\sin^{-1}\left( \frac{2\tilde{\mathbf{x}}^T\mathbf{\Sigma}\tilde{\mathbf{x}}^{\prime}}{\sqrt{(1+2\tilde{\mathbf{x}}^T\mathbf{\Sigma}\tilde{\mathbf{x}})(1+2(\tilde{\mathbf{x}}^{\prime})^T\mathbf{\Sigma}(\tilde{\mathbf{x}}^{\prime}))}} \right)        \tag{15.95}
$$
其中$\tilde{\mathbf{x}}=(1,x_1,\cdots,x_D)$。这是一个真实的"神经网络"核，不像“sigmoid”核$\kappa(\mathbf{x},\mathbf{x}^{\prime})=\tanh(a+b\mathbf{x}^T\mathbf{x}^{\prime})$，该核是非正定的。

当$D=2$与$\mathbf{\Sigma}=\text{diag}(\sigma_0^2,\sigma^2)$时，图15.9(a)解释了这个核。图15.9(b)显示了一些函数采样与对应的GP。这些是等效于覆盖了$\text{erf}(u_0+ux)$的函数，其中$u_0,u$是随机的。随着$\sigma^2$增加，$u$的方差也增加，所以函数变化很快。不像RBF核，从该内核采样的函数不会趋向于远离数据0，而是趋于保持与数据“边缘”处相同的值。

选择假设，我们使用一个RBF网络，其等效于一个形式为
$g(\mathbf{x;u})=$$\exp(-\vert\mathbf{x}-\mathbf{u}\vert^2/(2\sigma_g^2))$的隐单元激活函数。如果$\mathbf{u}\sim\mathcal{N}(0,\sigma_u^2\mathbf{I})$，可以证明对应的核等效于RBF或SE核。
 
### 15.4.6 平滑线条与GPs的比较(Smoothing splines compared to GPs *)

**平滑样条曲线**是一种用于平滑插值数据的广泛使用的非参数方法(Green和Silverman 1994)。我们将看到，它们是GP的特例。它们通常在输入为一维或二维时使用。

#### 15.4.6.1 单元线条(Univariate splines)

基本思想是**通过最小化与数据的差异以，及对“过于摆动”的函数进行惩罚的平滑项(smoothing term)来拟合函数$f$**。如果我们惩罚函数的第$m$阶导数，目标函数变为
$$
J(f) = \sum_{i=1}^N (f(x_i)-y_i)^2 + \lambda \int \left( \frac{d^m}{dx^m}f(x) \right)^2 dx  \tag{15.96}
$$
可以证明解是一个**分段多项式(piecewise polynomial)**，其中多项式在$[x_{i-1},x_i]$(记为$\mathcal{I}$)内为$2m-1$阶，在两个两个最外面的间隔$(-\infty,x_1],[x_N,\infty)$中阶数为$m-1$,
$$
f(x) = \sum_{j=0}^{m-1}\beta_j x^j + \mathbb{I}(x\in\mathcal{I}) \left(  \sum_{i=1}^N\alpha_i(x-x_i)_+^{2m-1} \right) + \mathbb{I}(x\notin\mathcal{I})\left(  \sum_{i=1}^N\alpha_i(x-x_i)_+^{m-1} \right)
$$
如果$m=2$，我们得到(自然)三次样条
$$
f(x) = \beta_0 + \beta_1 x + \mathbb{I}(x\in\mathcal{I}) \left(  \sum_{i=1}^N\alpha_i(x-x_i)_+^{3} \right) + \mathbb{I}(x\notin\mathcal{I})\left(  \sum_{i=1}^N\alpha_i(x-x_i)_+ \right)
$$
这是一个截断三次多项式的级数，其左手边位于$N$个训练点中的每一个上。(由于模型在边缘上是线性的，因此可以防止模型在数据范围之外进行过多的外推；如果放弃此要求，则会得到“无限制”样条。)

我们可以使用岭回归拟合这个模型:$\hat{\mathbf{w}} = (\mathbf{\Phi^T\Phi}+\lambda\mathbf{I}_N)^{-1}\mathbf{\Phi}^T\mathbf{y}$，其中$\mathbf{\Phi}$的列为1、$x_i$、对于$i=2:N-1$有$(x-x_i)^3_+$以及$i=1/i=N$的$(x-x_i)_+$。

#### 15.4.6.2 回归样条(Regression splines)

一般，我们可以将多项式放置在固定的$K$个点上，称为结点(**knots**)，记为$\xi_k$。这个结果称为**回归样条**。这是一个参数模型，使用如下形式的基函数扩展(其中我们丢弃调内部或外部差异)
$$
f(x)=\beta_0 + \beta_1x+\sum_{k=1}^K\alpha_j(x-\xi_k)_+^3       \tag{15.99}
$$
选择结点的数量以及位置就像选择在14.3.2节中选择支持向量的数量与值。如果我们在回归系数$\alpha_j$上强加一个$\ell_2$正则化，方法称为**惩罚线条**。

#### 15.4.6.3 与GPs的联系(The connection with GPs)

可以证明三次样条是如下函数的MAP估计
$$
f(x)=\beta_0 + \beta_1x + r(x)  \tag{15.100}
$$
其中$p(\beta_j)\propto1$，且$r(x)\sim\text{GP}(0,\sigma_f^2\kappa_{sp}(x,x^{\prime}))$，其中
$$
\kappa_{sp}(x,x^{\prime})\triangleq\int_0^1(x-u)_+(x^{\prime}-u)_+du    \tag{15.101}
$$
注意到，公式15.101中的内核是相当不自然的，实际上，来自所得GP的后验样本相当不平滑。**但是，后验众数/均值是平滑的。这表明正则化并不一定总是具有先验优势**。

#### 15.4.6.4 二维输入(2d input (thin-plate splines))

可以通过定义一个正则器将三次型样条推广到2维输入
$$
\int\int\left[  \left( \frac{\partial^2f(x)}{\partial x_1^2} \right)^2 + 2\left( \frac{\partial^2f(x)}{\partial x_1\partial x_2}   \right)  + \left( \frac{\partial^2f(x)}{\partial x_2s^2} \right)^2  \right] dx_1dx_2 \tag{15.102}
$$
可以证明解的形式为
$$
f(x) = \beta_0 + \boldsymbol{\beta}_1^T \mathbf{x} + \sum_{i=1}^N \alpha_i \boldsymbol{\phi}_i(\mathbf{x})      \tag{15.103}
$$
其中$\boldsymbol{\phi}_i(\mathbf{x})=\eta(\lVert\mathbf{x}-\mathbf{x}_i\rVert)$，且$\eta(z)=z^2\log z^2$。这称为**薄板样条**。这等效于一个有GP的MAP估计。

#### 15.4.6.5 更高维的输入(Higher-dimensional inputs)

使用高阶输入时，很难解析求解最优解的形式。但是，在参数回归样条曲线背景下，我们放弃了$f$的正则化器，因此在定义基函数方面有更多的自由。在处理多输入方面的一种方法是使用**张量积基**，定义为一维基函数的交叉积。例如，对于2维输入，我们定义
$$
\begin{aligned}
f(x_1,x_2) &= \beta_0 + \sum_{m}\beta_{1m}(x_1 - \xi_{1m})_+ + \sum_{m}\beta_{2m}(x_2 - \xi_{2m})_+ \\
& + \sum_{m}\beta_{12m}(x_1 - \xi_{1m})_+(x_2 - \xi_{2m})_+ \tag{15.104-15.105}
\end{aligned}
$$
很明显，对于高维数据，我不能允许高阶交互，因为有太多的参数要拟合。这个问题的一种方法是使用一种搜索过程来寻找有用的交互项。这个称为$MARS$，意思是'multivariate adaptive regression splines'。

### 15.4.7 RKHS methods compared to GPs *

我们可以将类似平滑线条中那样，对函数导数进行惩罚的思想进行推广，使函数具有更一般的光滑性概念。回忆14.2.3节中，Mercer的理论认为**任何正定核函数都可以用特征函数表示**：
$$
\kappa(\mathbf{x},\mathbf{x}^{\prime}) = \sum_{i=1}^{\infty}\lambda_i\phi_i(\mathbf{x})\phi_i(\mathbf{x}^{\prime})      \tag{15.106}
$$
$\phi_i$组成了一个函数空间中的正交基
$$
\mathcal{H}_k = \{f : f(\mathbf{x}) = \sum_{i=1}^{\infty}f_i \phi_i(\mathbf{x}), \,  \sum_{i=1}^{\infty}f_i^2/\lambda_i\lt\infty  \}        \tag{15.107}
$$
现在定义两个函数$f(\mathbf{x})=\sum_{i=1}^{\infty}f_i \phi_i(\mathbf{x})$与$g(\mathbf{x})=\sum_{i=1}^{\infty}g_i \phi_i(\mathbf{x})$的内积为
$$
\langle f,g \rangle_{\mathcal{H}} \triangleq \sum_{i=1}^{\infty} \frac{f_ig_i}{\lambda_i}           \tag{15.108}
$$
在练习15.1中，我们证明这个定义意味着
$$
\langle \kappa(\mathbf{x}_1,\cdot),\kappa(\mathbf{x}_2,\cdot) \rangle_{\mathcal{H}} = \kappa(\mathbf{x}_1, \mathbf{x}_2)        \tag{15.109}
$$
这个称为**再生性**，函数空间$\mathcal{H}_k$称为**再生核Hilbert空间**或**RKHS**。

现在考虑一个形式为最优化问题
$$
J(f) = \frac{1}{2\sigma_y^2} \sum_{i=1}^{N} \left(y_i - f(\mathbf{x}_i)\right)^2 + \frac{1}{2}\lVert f\rVert_H^2       \tag{15.110}
$$
其中$\lVert f\rVert_H$称为函数的范数：
$$
\lVert f\rVert_H = \langle f,f \rangle_{\mathcal{H}} = \sum_{i=1}^{\infty} \frac{f_i^2}{\lambda_i}
$$
直觉是，复杂的函数在内核中会有很大的范数，因为它们需要许多特征函数来表示它们。我们想选择一个简单的函数，它能很好地适应数据。

可以证明解的形式为
$$
f(\mathbf{x}) = \sum_{i=1}^N \alpha_i \kappa(\mathbf{x}, \mathbf{x}_i)      \tag{15.112}
$$
这个已知为**表示定理(representer theorem)**，对于除平方误差的其他凸损失函数都成立。

我们可以通过替代$f(\mathbf{x}) = \sum_{i=1}^N \alpha_i \kappa(\mathbf{x}, \mathbf{x}_i)$并使用再生性质求解$\boldsymbol{\alpha}$得到
$$
J(\boldsymbol{\alpha}) = \frac{1}{2\sigma_y^2}\vert\mathbf{y-K}\boldsymbol{\alpha}\vert^2 + \frac{1}{2}\boldsymbol{\alpha}^T\mathbf{K}\boldsymbol{\alpha}
$$
相对$\boldsymbol{\alpha}$最小化得到
$$
\hat{\boldsymbol{\alpha}} = (\mathbf{K} + \sigma_y^2\mathbf{I})^{-1}
$$
因此
$$
\hat{f}(\mathbf{x}_*) = \sum_i \hat{{\alpha}}_i \kappa(\mathbf{x}_*, \mathbf{x}_i) = \mathbf{k}_*^T (\mathbf{K}+\sigma_y^2\mathbf{I})^{-1}\mathbf{y}         \tag{15.115}
$$
这等效于公式15.18，一个GP后验分布的后验均值。确实，因为一个高斯分布的均值与众数是相同的，我们可以看到有RKHS正则化的线性回归等效于一个GP的MAP估计。类似的描述与高斯过程logistics回归情况也成立，其中也使用一个凸似然/损失函数。

## 15.5 GP latent variable model

在14.4.4节中，我们讨论了核PCA(kernel PCA)，就是将核技巧应用到常规PCA中。本节中，我们将讨论另一种将核与概率PCA组合在一起的技术。得到的方法称为**GP-LVM**，意思是"**Guassian process with latent variable model**"。 

为了解释方法，我们以PPCA开始。回忆12.4节中的PPCA模型定义如下：
$$
\begin{aligned}
p(\mathbf{z}_i) &= \mathcal{N}(\mathbf{z}_i \vert \mathbf{0, I}) \\
p(\mathbf{y}_i \vert \mathbf{z}_i, \boldsymbol{\theta}) &= \mathcal{N}(\mathbf{y}_i \vert \mathbf{Wz}_i, \sigma^2\mathbf{I}) 
\end{aligned}
$$

我们可以通过对$\mathbf{z}_i$积分并最大化$\mathbf{W}$(以及$\sigma^2$)，使用最大似然来拟合这个模型。目标函数给定为
$$
p(\mathbf{Y}\vert \mathbf{W},\sigma^2) = (2\pi)^{-DN/2}\vert\mathbf{C}\vert^{-N/2}\exp \left( -\frac{1}{2}\text{tr} (\mathbf{C}^{-1}\mathbf{Y}^T\mathbf{Y})  \right)
$$
其中$\mathbf{C=WW}^T+\sigma^2\mathbf{I}$。如我们理论12.2中介绍的，MLE可以计算$\mathbf{Y}^T\mathbf{Y}$的特征向量的项。

现在考虑对偶问题，因此我们最大化$\mathbf{Z}$并积分$\mathbf{W}$。我们将使用形式$p(\mathbf{W})=\prod_j\mathcal{N}(\mathbf{w}_j\vert \mathbf{0, I})$的一个先验形式。对应的似然项是
$$
\begin{aligned}
    p(\mathbf{Y}\vert\mathbf{Z}, \sigma^2) &= \prod_{d=1}^D \mathcal{N}(\mathbf{y}_{:, d}\vert \mathbf{0, ZZ}^T+\sigma^2\mathbf{I})     \\
    & = (2\pi)^{-DN/2} \vert \mathbf{K}_z \vert^{-D/2}    \exp \left( -\frac{1}{2}\text{tr} (\mathbf{K}_z^{-1}\mathbf{Y}\mathbf{Y}^T)  \right)        \tag{15.119}
\end{aligned}
$$
其中$\mathbf{K}_z=\mathbf{ZZ}^T+\sigma^2\mathbf{I}$。基于我们关于$\mathbf{YY}^T$与$\mathbf{Y}^T\mathbf{Y}$的特征值之间的关系，我们可以使用特征值方法也可以求解这个对偶问题也就是
不足为奇了。

如果我们使用一个线性核，我们恢复PCA。但是也可以使用一个更加普通的核:$\mathbf{K}_z=\mathbf{K}+\sigma^2\mathbf{I}$，其中$\mathbf{K}$是$\mathbf{Z}$的Gram矩阵。$\hat{\mathbf{Z}}$的MLE不再适用于特征值方法；相反我们必须使用基于梯度的优化方法。目标给定为：
$$
\ell = -\frac{D}{2} \log\vert\mathbf{K}_z\vert - \frac{1}{2}\text{tr}(\mathbf{K}_z^{-1}\mathbf{Y}\mathbf{Y}^T)
$$

梯度给定为
$$
\frac{\partial\ell} {\partial Z_{ij}} = \frac{\partial \ell}{\partial \mathbf{K}_z} \frac{\partial \mathbf{K}_z}{\partial Z_{ij}}
$$
其中
$$
\frac{\partial \ell}{\partial \mathbf{K}_z} = \mathbf{K}_z^{-1}\mathbf{YY}^T\mathbf{K}_z^{-1}
$$


[^1]:之所以称其为边际似然，而不仅仅是似然的原因，是因为我们已经将潜在的高斯向量$\mathbf{f}$边缘化了。这使我们上移了贝叶斯层次结构的一层，并减少了过度拟合的机会（与标准参数模型相比，内核参数的数量通常很小）。