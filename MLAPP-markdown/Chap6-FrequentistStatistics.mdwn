<!-- TOC -->

- [6 频率统计(Frequentist statistics)](#6-频率统计frequentist-statistics)
  - [6.1 引言(Introduction)](#61-引言introduction)
  - [6.2 估计器的采样分布(Sampling distribution of an estimator)](#62-估计器的采样分布sampling-distribution-of-an-estimator)
    - [6.2.1 自助采样(Boostrap)](#621-自助采样boostrap)
    - [6.2.2 MLE的大规模采样理论(Large sample theory for the MLE)](#622-mle的大规模采样理论large-sample-theory-for-the-mle)
  - [6.3 Frequentist decision theory](#63-frequentist-decision-theory)
    - [6.3.1 贝叶斯风险(Bayesian Risk)](#631-贝叶斯风险bayesian-risk)
    - [6.3.2 Minimax risk](#632-minimax-risk)
    - [6.3.3 Admissible estimators](#633-admissible-estimators)
      - [6.3.3.1 例子](#6331-例子)
      - [6.3.3.2 斯坦悖论(Stein’s paradox)](#6332-斯坦悖论steins-paradox)
  - [6.4 Desirable properties of estimators](#64-desirable-properties-of-estimators)
    - [6.4.1 一致估计器(Consistent estimators)](#641-一致估计器consistent-estimators)
    - [6.4.2 Unbiased estimators](#642-unbiased-estimators)
    - [6.4.3 最小方差估计器(Minimum variance estimators)](#643-最小方差估计器minimum-variance-estimators)
    - [6.4.4 偏差-方差平衡(The bias-variance tradeoff)](#644-偏差-方差平衡the-bias-variance-tradeoff)
      - [6.4.4.1 Example: estimating a Gaussian mean](#6441-example-estimating-a-gaussian-mean)
    - [6.4.4.2 Example: ridge regression](#6442-example-ridge-regression)
      - [6.4.4.3 Bias-variance tradeoff for classification](#6443-bias-variance-tradeoff-for-classification)
  - [6.5 Empirical risk minimization](#65-empirical-risk-minimization)
    - [6.5.1 Regularized risk minimization](#651-regularized-risk-minimization)
    - [6.5.2 Structural risk minimization](#652-structural-risk-minimization)
    - [6.5.3 Estimating the risk using cross validation](#653-estimating-the-risk-using-cross-validation)
      - [6.5.3.1 Example: using CV to pick $\lambda$ for ridge regression](#6531-example-using-cv-to-pick-lambda-for-ridge-regression)
    - [6.5.5 Surrogate loss functions](#655-surrogate-loss-functions)

<!-- /TOC -->
# 6 频率统计(Frequentist statistics)

## 6.1 引言(Introduction)

我们在第5章中描述的统计推断方法称为贝叶斯统计。也许令人惊讶的是，这被一些人认为是有争议的，而Bayes规则在非统计问题上的应用，例如医学诊断（第2.2.3.1节）、垃圾邮件过滤（第3.4.4.1节）或飞机跟踪（第18.2.1节）则没有争议。反对的原因与统计模型参数与其他未知量之间的错误区分有关。[^1]

人们试图设计出统计推断的方法，避免把参数当作随机变量来处理，从而避免使用先验和贝叶斯规则。这类方法被称为**频率统计、经典统计或正统统计**。它们不是基于后验分布，而是基于抽样分布的概念。对于未知分布，请参见第6.6节所述的抽样分布，但对于这一点，请参见第6.2节。正是这种在反复试验中的变化概念，构成了频率专家方法所使用的建模不确定性的基础。

相比之下，在贝叶斯方法中，我们只对实际观察到的数据进行条件处理；不存在重复试验的概念。这允许使用贝叶斯计算一次性事件的概率，正如我们在第2.1节中讨论的那样。也许更重要的是，贝叶斯方法避免了某些困扰频率派方法的悖论(见第6.6节)。然而，熟悉频率统计量(尤其是第6.5节)是很重要的，因为它在机器学习中被广泛使用。

## 6.2 估计器的采样分布(Sampling distribution of an estimator)

在频率统计方面，一个估计参数$\hat{\boldsymbol{\theta}}$的计算是将一个**估计器**$\delta$应用到一些数据$\mathcal{D}$得到的，所以$\hat{\boldsymbol{\theta}}=\delta(\mathcal{D})$。**参数是固定的，数据是随机的**，这与贝叶斯方法完全相反。参数估计中的不确定性可以通过计算估计器的抽样分布来测度。要理解这一概念，请设想对许多来自一些真实模型$p(\cdot|\theta^*)$的不同的数据集$\mathcal{D}^{(s)}$进行采样;也就是令$\mathcal{D}^{(s)}=\{x_{i}^{(s)}\}_{i=1}^{N}$，其中$x_{i}^{s}\sim p(\cdot|\theta^*)$，且$\theta^*$是真实参数。这里$s=1:S$索引了采样数据集，$N$是每个子集的大小。选择将估计器$\hat{\theta}(\cdot)$应用到每个$\mathcal{D}^{(s)}$从而得到一个估计器集合$\{\hat{\theta}\mathcal{D}^{(s)}\}$。随着我们令$S\rightarrow\infty$，由$\hat{\theta}(\cdot)$得出的分布是估计器的采样分布。随后章节中我们将讨论各种使用采样分布的方式。但是，首先我们勾勒出两种方法来计算采样分布本身。

### 6.2.1 自助采样(Boostrap)

**booostrap**是一种简单的**蒙特卡洛技术**，用于近似采样分布。这在**估计器是真实参数的复杂函数的情况下特别有用**。

这个思想很简单，如果我们知道真实参数$\theta^*$，我们将从真实分布$x^s_i\sim p(\cdot|\theta^*)$,$s=1:S,i=1:N$中产生许多虚假数据集，每个大小为$N$。我们然后从每个采样$\hat{\theta}^s=f(x_{1:N}^s)$计算出计算出我们的估计器，并使用得到的采样的经验分布作为我们的采样分布的估计。因为$\theta$是未知的，因此**参数化bootstrap**使用$\hat{\theta}(\mathcal{D})$来产生采样。另外一种称为**非参数化的bootstrap**，从原始数据集采样$x_i^s$*(有放回的)，然后计算诱导分布。

![image](Figure6.1.png)

图6.1显示了一个使用参数bootstrap为伯努利分布计算了MLE的采样分布的例子。(结果使用一个非参数化的)我们看到采样分布是对称的，因此当$N=10$时是远离高斯的；当$N=100$时分布看起来更加高斯。

自然会产生一个问题：通过bootstrap采样计算得到的参数估计$\hat{\theta}^s=\hat{\theta}(x_{1:N}^s)$与通过后验采样得到的参数值$\theta^s\sim p(\cdot|\theta)$之间有什么联系呢？概念上确实差异很大，但是在先验不是很强的普通情况下，它们确实是非常相似。

然而，bootstrap比后验采样更慢。**原因是bootstrap必须拟合模型S次，而在后采样中，我们通常仅拟合一次模型(以找到局部模式)**，然后围绕该模式执行局部探索。这种局部探索通常比从头开始拟合模型要快得多。

### 6.2.2 MLE的大规模采样理论(Large sample theory for the MLE) 

在某些情况下，某些估计器的采样分布可以解析计算。特别地，可以证明，在确定条件下，随着样本量趋于无穷大，MLE的样本分布变为高斯分布。非正式地，要保持此结果，就必须使模型中的每个参数都能“看到”无限量的数据，并且模型是可识别的。不幸的是，这排除了机器学习感兴趣的许多模型。尽管如此，让我们假设我们处于定理成立的简单环境中。

高斯的核心是MLE$\hat{\theta}$。但是高斯的方差是多少呢？直觉上，估计器的方差将与似然表面在其峰值处的曲率有关。如果曲率大，则峰将是“尖锐的”，并且方差低；在这种情况下，估计是“确定的”。相反，如果曲率小，则峰将接近“平坦”，因此方差大。

我们现在正式描述这种直觉。将**分数函数**定义为对数似然函数在点$\hat{\theta}$出的梯度：
$$
\mathbf{s}(\hat{\boldsymbol{\theta}})\triangleq \nabla\log p(\mathcal{D}|\boldsymbol{\theta})|_{\hat{\boldsymbol{\theta}}}
$$


## 6.3 Frequentist decision theory

在频率学派或是经典的决策理论中，一般都有一个**损失函数**以及一个**似然**，但是没有先验，因此没有后验或是后验预期损失。因此，不像贝叶斯情况,没有得到最优估计器的自然方式。相反的，频率方法中，我们可自由选择任何我们想要的估计器或是决策过程，$\sigma:\mathcal{X}\rightarrow\mathcal{A}$[^3]。

已然选择好估计器后，我可以定义它的期望损失(expected loss)或是风险(risk)如下：
$$
R(\theta^*,\delta)\triangleq\mathbb{E}_{p(\tilde{\mathcal{D}}|\theta^*)}[L(\theta^*,\delta(\tilde{\mathcal{D}}))] = \int L(\theta^*,\delta(\tilde{\mathcal{D}}))p(\tilde{\mathcal{D}}|\theta^*)d\tilde{\mathcal{D}}        \tag{6.9}
$$
其中$\tilde{\mathcal{D}}$是从自然分布中采样得到的数据，由参数$\theta^*$来表示。换而言之，期望是相对估计其采样分布的。将这个与贝叶斯后验期望损失进行对比：
$$
\rho(a|\mathcal{D},\pi)\triangleq\mathbb{E}_{p(\theta|\mathcal{D},\pi)}[L(\theta,a)]=\int_{\Theta}L(\theta,\mathbf{a})p(\theta|\mathcal{D},\pi)d\theta  \tag{6.10}
$$

我们看到贝叶斯方法是在$\theta$之上进行平均，并以$\mathcal{D}$(是已知的)为条件，然而频率方法在$\tilde{\mathcal{D}}$(忽略了观测数据)之上进行平均，并以$\theta^*$(是未知的)为条件。

不只是频率定义不自然，同时也很难计算，因为$\theta^*$是未知的。因此，我们无法以频率风险对比不同的估计器。我们讨论大量的解。


### 6.3.1 贝叶斯风险(Bayesian Risk)

我们如何在估计器之间进行选择？我们需要知道一些将$R(\boldsymbol{\theta}^*,\delta)$转化为质量的单个测量$R(\delta)$，不依赖于对$\boldsymbol{\theta}^*$的了解。一种方法是$\boldsymbol{\theta}^*$在之上加一些先验，然后定义一个估计器的贝叶斯风险或是集成风险如下
$$
R_{B}(\delta)\triangleq\mathbb{E}_{p(\boldsymbol{\theta}^*)}[R(\boldsymbol{\theta}^*,\delta)] = \int R(\boldsymbol{\theta}^*,\delta)p(\boldsymbol{\theta}^*)d\boldsymbol{\theta}^*  \tag{6.11}
$$
一个贝叶斯估计器或是贝叶斯决策准则是最小化期望风险
$$
\delta_{B}\triangleq\argmax_{\delta}R_{B}(\delta)   \tag{6.12}
$$
请注意，综合风险也称为前置风险，因为它是在我们看到数据之前。最小化这一点对于实验设计是有用的。

我们现在证明一个非常重要的理论。然后将贝叶斯与频率方法与决策理论联系起来。

**理论6.3.1**通过最小化每个$\mathbf{x}$后验期望损失，我们可以得到一个贝叶斯估计器。
证明：通过改变积分顺序，我们有
$$
\begin{aligned}
    R_{B}(\delta) &= \int\left[\sum_{\mathbf{x}}\sum_{y} L(y,\delta(\mathbf{x}))p(\mathbf{x},y|\boldsymbol{\theta}^*) \right]p(\boldsymbol{\theta}^*)d\boldsymbol{\theta}^* \\
    &=\sum_{\mathbf{x}}\sum_{y}\int_{\Theta}L(y,\delta(\mathbf{x}))p(\mathbf{x},y,\boldsymbol{\theta}^*) d\boldsymbol{\theta}^* \\
    &= \sum_{\mathbf{x}}\left[ \sum_y L(y,\delta(\mathbf{x})) p(y|\mathbf{x})dy \right]p(\mathbf{x})\\
    &=\sum_{\mathbf{x}}\rho(\delta(\mathbf{x})|\mathbf{x})p(\mathbf{x})
\end{aligned}
$$

为了最小化总体期望，我们可以最小化每个$\mathbf{x}$的内部项，所以我们的决策规则是挑选
$$
\delta_{B}(\mathbf{x})=\argmin_{a\in\mathcal{A}}\rho(a|\mathbf{x})  \tag{6.17}
$$
因此，我们看到在逐个情况的基础上(如贝叶斯方法)挑选最优行为是在平均上的最优(如频率方法)。换而言之，贝叶斯方法提供了一种达到频率目标的很好的方法。实际上，我们可以更进一步，且如下。

### 6.3.2 Minimax risk
![image](Figure6.2.png)
很明显一些频率派的不喜欢贝叶斯风险，因为需要选择一个先验(尽管这只是在计算估计器时需要，对于构造时不是必须的)。另一种方法如下。定义一个估计器的maximum risk如下
$$
R_{max}(\delta)\triangleq\max_{\boldsymbol{\theta}^*}R(\boldsymbol{\theta}^*,\delta)    \tag{6.18}
$$
一个$\text{minimax}$准则是最小化最大化风险：
$$
\delta_{MM}\triangleq\argmin_{\delta} R_{max}(\delta)   \tag{6.19}
$$
例如，在图6.2中，我们看到在$\theta^*$的所有可能值中，$\delta_1$有比$\delta_2$更低的最坏情况风险，所以这是一个$\text{minimax}$问题。

$\text{Minimax}$估计有特征的吸引力。但是，计算它们是很难的。更进一步，它们非常悲观。事实上，可以看出来所有的$\text{minimax}$估计器等效于**最不利先验-least favorable prior**下的贝叶斯估计。在大多数统计情况下(除博弈论之外)，假设nature是一个对手不是一种合理的假设。

### 6.3.3 Admissible estimators

频率决策问题的基本问题是其为了评估风险会依赖于对真实分布的$p(\cdot|\theta^*)$的认知。然而，不管$\theta^*$的值，一些估计器可能比其他会更差。尤其是，如果对所有的$\theta \in \Theta$有$R(\theta,\delta_1)\leq R(\theta,\delta_2)$，那么我们说$\delta_1$主导$\delta_2$。如果对某些$\theta$不等式是严格的，则称为严格的主导。如果一个估计量不被任何其它估计量严格控制，则称其为可容许的。 

#### 6.3.3.1 例子

考虑估计一个高斯的均值。我们假设数据是从$x_{i}\sim\mathcal{N}(\theta^*,\sigma^2=1)$采样得到的，使用二次型损失$L(\theta,\hat{\theta})=(\theta-\hat{\theta})^2$。对应的风险函数就是MSE。一些可能的决策规则或是估计器$\hat{\theta}(\mathbf{\theta})=\delta(\mathbf{x})$如下：
- $\delta_1(\mathbf{x})=\bar{x}$，采样均值
- $\delta_2(\mathbf{x})=\tilde{\mathbf{x}}$，采样中位数
- $\delta_3(\mathbf{x})=\theta_0$，一个固定值
- $\delta_{\kappa}(\mathbf{x})$，先验$\mathcal{N}(\theta|\theta_0,\sigma^2/\kappa)$下的后验均值
$$
\delta_{\kappa}(\mathbf{x})=\frac{N}{N+K}\bar{x}+\frac{\kappa}{N+\kappa}\theta_0=w\bar{x}+(1-w)\theta_0 \tag{6.20}
$$
对于$\delta_{\kappa}$，我们考虑一个弱先验$\kappa=1$，以及一个强先验$\kappa=5$。先验均值$\theta_0$，一些固定值。我们假设$\sigma^2$是已知的。(那么在无限强先验下$\kappa=\infty$下$\delta_3(\mathbf{x})$与$\delta_{\kappa}(\mathbf{x})$)。

我们现在解析推导出风险函数。在6.4.4节中，我们显示MSE可以分解为**平方偏差加方差**：
$$
MSE(\hat{\theta}(\cdot)|\theta^*)=\text{var}[\hat{\theta}]+\text{bias}^2(\hat{\theta})  \tag{6.21}
$$
采样均值是无偏的，所以它的风险是
$$
MSE(\delta_1|\theta^*)=\text{var}[\bar{x}]=\frac{\sigma^2}{N}   \tag{6.22}
$$
采样中位数也是无偏的。可以证明方差近似于$\pi/(2N)$，所以
$$
MSE(\delta_2|\theta^*)=\frac{\pi}{2N}   \tag{6.23}
$$
对于$\delta_3(\mathbf{x})=\theta_0$，方差是0，所以
$$
MSE(\delta_3|\theta^*)=(\theta^*-\delta_0)^2    \tag{6.24}
$$
最终，对于后验均值，我们有
$$
\begin{aligned}
    MSE(\delta_{\kappa}|\theta^*) &= \mathbb{E}[(w\bar{x}+(1-w)\theta_0-\theta^*)^2] \\
    &= \mathbb{E}[(w(\bar{x}-\theta^*)+(1-w)(\theta_0-\theta^*))^2] \\
    &= w^2\frac{\sigma^2}{N} + (1-w)^2(\theta_0-\theta^*)^2 \\
    &= \frac{1}{(N+\kappa)^2}(N\sigma^2+\kappa^2(\theta_0-\theta^*)^2) \tag{6.25-6.28}
\end{aligned}
$$
![image](Figure6.3.png)
这些函数如图6.3，$N\in\{5,20\}$。我们看到最好的估计器依赖于$\theta^*$的值，而这个值是未知的。如果$\theta^*$离$\theta_0$很近，那么$\delta_3$是最好的。如果$\theta^*$位于围绕$\theta_0$的合理区域内，那么后验均值是最好的，该后验均值将$\theta_0$的先验猜测与实际数据进行组合。如果$\theta^*$远离$\theta_0$则MLE是最好的。这一切都不应该令人惊讶：假设我们的前一个平均值是合理的，通常需要少量收缩（使用后均值加上弱先验值）。

更令人感觉惊讶的是对于每个$\theta^*$的值，决策规则$\delta_2$的风险是一般总是高于$\delta_1$的。因此，采样中位数对于某些问题(数据来自高斯分布)不是容许估计器。

实际中，采样中位数通常好于采样均值，因为对异常值更加鲁棒。可以证明，如果我们假设数据来自一个Laplace分布，中位数是一个贝叶斯估计器，比高斯分布有肥尾。更一般的，我们可以使用数据的灵活模型来构造鲁棒的估计器，例如混合模型或是非参密度估计，然后计算后验均值或是中位数。

#### 6.3.3.2 斯坦悖论(Stein’s paradox)
## 6.4 Desirable properties of estimators

由于频率决策理论并没有提供一种自动选择最优估计器的方法，我们需要想出其他的启发式方法来选择其中的一个。在本节中，我们将讨论一些我们希望估计量具有的性质。不幸的是，我们将看到我们不能同时实现所有这些属性。

### 6.4.1 一致估计器(Consistent estimators)

当样本大小趋于无穷大时，如果估计器最终恢复产生数据的真实参数，也就是$\hat{\theta}(\mathcal{D})\rightarrow\theta^*, \vert\mathcal{D}\vert\rightarrow\infty$(箭头表示概率收敛)，则称其为一致的。当然，这个概念只有在数据实际来自参数$\theta^*$的指定模型时才有意义，然而实际数据通常不是这样。尽管如此，它依然是一个有用的理论性质。

可以证明MLE是一种一致性估计器。直觉原因是最大似然等效于最小化$\mathbb{KL}\left( p(\cdot|\boldsymbol{\theta}^*)\Vert p(\cdot|\hat{\boldsymbol{\theta}})  \right)$，其中$p(\cdot|\boldsymbol{\theta}^*)$是真是的分布，$p(\cdot|\hat{\boldsymbol{\theta}})$是我们的估计。当且仅当$\hat{\boldsymbol{\theta}}=\boldsymbol{\theta}^*$时，我们可以实现0 KL收敛。[^4]

### 6.4.2 Unbiased estimators

估计器的偏差定义为
$$
\text{bias}(\hat{\theta}(\cdot))=\mathbb{E}_{p(\mathcal{D}|\theta^*)}\left[ \hat{\theta}(\mathcal{D})-\theta_*  \right] \tag{6.32}
$$
其中$\theta_*$是真实的参数值。如果偏差为0，估计器为无偏的。这意味着采样分布是以真实参数中心化的。例如，一个高斯均值的MLE是无偏的
$$
\text{bias}(\hat{\mu})=\mathbb{E}[\bar{x}]-\mu=\mathbb{E}\left[\frac{1}{N}\sum_{i=1}^Nx_i   \right]-\mu=\frac{N_{\mu}}{N}-\mu=0 \tag{6.33}
$$
然而，一个高斯方差的MLE$\hat{\sigma}^2$不是$\sigma^2$一个无偏估计。事实上，我们可以证明
$$
\mathbb{E}[\hat{\sigma}^2]=\frac{N-1}{N}\sigma^2    \tag{6.34}
$$
然而如下估计器
$$
\hat{\sigma}_{N-1}^{2}=\frac{N}{N-1}\hat{\sigma}^2=\frac{1}{N-1}\sum_{i=1}^{N}(x_i-\bar{x})^2   \tag{6.35}
$$
是一个无偏估计器，我们可以轻松证明
$$
\mathbb{E}[\hat{\sigma}_2^{N-1}]=\mathbb{E}\left[\frac{N}{N-1}\hat{\sigma}^2\right]=\frac{N}{N-1}\frac{N-1}{N}\sigma^2=\sigma^2 \tag{6.36}
$$
尽管有时MLE是一种有偏估计，但是可以渐进的证明它总是无偏的。
尽管无偏听起来像一种期望的性质，但是真实情况不是这样的。

### 6.4.3 最小方差估计器(Minimum variance estimators)

直觉看起来好像我们的希望估计器无偏是很合理的。然而，只是无偏是不够的。例如，假设
我们估计一个来自$\mathcal{D}=\{x_1,\cdots,x_N\}$数据的均值。只看第一个数据点的估计量$\hat{\theta}(\mathcal{D})=x_1$是一个无偏估计量，但通常距离$\theta_*$远于经验平均值$\bar{x}$(这也是无偏的)。所以估计量的方差也很重要。

一个自然的问题是：方差可以维持多久？一个著名的结果叫做**Cramer-Rao下界**，它提供了任何无偏估计量方差的下界。更确切地说**理论6.4.1**令$X_1,\cdots,X_n\sim p(X|\theta_0)$且$\hat{\theta}=\hat{\theta}(x_1,\cdots,x_n)$是$\theta_0$的一个无偏估计。那么，在各种$p(X|\theta_0)$的平滑假设之下，我们有
$$
\text{var}[\hat{\theta}]\geq\frac{1}{nI(\theta_0)}  \tag{6.37}
$$
其中$I(\theta_0)$是Fisher信息矩阵。
可以证明MLE达到了Cramer-Rao的下界，因此有任何无偏估计器的最小的渐进方差。MLE是渐进最优的。

### 6.4.4 偏差-方差平衡(The bias-variance tradeoff)

虽然使用无偏估计似乎是个好主意，但情况并非总是如此。为了了解原因，假设我们使用二次型损失。如上所示，相应的风险是**MSE**。我们现在得到了一个非常有用的**MSE**分解。(所有的期望与方差是相对真实分布的$p(\mathcal{D}|\theta^*)$)，但是出于符号简化，我们抛弃清晰条件。令$\hat{\theta}=\hat{\theta}(\mathcal{D})$为估计，且$\bar{\theta}=\mathbb{E}[\hat{\theta}]$代表估计的期望值。那么我们有
$$
\begin{aligned}
    \mathbb{E}\left[(\hat{\theta}-\theta^*)^2\right] &= \mathbb{E}\left[     \left[ (\hat{\theta}-\bar{\theta}) + (\bar{\theta}-\theta^*) \right]^2\right] \\
    &= \mathbb{E}\left[(\hat{\theta}-\bar{\theta})^2\right] + 2(\bar{\theta}-\theta^*)\mathbb{E}\left[\hat{\theta}-\bar{\theta}\right] + (\bar{\theta}-\theta^*)^2\\
    &=\mathbb{E}\left[(\hat{\theta}-\bar{\theta})^2\right] + (\bar{\theta}-\theta^*)^2\\
    &=\text{var}\left[\hat{\theta}\right] + \text{bias}^2(\hat{\theta})
\end{aligned}
$$
[^5]
换而言之，$\text{MSE}=\text{variance}+\text{bias}^2$。
这称为**偏差-方差平衡(bias-variance tradeoff)**。它的意思是，如果我们的目标是最小化平方误差，只要减少方差，那么使用有偏估计量可能是明智的。

#### 6.4.4.1 Example: estimating a Gaussian mean

假设我们希望估计一个来自$\mathbf{x}=\{x_1,\cdots,x_N\}$的高斯分布的均值。我们假设数据采样来自$x_i\sim \mathcal{N}(\theta^*=1,\sigma^2)$。一个明显的估计是$\text{MLE}$。这个有一个为0的偏差，方差为
$$
\text{var}[\bar{x}|\theta^*]=\frac{\sigma^2}{N} \tag{6.43}
$$
但是我们也可以使用一个$\text{MAP}$估计。在4.6.1节中，我们证明在一个形如$\mathcal{N}(\theta_0, \sigma^2/\kappa_0)$的高斯先验下，$\text{MAP}$给定如下
$$
\tilde{x}\triangleq\frac{N}{N+\kappa_0}\bar{x} + \frac{\kappa_0}{N+\kappa_0}\theta_0 = w\bar{x}+(1-w)\theta_0\tag{6.44}
$$
其中$0\leq w \leq 1$控制相比先验我们可以相信多少$\text{MLE}$。(这也是后验均值，因为高斯分布的均值与众数相同。)偏差与方差给定为
$$
\begin{aligned}
    \mathbf{E}[\tilde{x}-\theta^*]&=w\theta_0+(1-w)\theta_0-\theta^*=(1-w)(\theta_0-\theta^*)\\
    \text{var}[\tilde{x}]&=w^2\frac{\sigma^2}{N}
\end{aligned}
$$
尽管$\text{MAP}$估计是有偏的(假设$w\lt1$)，但是它有低方差。
![image](Figure6.4.png)
我们假设我们的先验有点错误，使用我们使用$\theta_0$，因此真实的是$\theta^*=1$。在图6.4(a)中，我们看到$\kappa\gt0$的$\text{MAP}$估计的采样分布为偏离真实值的，但是比$\text{MLE}$估计的方差更低。

在图6.4(b)中，我们画出了$\text{mse}(\tilde{x})/\text{mse}(\bar{x})$。我们看到$\text{MAP}$估计比$\text{MLE}$估计的方差更低，尤其是对于小规模采样$\kappa\in\{1,2\}$。在$\kappa_0=0$的情况下，对应于$\text{MLE}$估计，在$\kappa_0=3$的情况下对应于一个强先验，因为先验均值是错误的，影响了业绩。显然，“调整”前一个主题的强度是很重要的，我们稍后将讨论这个主题。

### 6.4.4.2 Example: ridge regression
![image](Figure6.5.png)
另一偏差方差平衡的重要例子是岭回归，将在7.5节中讨论。简而言之，这个对应于高斯先验下线性回归的$\text{MAP}$估计,$p(\mathbf{x})=\mathcal{N}(\mathbf{w}|\mathbf{0},\lambda^{-1}\mathbf{I})$。零均值先验使得权重更小，将减少过拟合；精确项$\lambda$，控制了先验的强度。将$\lambda=0$导致了$\text{MLE}$；使用$\lambda\gt0$将导致一个有偏估计。为了考虑方差的影响，可以考虑一个简单的例子。图6.5画出了每个单独的拟合曲线，右边画出了平均拟合曲线。我们看到随着我们增加正则化的强度，方向不断下降，但是偏差上升。

#### 6.4.4.3 Bias-variance tradeoff for classification

如果我们使用0-1损失代替平方误差，上述分析将会失效，因为频率风险不再是可以描述为平方偏差加上方差。事实上，可以显示偏差与方差是相乘的形式组合的。如果估计是决策边界正确的一侧，那么偏差为负，降低方差将会提升误分率。但是如果估计在决策边界错误的一侧，偏差为正，因此为提升方差而付出代价。这个鲜为人知的事实说明，偏差-方差权衡对于分类并不是很有用。最好关注预期损失（见下文），而不是直接关注偏差和方差。如6.5.3节所述，我们可以使用交叉验证来近似估计预期损失。

## 6.5 Empirical risk minimization

频率决策理论遇到的基本问题是不能真实计算风险函数，因为它依赖于对真实数据分布的认知。(相反，**贝叶斯先验期望损失通常是可以计算的**，因为它是以数据为条件而不是以$\theta^*$为条件。)然而， 这里有一种设置避免了这个问题，那就是任务是预测可观测变量，而不是估计隐变量或是参数。这就是说，我们不是观测形式为$L(\boldsymbol{\theta},\delta(\mathcal{D}))$的损失函数(其中$\boldsymbol{\theta}$是真实的未知参数，$\delta(\mathcal{D})$是我们的估计)，我们是观测形式为$L(y,\delta(\mathbf{x}))$，其中$y$是真实的但是未知的响应变量，且$\delta(\mathbf{x})$是给定输入$\mathbf{x}$后的预测值。这种情况下，频率风险变为
$$
R(p_*,\delta)\triangleq\mathbb{E}_{(\mathbf{x},y)\sim p_*}\left[L(y,\delta(\mathbf{x}))\right]=\sum_{\mathbf{x}}\sum_{y}L(y,\delta(\mathbf{x}))p_*(\mathbf{x},y)    \tag{6.47}
$$
其中$p_*$展示了"自然分布"。当然，这个分布是未知的，但是一个简单的方法是使用经验分来近似$p_*$，经验分布来源于一些训练数据；也就是
$$
p_*(\mathbf{x},y)\approx p_{\text{emp}}(\mathbf{x},y)\triangleq\frac{1}{N}\sum_{i=1}^{N}\delta_{\mathbf{x}_i}(\mathbf{x}_i)\delta_{y_i}(y)  \tag{6.48}
$$
我们定义**经验风险**如下：
$$
R_{emp}(\mathcal{D},\mathcal{D})\triangleq R(p_{emp},\delta) = \frac{1}{N}\sum_{i=1}^{N}L(y_i, \delta(\mathbf{x}_i))    \tag{6.49}
$$
在0-1损失的情况下，$L(y,\delta(\mathbf{x}))=\mathbb{I}(y\neq \delta(\mathbf{x}))$，这个变为了**误分类率**。在平方误差损失的情况下，$L(y,\delta(\mathbf{x}))=(y-\delta(\mathbf{x}))^2$，这个变为了均方误差。我们定义**经验风险最小化**的任务为发现一个决策过程来最小化经验风险：
$$
\delta_{ERM}(\mathcal{D})=\argmin_{\delta}R_{emp}(\mathcal{D},\delta)   \tag{6.50}
$$
在非监督情况下，我们删除了所有对$y$的参考，将$L(y,\delta(\mathbf{x}))$替换为了$L(\mathbf{x},\delta(\mathbf{x}))$，其中，例如$L(\mathbf{x},\delta(\mathbf{x}))=\Vert \mathbf{x}-\delta(\mathbf{x})\Vert^2_2$，这个测量了重构误差。我们定义使用$\delta(\mathbf{x})=\text{decode}(\text{encode}(\mathbf{x}))$的决策规则，如图向量量化以及PCA中的那样。最终，我们定义经验风险最小化为
$$
R_{emp}(\mathcal{D},\delta) = \frac{1}{N}\sum_{i=1}^{N}L(\mathbf{x}_i, \delta(\mathbf{x}_i))    \tag{6.51}
$$
当然，我们总是可以通过设置$\delta(\mathbf{x})=\mathbf{x}$来最小化风险，因此通过以下bottleneck的编码-解码是很重要的。

### 6.5.1 Regularized risk minimization

注意到，如果我们关于"自然分布"的先验完全等于经验分布的，那么经验风险最小化等效于贝叶斯风险
$$
\mathbb{E}[R(p_*,\delta)|p_*=p_{\text{emp}}]=R_{emp}(\mathcal{D},\delta)    \tag{6.52}
$$

因此，最小化经验风险将会导致过拟合，因此通常需要往目标函数加一个复杂项惩罚：
$$
R^{\prime}(\mathcal{D},\delta)=R_{emp}(\mathcal{D},\delta)+\lambda C(\delta)    \tag{6.53}
$$
其中$C(\delta)$测量了预测函数$\delta(\mathbf{x})$的复杂度，$\lambda$控制了复杂项惩罚的强度。这种方法称为**正则化最小化风险(RRM)**。注意到损失是负对数似然函数，正则化项是一个负对数先验，这等效于MAP估计。

RRM中两个关键的问题是：如何测量复杂度，如果挑选$\lambda$。对于一个线性模型，我们可以根据自由度定义复杂度，将在7.5.3节中讨论。对于更一般的模型，我们可以使用VC维度，在6.5.4节中已经讨论了。为了挑选$\lambda$，我们可以使用6.5.2节中描述的方法。

### 6.5.2 Structural risk minimization

正则化风险最小化准则认为对于一个给定复杂度惩罚项的模型，我们应该使用
$$
\hat{\delta_{\lambda}} = \argmin_{\lambda}[R_{emp}(\mathcal{D},\delta)+\lambda C(\delta)]   \tag{6.54}
$$
但是我应该如何挑选$\lambda$呢？我们不能使用训练集，因为这将低估真实的风险，一个问题已知为**训练误差的最优化**。作为一种选择，我们可以使用如下规则，称为**结构化风险最小化**准则：
$$
\hat{\lambda}=\argmin_{\lambda}\hat{R}(\hat{\delta_{\lambda}})\tag{6.55}
$$
其中$\hat{R}(\delta)$是风险的估计。有两种常用的估计：交叉验证以及风险的理论上界。

### 6.5.3 Estimating the risk using cross validation
![image](Figure6.6.png)
我们可以使用交叉验证来估计一些估计器的风险。如果我们没有一个分割的验证集，我们可以使用交叉验证。更简洁的，CV定义如下。令训练集中有$N=\vert\mathcal{D} \vert$个数据情况。记数据中的$k$个测试集为$\mathcal{D}_k$，且所有的其他数据记为$\mathcal{D}_{-k}$。(在分层CV中，这些折叠的选择是为了类比例（如果存在离散标签)大致等于每个折叠)。令$\mathcal{F}$一个学校算法或是一个拟合函数，取一个数据集以及模型索引$m$，并返回一个参数向量
$$
\hat{\boldsymbol{\theta}}_m=\mathcal{F}(\mathcal{D},m)  \tag{6.56}
$$
最终，令$\mathcal{P}$为一个预测函数，使用一个输入与一个参数向量并返回一个预测：
$$
\hat{y}=\mathcal{P}(\mathbf{x},\hat{\boldsymbol{\theta}})=f(\mathbf{x},\hat{\boldsymbol{\theta}})   \tag{6.57}
$$
组合拟合循环为
$$
f_m(\mathbf{x},\mathcal{D})=\mathcal{P}(\mathbf{x},\mathcal{F}(\mathcal{D},m))  \tag{6.58}
$$
$f_m$的风险的K折CV估计定义为
$$
R(m,\mathcal{D},K)\triangleq \frac{1}{N}\sum_{k=1}^K\sum_{i\in\mathcal{D}_k}L(y_i,\mathcal{P}(\mathbf{x}_i,\mathcal{F}(\mathcal{D}_{-k},m)))    \tag{6.59}
$$
注意到每折我们只可以调用一次算法。令$f_{m}^k(\mathbf{x})=\mathcal{P}(\mathbf{x},\mathcal{F}(\mathcal{D}_{-k},m))$是在处k折测试数据之外所有训练数据上的函数。那么我们可以将CV估计重写为
$$
R(m,\mathcal{D},K)=\frac{1}{N}\sum_{k=1}^K\sum_{i\in\mathcal{D}_k}L(y_i,f_{m}^k(\mathbf{x}_i))=\frac{1}{N}\sum_{i=1}^N L(y_i, f_m^{k(i)}(\mathbf{x}_i))     \tag{6.60}
$$
其中$k(i)$是$i$作为测试数据使用的折数。换而言之，我们使用一个由不包含$\mathbf{x}_i$的数据训练得到的模型预测$y_i$。

对于$K=N$，该方法被称为**留一交叉验证(LOOCV)**。这种情况下，估计风险变为
$$
R(m,\mathcal{D},N)=\frac{1}{N}\sum_{i=1}^NL(y_i, f_m^{-i}(\mathbf{x}_i))    \tag{6.61}
$$
其中$f_m^i (\mathbf{x})=\mathcal{P}(\mathbf{x},\mathcal{F}(\mathcal{D}_{-i},m))$。这个需要拟合$N$次模型，其中$f_{m}^{-i}$我们丢弃调第$i$个训练情况。幸运的是，对于一些模型类似与损失函数，我们可以一次性拟合函数，并解析的移除第i训练情况的影响。这称为**广义交叉验证GCV**。

#### 6.5.3.1 Example: using CV to pick $\lambda$ for ridge regression

作为一个具体的例子，考虑挑选惩罚线性回归中$\ell_2$正则化的强度。使用如下规则
$$
\hat{\lambda}=\arg\min_{\lambda\in[\lambda_{min},\lambda_{max}]}R(\lambda,\mathcal{D}_{\text{train}},K)\tag{6.62}
$$
其中$[\lambda_{min},\lambda_{max}]$是$\lambda$值搜索的有限区间，且$R(\lambda,\mathcal{D}_{\text{train}},K)$是使用$\lambda$K折CV估计的风险，给定为
$$
R(\lambda,\mathcal{D}_{\text{train}},K)=\frac{1}{\vert\mathcal{D}_{\text{train}}\vert}\sum_{k=1}^{K}\sum_{i\in\mathcal{D}_k}L(y_i,f_{\lambda}^{k}(\mathbf{x}_i))\tag{6.63}
$$
其中$f_{\lambda}^k(\mathbf{x})=\mathbf{x}^T\hat{\mathbf{w}}_{\lambda}(\mathcal{D}_{-k})$是在除k折外的数据上训练的预测函数，且$\hat{\mathbf{w}}_\lambda(\mathcal{D})=\argmin_{\mathbf{w}}\text{NLL}(\mathbf{w},\mathcal{D})+\lambda\Vert \mathbf{w}\Vert^2_2$是后验估计。图6.6(b)给定了一个CV估计风险vs$\log(\lambda)$的例子，其中损失函数为平方误差。

当执行分类时，我们通常使用0-1损失。这种情况下，我们优化一个经验风险的凸上边界来估计$\mathbf{w}_{\lambda}$，但是我们优化风险本身相对估计$\lambda$。在估计$\lambda$时，我们可以处理非光滑的0-1损失函数，因为我们在整个（一维）空间中使用了暴力搜索。

当我们有一个或两个以上的调谐参数时，这种方法就变得不可行了。在这种情况下，**可以使用经验Bayes，这使得我们可以使用基于梯度的优化器而不是暴力搜索来优化大量的超参数**。详见第5.6节。 


### 6.5.5 Surrogate loss functions

在ERM/RRM框架中并不总是那么容易最小化损失。例如，我们可能想优化AUV或F1分数。或更简单的，我只是想最小化0-1损失，如图分类一样。不幸的是，0-1风险不是一个平滑的目标函数，因此很难被优化。一个选择是使用最大似然估计来替代，因为对数似然在0-1风险上是一个平滑的凸上界。

考虑一个二项logistics回归，令$y_{i}\in\{-1,+1\}$。假设决策函数计算对数几率比例
$$
f(\mathbf{x}_i)=\log \frac{p(y=1|\mathbf{x}_i,\mathbf{w})}{p(y=-1|\mathbf{x}_i,\mathbf{w})}=\mathbf{w}^T\mathbf{x}_i=\eta_i \tag{6.71}
$$
那么输出标签对应的概率分布为
$$
p(y_i|\mathbf{x}_i,\mathbf{w})=\text{sigm}(y_i\eta_i)
$$
我们定义对数损失为
$$
L_{nul}(y,\eta)=\log p(y|\mathbf{x},\mathbf{w})=\log(1+e^{-y\eta})\tag{6.73}
$$
很明显，最小化平均对数损失等效于最大化似然。

选择考虑计算最大概率标签，等效于使用$\hat{y}=-1$如果$\eta_i\lt0$且$\hat{y}=+1$如果$\eta_i\geq0$。我们函数的0-1损失变为
$$
L_{01}(y,\eta)=\mathbb{I}(y\neq \hat{y})=\mathbb{I}(y\eta\lt0)\tag{6.74}
$$
图6.7描绘了这两个损失函数。我们看到NLL确实是0-1损失的上限。

对数损失是一个替代损失函数的例子。另一个例子是Hinge损失。
$$
L_{hinge}(y,\eta)-\max(0,1-y\eta)   \tag{6.75}
$$
我们看到函数像一个门枢，因此得名。这个损失函数是一个非常流行的分类方法的基础，该方法称为支持向量机。

替代通常选择为一个凸上界，因为凸函数易于最小化。








[^1]:有时将参数视为代表真实(但未知)的物理量，因此不是随机的。但是，我们已经看到，使用概率分布来表示一个未知常数的不确定性是完全合理的。
[^3]:在实际应用中，频率分析方法通常只适用于一次性的统计决策问题，如分类、回归和参数估计，因为它的非构造性使得很难应用于适应在线数据的序列决策问题。
[^4]:如果模型是不可识别的，MLE可能会选择一个不同于真实参数的参数集合，但是诱导分布$p(\cdot|\hat{\boldsymbol{\theta}})$可能是与真实分布一样。这样的参数称为似然等效。
[^5]:方差定义：$\text{var}(x)=E[(x-\mu)^2]=E[x^2]-E[x]^2$，其中$\mu=E[x]$