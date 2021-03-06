[toc]

# 5 Bayesian statistics

## 5.1 Introduction
我们已经讨论了大量不同的概率模型，我们已经讨论了如何使用它们拟合数据，也就是讨论了如何使用各种不同的先验计算MAP参数估计$\hat{\boldsymbol{\theta}} = \argmax p(\boldsymbol{\theta}\vert \mathcal{D})$。我们也讨论了针对特特定情况如何计算完全后验$p(\boldsymbol{\theta}\vert \mathcal{D})$，以及后验预测密度$p(\mathbf{x}\vert \mathcal{D})$。

利用后验分布来总结我们所知道的关于一组未知变量的一切，是**贝叶斯统计**的核心。在本章中，我们将更详细地讨论这种统计方法。在第6章中，我们讨论了一种被称为频度统计或经典统计的替代方法。 

## 5.2 Summarizing posterior distributions

后验$p(\boldsymbol{\theta}\vert \mathcal{D})$总结了关于未知量$\theta$的所有已知信息。在本节中，我们讨论一些可以从概率分布中得出的简单量，例如后验。这些摘要统计信息通常比完整联合统计(full joint)信息更易于理解和可视化。

## 5.2.1 MAP estimation

我们可以通过后验均值、中位数或众数来计算一个未知变量的**点估计**。在5.7节中，我们讨论如何使用决策理论选择这些方法。**一般对于实值量(real-valued quantity)，后验均值或中位数是最恰当的选择，后验边缘的向量对于离散变量时最佳的选择**。然而，后验众数，类似MAP估计，是最流行的选择，因为它可以简化为优化问题，而对于该问题，通常存在有效的算法。此外，通过将对数先验视为正则器，可以用非贝叶斯术语来解释MAP估计(有关更多详细信息，请参见第6.5节)。

尽管此方法在计算上很有吸引力，但必须指出，MAP估计存在各种缺点，我们将在下面简要讨论。 这将为更彻底的贝叶斯方法提供动力，我们将在本章的后面（以及本书的其他地方）进行研究。

#### 5.2.1.1 No measure of uncertainty
MAP估计以及任何其他点估计(如后验平均值或中位数)最明显的缺点是，它不提供任何不确定性度量。在许多应用中，了解一个给定的估计值有多可信是很重要的。如第5.2.2节所述，我们可以从后验概率中得出此类置信度。

#### 5.2.1.2 Plugging in the MAP estimate can result in overfitting
在机器学习中，我们通常更加关心预测的精确度，而不是我们模型的参数的可解释性。然而， 如果我们没有建模参数的不确定性，那么我们的预测分布将会overconfident。
#### 5.2.1.3 The mode is an untypical point
通常选择mode作为后验分布的一个总结是一个不好的选择，因为mode确实不是分布的代表，不像均值与中位数。基本问题是mode是一个测量零点的点，而平均值和中值考虑了空间的体积。另一个例子如图5.1(b)所示：这里的模式为0，但平均值不为零。这种偏态分布在推断方差参数时经常出现，尤其是在层次模型中。在这种情况下，MAP估计(以及MLE)显然是一个非常糟糕的估计。

如果mode不是一个好的选择，我们应该如何总结一个后验？答案是使用决策理论，我们将在第5.7节中讨论。基本思想是指定一个损失函数，其中$L(\theta, \hat{\theta})$是如果真实是$\theta$估计是$\hat{\theta}$时引起的损失。如果我们使用0-1损失，$L(\theta, \hat{\theta})=\mathbb{I}(\theta\not ={\hat{\theta}})$，那么最优估计是后验众数。0-1损失意味着如果你没有犯错你将得到一个点，要不然你什么都得不到。对于连续值量，我们更喜欢使用平方误差损失，$L(\theta,\hat{\theta})=(\theta-\hat{\theta})^2$；对应的最优估计器是后验均值。或是使用更加鲁棒的损失函数$L(\theta,\hat{\theta})=\vert \theta-\hat{\theta} \vert$，得到后验中位数。

#### 5.2.1.4 MAP 估计对于重新参数化不是一成不变的(MAP estimation is not invariant to reparameterization *)
MAP估计一个更加微妙的问题是我们得到的结果依赖于我们如何参数化概率分布。如果从一个表示变化到另一个等效的表示会改变结果，这不是我们期望得到的，因为测量单元是任意的。

为了理解这个问题，我们假设计算$x$的后验。如果我们定义$y=f(x)$，$y$的分布，
$$
p_y(y) = p_x(x) \lvert \frac{dx}{dy}    \rvert      \tag{5.1}
$$
$\lvert \frac{dx}{dy}    \rvert$项称为Jacobian，其测量了一个单位的体积通过$f$后尺寸的改变。令$\hat{x}=\argmax_x p_x(x)$是$x$的MAP估计。一般情况下，给定$f(\hat{x})$后并非$\hat{y}=\argmax_y p_y(y)$。例如，令$x\sim \mathcal{N}(6, 1)$且$y=f(x)$，其中
$$
f(x) = \frac{1}{1+\exp(-x + 5)} \tag{5.2}
$$
我们可以使用蒙特卡洛模拟来推演$y$的分布。结果在图5.2中有显示。我们看到原始的高斯分布通过sigmoid非线性后变成了"挤压"形状。尤其是转换分布的mode不再等于原始mode。

要了解这个问题是如何在MAP估计的上下文中出现的，请考虑以下示例，由于Michael Jordan。伯努利分布是典型的由其均值$\mu$参数化的分布，$p(y=1\vert \mu)=\mu,y\in\{0,1\}$。假设我们在单位间隔$p_{\mu}(\mu)=1\mathbb{I}(0\leq \mu \leq 1)$上有一个均匀分布。如果没有数据，MAP估计只是先验的mode，可以是0到1之间的任何位置。我们现在将展示不同的参数化可以在这个区间内任意选择不同的点。

首先，令$\theta = \sqrt{\mu}$所以$\mu=\delta^2$。新的先验是
$$
p_{\theta}(\theta) = p_{\mu}(\mu)\lvert \frac{d\mu}{d\theta}\rvert = 2\theta    \tag{5.3}
$$

5.3 Bayesian model selection
