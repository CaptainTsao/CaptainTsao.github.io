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

## 5.3 Bayesian model selection
在图1.18中，我们看到使用过高的阶数会导致过拟合，而使用过低的阶数会导致欠拟合。类似的，在图7.8(a)中，我们看到使用很小的一个正则化参数导致过拟合，使用一个很大的值导致欠拟合。一般来说，当面对一组不同复杂度的模型(即参数分布族)时，我们应该如何选择最好的模型? 这称为\textbf{模型选择(model selection)}问题。
	
一种方法是使用交叉验证来估计所有候选模型的泛型误差，然后挑选看起来最优的模型。然而，这需要针对每个模型拟合$K$次，其中$K$是CV折的数量。一个更有效的方法是计算模型的后验，
$$
p(m\vert\mathcal{D}) = \frac{p(\mathcal{D}\vert m)p(m)}{\sum_{m\in\mathcal{M}}p(m, \mathcal{D})}	\tag{5.12}
$$
从这点可以看出，我们可以轻易计算MAP模型，$\hat{m}=\arg\max p(m	| \mathcal{D})$。这称为\textbf{贝叶斯模型选择}。
如果我们使用模型的一个均匀先验$p(m)\propto 1$，这等效于挑选一个最大化
$$
p(\mathcal{D}\vert m) = \int p(\mathcal{D}\vert \boldsymbol{\theta} )p(\boldsymbol{\theta}\vert m) d \boldsymbol{\theta} \tag{5.13}
$$
的模型。这个量称为**边缘似然(marginal likelihood)**，$\textbf{积分似然(integrated likelihood)}$或模型$m$的$\textbf{证据(evidence)}$。关于如何计算这个积分的细节在5.3.2%\fullref{(section5.3.2)}%
节中详细描述。但首先我们对这个数量的含义给出一个直观的解释。

### 5.3.2 Computing the marginal likelihood (evidence)}
当为一个固定模型讨论参数推理时，我们通常写作
$$
p(\boldsymbol{\theta}\vert \mathcal{D}, m) \propto p(\boldsymbol{\theta}\vert \ m) p(\mathcal{D} \vert \boldsymbol{\theta}, m) 	\tag{5.15}
$$
然后忽略归一化常数$p(\mathcal{D}\vert m)$。因为$p(\mathcal{D}\vert m)$相对$\boldsymbol{\theta} $是常数，所以有效。然而，当对比模型时，我们需要知道如何计算边缘似然$p(\mathcal{D}\vert m)$。一般而言，这是很困难的，因为我们需要计算所有可能的参数值，但是当我们有一个共轭先验时，就很容易计算了。

令$p(\boldsymbol{\theta})=q(\boldsymbol{\theta})/Z_0$是我们的先验，其中$q(\boldsymbol{\theta})$是未归一化的分布，且$Z_0$是先验的归一化常数。令$p(\mathcal{D} \vert \boldsymbol{\theta})= p(\mathcal{D} \vert \boldsymbol{\theta}) / Z_{\ell}$是似然，其中$ Z_{\ell}$是似然中包含的任何常量因子。最终，$p(\boldsymbol{\theta}\vert \mathcal{D}) = p(\boldsymbol{\theta}\vert \mathcal{D})/Z_N$是我们的后验，其中$	p(\boldsymbol{\theta}\vert \mathcal{D}) =  p(\boldsymbol{\theta}) p(\mathcal{D} \vert \boldsymbol{\theta}) $是未归一化的后验，$Z_N$是后验的归一化常数。我们有
$$
\begin{aligned}
p(\boldsymbol{\theta}\vert \mathcal{D}) &= \frac{p(\mathcal{D} \vert \boldsymbol{\theta})p(\boldsymbol{\theta})}{p(\mathcal{D})}  \\
\frac{q(\boldsymbol{\theta}\vert \mathcal{D}) }{Z_N} &= \frac{  p(\mathcal{D} \vert \boldsymbol{\theta})  p(\boldsymbol{\theta})   }{p(Z_{\ell} Z_0 \mathcal{D})} 		\\
p(\mathcal{D}) &= \frac{Z_N}{Z_0 Z_{\ell}}	
\end{aligned}\tag{5.18}
$$
因此，假设相关的归一化常数易于处理，我们就有了一种计算边际似然的简单方法。 我们在下面给出一些例子。

#### 5.3.2.1 Beta-binomial model
我们将上述结果应用到Beta-binomial模型中。因为我们知道$p(\theta\vert\mathcal{D})= \text{Beta} (\theta\vert a^{\prime}, b^{\prime}) $，其中$a^{\prime}=a+N_1, b^{\prime}=b+N_0$，我们知道后验的归一化常数为$B(a^{\prime}, b^{\prime}) $。因此
$$
\begin{aligned}
    p(\boldsymbol{\theta}\vert \mathcal{D}) &= \frac{p(\mathcal{D} \vert \boldsymbol{\theta})p(\boldsymbol{\theta})}{p(\mathcal{D})}  \\
&=\frac{1}{p(\mathcal{D})}\left[ \frac{1}{B(a, b)}	\theta^{a-1}(1-\theta)^{b-1}	\right] \left[  \begin{pmatrix}
N \\ N_1
\end{pmatrix} \theta^{N_1}(1-\theta)^{N_0}  \right] 	 \\
&=\begin{pmatrix}
N \\ N_1
\end{pmatrix}   \frac{1}{p(\mathcal{D})} \frac{1}{B(a, b)} \left[  \theta^{a+N_1-1}(1-\theta)^{b+N_0-1}	\right]  
\end{aligned}    \tag{5.21}
$$
所以
$$
\begin{aligned}
    \frac{1}{B(a+N_1, b+N_0)} &= \begin{pmatrix} N \\ N_1 \end{pmatrix}  \frac{1}{p(\mathcal{D})}  \frac{1}{B(a, b)} \\
p(\mathcal{D}) &=  \begin{pmatrix} N \\ N_1 \end{pmatrix} \frac{B(a+N_1, b+N_0)} {B(a, b)}
\end{aligned}
$$
Beta-binomial模型的边缘似然与上述一样，除了丢弃了$ \begin{pmatrix} N \\ N_1 \end{pmatrix}$项。

#### 5.3.2.2 Dirichlet-multinoulli model
通过与Beta-binomial一样的推理，可以证明Dirichlet-multinoulli的边缘似然的给定为
$$
p(\mathcal{D}) = \frac{B(\mathbf{N}+\boldsymbol{\alpha})}{B(\boldsymbol{\alpha})}		\tag{5.24}
$$
其中
$$
B(\boldsymbol{\alpha})	= \frac{\prod_{k=1}^{K}\Gamma(\alpha_k)}{\Gamma(\sum_k\alpha_k)  }	\tag{5.24}
$$

#### 5.3.2.3 Gaussian-Gaussian-Wishart model}
考虑一个共轭先验为$\text{NIW(Normal-inverse-wishart)}$的MVN情况。令$Z_0$是先验的归一化常数，$Z_N$是后验的归一化常数，且令$ Z_{\ell}=(2\pi)^{ND/2} $是似然的归一化常数。那么很容易看到
$$
\begin{aligned}
    p(\mathcal{D}) &= \frac{Z_N}{Z_0 Z_{\ell}}	\\
&= \frac{1}{(2\pi)^{ND/2} }\frac{(\frac{2\pi}{\kappa_N})^{D/2}\vert \mathbf{S}_N\vert^{-\nu_N/2}2^{(\nu_0+N)D/2}\Gamma_D(\nu_N/2)}{(\frac{2\pi}{\kappa_0})^{D/2}\vert \mathbf{S}_0\vert^{-\nu_0/2}2^{\nu_0D/2}\Gamma_D(\nu_0/2)}  		\\
&=\frac{1}{(2\pi)^{ND/2} }\left(  \frac{\kappa_0}{\kappa_N} \right) ^{D/2} \frac{\vert \mathbf{S}_0\vert^{\nu_0/2}}{\vert \mathbf{S}_N\vert^{\nu_N/2}} \frac{\Gamma_D(\nu_N/2)}{\Gamma_D(\nu_0/2)} \tag{5.29}	
\end{aligned}
$$

#### 5.3.2.4 BIC approximation to log marginal likelihood
一般的，计算等式\ref{5.13}中的积分是非常困难的。一个简单的但是常用的近似称为$\textbf{贝叶斯信息准则(\text{Bayesian information criterion/BIC}})$。有如下形式
$$
\text{BIC} \triangleq \log p(\mathcal{D}\vert \hat{\theta}) - \frac{\text{dof}(\hat{\boldsymbol{\theta}})}{2}\log N \approx \log p(\mathcal{D})\tag{5.30}
$$
其中$ \text{dof}(\hat{\boldsymbol{\theta}}) $是模型中的自由度的数，$\hat{\boldsymbol{\theta}}$是模型的最大似然。我们看到这具有惩罚对数似然的形式，其中惩罚项取决于模型的复杂性。BIC 分数的推导见第8.4.2节。

## 5.5 Hierarchical Bayes

计算后验$ p(\boldsymbol{\theta}\vert\mathcal{D}) $的一个关键要求是指定一个先验$ p(\boldsymbol{\theta}\vert\boldsymbol{\eta}) $，其中$ \boldsymbol{\eta} $是超参。如果我们不知道如何设置$ \boldsymbol{\eta} $呢？在某些情况下，我们应该使用无信息先验。一个更加贝叶斯的方法是在我们的先验上面再添加一个先验！用图模型表示的化，我们将这种情况表示为如下
$$
\begin{align*}
	\boldsymbol{\eta} \mapsto \boldsymbol{\theta} \mapsto	\mathcal{D}	\tag{5.76}
\end{align*}
$$
这是$\textbf{阶层贝叶斯模型(hierarchical Bayesian model)}$的一个例子，也称为$\textbf{多层贝叶斯模型( multi-level model)}$，因为有多层的未知量。我们给定一个简单的例子，

### 5.5.1 Example: modeling related cancer rates
考虑预测各个城市中癌症比例的情况。尤其，假设我们测量了在各种城市$ N_i $，以及在各城市中思域癌症的数量$ x_i $。我们假设$ x_i\sim\text{Bin}(N_i, \theta_i) $，且我们想估计癌症比例$ \theta_i $。一个方法是单独估计它们，但是这会遇到稀疏数据问题。另一个方法是假设所有的$ \theta_i $是相同的；这称为\textbf{参数绑定(parameter tying)}。得到的似然只是$ \hat{\theta}=\frac{\sum_i x_i}{\sum_i N_i} $。但是假设所有的城市都有相同的比例是一个很强的假设。一个折中的办法是假设所有的$ \theta_i $都是相似的，但是可能根据城市会变化。这个可以通过假设$ \theta_i $服从一些常见分布，如$ \theta_i\sim \text{Beta}(a,b) $。完全联合分布可以写为
$$
\begin{align*}
p(\mathcal{D} \vert \boldsymbol{\eta},\boldsymbol{\theta}, N) = p(\boldsymbol{\eta})\prod_{i=1}^N \text{Bin}(x_i\vert N_i,\theta_i)\text{Beta}(\theta_i\vert\boldsymbol{\eta})  \tag{5.77}
\end{align*}
$$


## 5.6 Empirical Bayes
在阶层贝叶斯模型中，我们需要计算隐变量在多层级的后验。例如，在二层模型中，我们需要计算
$$
p(\boldsymbol{\eta},\boldsymbol{\theta}\vert \mathcal{D}) \propto p(\mathcal{D}\vert \boldsymbol{\theta}) p(\boldsymbol{\theta} \vert \boldsymbol{\eta} )p(\boldsymbol{\eta})  \tag{5.78}
$$
在某些情况中，我们解析的边缘化$\boldsymbol{\theta}$；剩下的就是简单的计算$p(\boldsymbol{\eta} \vert\mathcal{D})$问题。

作为计算的捷径，我们可以用点估计来逼近超参数的后验值，$p(\boldsymbol{\eta} \vert\mathcal{D})\approx \delta_{\hat{\boldsymbol{\eta}}}(\boldsymbol{\eta})$，其中$\hat{\boldsymbol{\eta}} = \argmax p(\boldsymbol{\eta\vert\mathcal{D}})$。因为$\boldsymbol{\eta}$在维度一般要比$\boldsymbol{\theta}$小很多，不易过拟合，所以我们可以安全的在$\boldsymbol{\eta}$上使用均匀分布。那么估计变为
$$
\hat{\boldsymbol{\eta}} = \argmax p(\mathcal{D}\vert\boldsymbol{\eta}) = \argmax \left[\int p(\mathcal{D}\vert \boldsymbol{\theta}) p(\boldsymbol{\theta} \vert \boldsymbol{\eta} )p(\boldsymbol{\eta}) \right]  \tag{5.79}
$$
括号内的数量是边缘或积分似然，有时称为**证据/evidence**。这种总体方法被称为经验贝叶斯(empirical Bayes/EB)或II型最大似然法。在机器学习中，它有时被称为**证据过程/evidence procedure**。

经验贝叶斯违背了先验要独立于数据进行选择的原则。然而，我们可以将其视为在阶层贝叶斯模型中推理的一个低计算成本的近似；就如图将MAP估计看作在单层模型$\boldsymbol{\theta}\mapsto\mathcal{D}$中推理的一个简单近似。事实上，我们可以构建一个阶层，其中更多的积分执行一次，

## 5.7 Bayesian decision theory(贝叶斯决策理论)

我们现在看概率论如何用来表示并更新我们关于世界状态的信念。然而，最终我们的目标是将我们的信念转化为行为。本节中，我们讨论这种行为的最优方式。

我们可以将任何给定的**统计决策问题形**式化为与自然的**博弈**(而不是与其他战略参与者的博弈，这是博弈论的主题，详见(Shoham和Leyton Brown 2009))。博弈中，自然挑选一个状态或是参数或是label，$y\in\mathcal{Y}$，这些对我们是未知的，然后产生一个观测$\mathbf{x}\in\mathcal{X}$，我们可以观测到的。我们然后必须做出一个决策，这就是我们必须从一些行为空间**action space**$\mathcal{A}$中选的一个行为$a$。最后我们蒙受了一些**损失**$L(y,a)$，这衡量了我们的行为a与自然界隐藏状态y的相容性。

我们的目标是设计一个**决策过程**或是**策略**$\delta:\mathcal{X}\mapsto\mathcal{A}$，指定了对于每个可能输入的最优行为。通过最优，意味着行为最小化了期望损失：
$$
\delta(\mathbf{x}) = \argmax_{a\in\mathcal{A}} \mathbb{E}[L(y,a)]   \tag{5.96}
$$
在经济学中，最常用的是用**效用函数**；也就是负的损失$U(y,a)=-L(y,a)$。那么上述规则变为
$$
\delta(\mathbf{x}) = \argmin_{a\in\mathcal{A}} \mathbb{E}[U(y,a)]   \tag{5.97}
$$