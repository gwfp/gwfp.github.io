---
layout: post
category: 机器学习概念
tags: [过拟合,正则化,台大-林轩田]
---


过拟合 Overfitting
================

## 造成overfitting的原因

> 我们举个开车的例子，把发生车祸比作成overfitting，那么造成车祸的原因包括：\\
  1.车速太快（VC Dimension太大);	\\
  2.道路崎岖（noise ~ stochastic & deterministic);	\\
  3.对路况的了解程度（训练样本数量N不够);	\\

> Ein 和Eout 表示为（d为模型阶次，N为样本数量),Ein很小，但是Eout很大，造成了过拟合现象。
$$
	E_{in}=noiselevel*(1-\frac{d+1}{N})	\\
	E_{out}==noiselevel*(1+\frac{d+1}{N})
$$

## 两种Noise

> 假设我们产生的数据分布由两部分组成:	\\
  1. 第一部分是目标函数f(x)，Qf阶多项式；	\\
  2. 第二部分是噪声ϵ，服从Gaussian分布。	\\
  (噪声水平σ2, 总数据量是N)
$$
	y = f(x) + \epsilon 	\\
	\epsilon \sim  Gaussian(\underset{f(x)}{\sum_{q=0}^{Q_{f}}}\alpha _{q}X^{q},\sigma ^{2})
$$

### stochastic noise 随机性噪音

> σ2对overfit是有很大的影响

### deterministic noise 确定性噪音

> Qf即模型复杂度对overfit有很大影响。目标函数f(x)的复杂度很高，再好的hypothesis都会跟它有一些差距，所产生的noise。类似于一个伪随机数发生器，它不会产生真正的随机数，而只产生伪随机数。

## 维度灾难

> [维度灾难](https://blog.csdn.net/red_stone1/article/details/71692444):
  随着维度的增加，分类器性能逐步上升，到达某点之后，其性能便逐渐下降.

##  避免overfitting的方法

> 避免overfitting的方法主要包括(继续以车祸为例):
  1. start from simple model（driver slowly）
  2. data cleaning/pruning (use more accurate road information)
  3. data hinting (exploit more road information)
  4. regularization (put the brakes踩刹车)
  5. validataion (monitor the dashboard控制仪表盘)

#### start from simple model

> 多项式回归

#### data cleaning/pruning

> 对训练数据集里label明显错误的样本进行修正（data cleaning），或者对错误的样本看成是noise，进行剔除（data pruning）。

#### data hinting

> 对已知的样本进行简单的处理、变换，从而获得更多的样本。(N不够大且难以获取更多样本时使用)	\\
  如：数字分类问题，可以对已知的数字图片进行轻微的平移或者旋转，从而让N丰富起来，达到扩大训练集的目的。这种额外获得的例子称之为virtual examples。(新构建的virtual examples要尽量合理，且是独立同分布iid)

#### regularization

> L1
$$
	\Omega (\omega ) = \sum_{q=0}^{Q}\omega ^2_{q}=\left \| \omega  \right \|_{2}^{2}
$$

> L2 (一般比较通用)
$$
	\Omega (\omega ) = \sum_{q=0}^{Q} | \omega _{q}|=\left \| \omega  \right \|_{1}
$$  

#### validataion

> 1.先将D分成两个部分，一个是训练样本Dtrain，一个是验证集Dval。	\\
  2.若有M个模型，那么分别对每个模型在Dtrain上进行训练，得到矩g−m。	\\
  3.再用Dval对每个g−m进行验证，选择表现最好的矩g−m∗，则该矩对应的模型被选择。\\
  4.最后使用该模型对整个D进行训练，得到最终的gm∗。	\\
  (M种模型hypothesis set，Dval的数量为K)
$$
	E_{out}(g_{m^{*}})\leqslant E_{out}(g_{m^{*}}^{-})\leqslant E_{out}(g_{m^{*}}^{-})+\sqrt{\frac{logM}{K}}	\\
$$
模型个数M越少，测试集数目越大，那么
$$
	\sqrt{\frac{logM}{K}}
$$
越小，即Etest(gm*)越接近于Eout(gm*),通常设置k=N/5。

> Leave-One-Out Cross Validation 留一法交叉验证	\\
  1.k=1，也就是说验证集大小为1，即每次只用一组数据对gm进行验证。\\
  2.每次从D中取一组作为验证集，直到所有样本都作过验证集，共计算N次，最后对验证误差求平均。	//
$$
	E_{loocv}(H,A)=\frac{1}{N}\sum_{n=1}^{N}e_{n}=\frac{1}{N}\sum_{n=1}^{N}err(g_{n}^{-}(x_{n}),y_{n})
$$

> V-Fold Cross V-折交叉验证 (更常用)
  1.将N个数据分成V份,计算过程与Leave-One-Out相似。\\
  2.计算过程与Leave-One-Out相似。(简少了计算量)		\\
$$
	E_{cv}(H,A)=\frac{1}{V}\sum_{v=1}^{V}E_{val}^{(V)}(g_{V}^{-})
$$

## 参考内容

1.[林轩田机器学习基石(视频)](https://www.bilibili.com/video/av36731342/?p=50)
2.[林轩田机器学习基石(笔记)](https://blog.csdn.net/red_stone1/article/details/72673204)
