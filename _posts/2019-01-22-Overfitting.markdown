---
layout: post
category: 机器学习基础
tags: [过拟合,正则化]
---


过拟合 Overfitting
================

## 造成overfitting的原因

> 我们举个开车的例子，把发生车祸比作成overfitting，那么造成车祸的原因包括：\\
  1.车速太快（VC Dimension太大);	\\
  2.道路崎岖（noise ~ stochastic & deterministic);	\\
  3.对路况的了解程度（训练样本数量N不够);	\\

> Ein 和Eout 表示为（d为模型阶次，N为样本数量）
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

#### data cleaning/pruning

> 对训练数据集里label明显错误的样本进行修正（data cleaning），或者对错误的样本看成是noise，进行剔除（data pruning）。

#### data hinting

> 对已知的样本进行简单的处理、变换，从而获得更多的样本。(N不够大且难以获取更多样本时使用)	\\
  如：数字分类问题，可以对已知的数字图片进行轻微的平移或者旋转，从而让N丰富起来，达到扩大训练集的目的。这种额外获得的例子称之为virtual examples。(新构建的virtual examples要尽量合理，且是独立同分布iid)

#### regularization

#### validataion

> Leave-One-Out

> V-Fold Cross(更常用)


## 参考内容

1.[林轩田机器学习基石(视频)](https://www.bilibili.com/video/av36731342/?p=50)
2.[林轩田机器学习>基石(笔记)](https://blog.csdn.net/red_stone1/article/details/72673204)
