---
layout: post
category: 机器学习基础
tags: [过拟合]
---


过拟合 Overfitting
================

## 造成overfitting的原因

> 我们举个开车的例子，把发生车祸比作成overfitting，那么造成车祸的原因包括：\\
  1.车速太快（VC Dimension太大);	\\
  2.道路崎岖（noise);	\\
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
	\epsilon ~ Gaussian(\underset{f(x)}{\sum_{q=0}^{Q_{f}}}\alpha _{q}X^{q},\sigma ^{2})
$$

### stochastic noise 随机性噪音

> σ2对overfit是有很大的影响

### deterministic noise 确定性噪音

> Qf即模型复杂度对overfit有很大影响。目标函数f(x)的复杂度很高，再好的hypothesis都会跟它有一些差距，所产生的noise。类似于一个伪随机数发生器，它不会产生真正的随机数，而只产生伪随机数。

## 维度灾难

> [维度灾难](https://blog.csdn.net/red_stone1/article/details/71692444)

##  避免overfitting的方法

> 避免overfitting的方法主要包括:
  1. start from simple model
  2. data cleaning/pruning
  3. data hinting
  4. regularization
  5. validataion

#### start from simple model

####

####

####

####

