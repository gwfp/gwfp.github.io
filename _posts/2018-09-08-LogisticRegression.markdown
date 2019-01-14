---
layout: post
category: 机器学习基础
tags: [回归分析,逻辑回归,算法]
---

Logistic Regression
===============

## logistic回归分析

> 预测发生的概率，用发生的概率进行分类 （p>0.5 为1| p<0.5为0）

### Logistic回归方程

> 逻辑回归方程，表示的是 存量 随时间增长渐增的关系。

> 一元
$$
	\hat{p}=\frac{1}{1+e^{-(a+bx)}}
$$

> 多元
$$
\widehat{p}=\sigma (\theta ^{T}\cdot x_{b})=\frac{1}{1+e^{-\theta ^{T}\cdot x_{b}}}
$$

### Logistic回归的损失函数

$$
	J(\theta )= -\frac{1}{m}\sum_{i=1}^{m}y^{(i)}log(\hat{p}^{(i)})+(1-y^{(i)})log(1-\hat{p}^{(i)}
$$

### Logistic回归的梯度下降函数

$$
	\bigtriangledown J(\theta )=\frac{1}{m}\cdot X_{b}^{T}\cdot (\sigma (X_{b}\theta )-y)
$$

### 极大似然法(maximum likehood method)

> “使样本出现的可能性最大”

> 设样本出现的概率为P，令似然函数取最大值是的P，称作“极大似然估计值”

### 决策边界 Decision Boundary

$$
	\hat{p}=\sigma (\theta ^{T}\cdot x_{b})=\frac{1}{1+e^{-\theta ^{T}\cdot x_{b}}}		\\
	\hat{y} = 1 ==> \hat{p} >= 0.5 ==> \theta ^{T}\cdot x_{b} >= 0	\\
	\hat{y} = 1 ==> \hat{p} < 0.5 ==> \theta ^{T}\cdot x_{b} < 0  \\
	决策边界：\theta ^{T}\cdot x_{b} = 0
$$


