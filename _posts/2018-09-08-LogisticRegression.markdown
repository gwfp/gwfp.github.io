---
layout: post
category: 机器学习基础
tags: [回归分析]
---

Logistic Regression
===============

### logistic回归分析

> 预测发生的概率，用发生的概率进行分类 （p>0.5 为1| p<0.5为0）

#### Logistic回归方程
$$
\widehat{p}=\sigma (\theta ^{T}\cdot x_{b})=\frac{1}{1+e^{-\theta ^{T}\cdot x_{b}}}
$$

#### 极大似然法(maximum likehood method)

> “使样本出现的可能性最大”

> 设样本出现的概率为P，令似然函数取最大值是的P，称作“极大似然估计值”

