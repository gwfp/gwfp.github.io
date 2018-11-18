---
layout: post
category: 机器学习基础
tags: [朴素贝叶斯分类器,算法]
---

朴素贝叶斯分类器 Naive Bayes Classifier
===========

## 贝叶斯定理

## 贝叶斯公式

> 在B 出现的前提下A出现的概率，等于A和B同时出现的概率除以B出现的概率。
$$
	P(A|B) = \frac{P(B|A)P(A)}{P(B)}
$$

$$
	对于 A =  {A_{1}, A_{2}, ... , A_{n}
$$

> 一般离散贝叶斯公式 
$$
	P(A|B) = \frac{P(B|A_{i})P(A_{i})}{\sum_{j=1}^{n}P(B|A_{j})P(A_{j})}
$$

> 一般连续贝叶斯公式
$$
	P(x|y) = \frac{f(y|x)f(x)}{\int_{-\infty }^{\infty }f(y|x)f(x)dx}
$$


