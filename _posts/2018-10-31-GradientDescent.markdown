---
layout: post
category: 机器学习基础
tags: [梯度下降,算法]
---

梯度算法 Gradient 
================

> 不是一个机器学习算法,是一种基于搜素的最优化方法

> 梯度下降法:最小化一个损失函数

> 梯度上升法:最大化一个效用函数

## 梯度下降法 Gradient Descent

> 梯度代表放向，对应J增大最快的方向
$$
	-\eta \bigtriangledown J   (0.1) \\
	 \bigtriangledown J = (\frac{\partial J}{\partial \theta_{0}},\frac{\partial J}{\partial \theta_{1}},...,\frac{\partial J}{\partial \theta_{n}})         (0.2) \\	
$$

### 线性回归中使用梯度下降法

> 目标：使
$$
	\sum_{i=1}^{m}(y_{i} - \hat{y}_{i})^2	(1.1)
$$
> 尽可能小

$$
	\hat{y}_{i} = \theta _{0} + \theta _{1}X_{1}^{(i)}+\theta _{2}X_{2}^{(i)}+...+\theta _{n}X_{n}^{(i)}	(1.2)
$$

> (1.2)带入(1.1)使 （1.3）尽可能小
$$
	\sum_{i=1}^{m}(y^{i} - \theta _{0} + \theta _{1}X_{1}^{(i)}+\theta _{2}X_{2}^{(i)}+...+\theta _{n}X_{n}^{(i)})	(1.3)
$$

> 对(1.3)求导
$$
	J(\theta ) = MSE(y,\hat{y})=\frac{1}{m}\sum_{i=1}^{m}(y_{i} - \hat{y}_{i})^2 
$$



