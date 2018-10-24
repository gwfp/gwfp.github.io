---
layout: post
category: 机器学习基础
tags: [回归分析,简单线性回归,算法]
---

LinearRegression
===========

## 基本形式
$$
	f(x)=\omega ^{T}x+b	(0)
$$

## 简单线性回归 Simple Linear Regression

> y has to be explained/forecasted on the basis of one single independent variable, x.(y and x both are interval variables)

> 假设我们找到了最佳拟合直线方程 
$$
    y_{i} = ax_{i} + b, (1.0)
$$

> 对于每个样本点
$$
  (x^{i})     (1.1)
$$

>，线性试图回归学得
$$
        \hat{y}_{i} = a x_{i} +b 使得 \hat{y}_{i} \cong y_{i}  (1.2)	
$$

> 使用“最小二乘法” 使得均方差最小化
$$
	\sum_{i=1}^{m}(y_{i} - \hat{y}_{i})^2   (1.3)
$$
> 尽可能小

>  将（1.2）带入 （1.3）
$$
	J(a,b) = \sum_{i=1}^{m}(y_{i} - ax_{i}+b)^2  (1.4)
$$
> 即找到a,b使得(1.4)最小

> 分别对a和b进行求导

> 对b求导
$$
	\frac{\delta J(a,b)}{\delta b} = 0	(1.5)	\\
	=> \sum_{i=1}^{m}2(y_{(i)}-ax_{(i)}-b)(-1) = 0  \\
	=> \sum_{i=1}^{m}(y_{(i)}-ax_{(i)}-b) = 0        \\
	=> \sum_{i=1}^{m}(y_{(i)})-a\sum_{i=1}^{m}(x_{(i)})-\sum_{i=1}^{m}b = 0 \\	=> mb = \sum_{i=1}^{m}(y_{(i)})-a\sum_{i=1}^{m}(x_{(i)})  \\
	=> b = \bar{y}-a\bar{x}  （1.6）
$$

> 对a求导
$$
	\frac{\delta J(a,b)}{\delta a} = 0	(1.7)   \\
	=>  \sum_{i=1}^{m}2(y_{(i)}-ax_{(i)}-b)(-x_{(i)}) = 0	\\
	=> \sum_{i=1}^{m}(y_{(i)}-ax_{(i)}-b)x_{(i)} = 0	 （1.8）\\
	=> 将（1.6） 带入 （1.8）   \\
	=> \sum_{i=1}^{m}(y_{(i)}-ax_{(i)}-\bar{y}-a\bar{x})x_{(i)} = 0  \\
	=> a = \frac{\sum_{i=1}^{m}(x_{i}y_{(i)}-x_{(i)}\bar{y})}{\sum_{i=1}^{m}((x_{(i)})^2-\bar{x}x_{(i)})} 	(1.9.1) \\
	=> \sum_{i=1}^{m}x_{(i)}\bar{y} = \bar{y}\sum_{i=1}^{m}x_{(i)}=m\bar{y}\bar{x}= \bar{x}\sum_{i=1}^{m}y_{(i)}=\sum_{i=1}^{m}y_{(i)}\bar{x} \\
	=> m\bar{y}\bar{x}=\sum_{i=1}^{m}\bar{x}\bar{y} \\
	=> a = \frac{\sum_{i=1}^{m}(x_{i}y_{(i)}-x_{(i)}\bar{y}-\bar{x}y_{(i)}+\bar{x}\bar{y})}{\sum_{i=1}^{m}((x_{(i)})^2-\bar{x}x_{(i)}-\bar{x}x_{(i)}+\bar{x}^2)}  \\
	=> a = \frac{\sum_{i=1}^{m}(x_{i}-\bar{x})(y_{(i)}-\bar{y})}{\sum_{i=1}^{m}(x_{(i)}-\bar{x})^2} 	(1.9.2) 
$$

