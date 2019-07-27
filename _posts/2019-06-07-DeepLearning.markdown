---
layout: post
category: 深度学习概念
tags: [前馈神经网络]
---

深度学习概念
===============

## 深度前馈网络/前馈神经网络/多层感知机(deep feedforward network/feedforward neural net-work/multilayer perception,MLP)

## 神经元（neuron）

> 简单的加权和
$$
	y = \sum x_{i}\omega _{i}
$$

## 激活函数 （activation function）

> 通过激活函数f(), 引入非线形
$$
	y = f(\sum x_{i}\omega _{i})
$$

### 常用的激活函数

#### Sigmoid

$$
	f(x)=\frac{1}{1+e^{-z}}
$$

#### TanH

$$
	tanh(x)=\frac{2}{1+e^{-2x}}-1
$$

#### ReLU

$$
	f(x)=\left\{\begin{matrix}
		0, x<0 \\ 
		1,x\geqslant 0
		\end{matrix}\right.	
$$

## 神经元层

$$
	f(x)=f^{(3)}(f^{(2)}(f^{(1)}))
$$

### 输入层 (input Layer)

> f(1) 第一层

### 隐藏层 （hidden Layer）

> f(2) 第二层

### 输出层 （output Layer） 

> f(3) 第三层 , 前馈网络最后一层

### 深度

> 3 ，链的全长

### 全连接层 （fully-connected Layer，FC，Dense）

> 每一个神经元的输入都来自上一层的所有输出

## CNN 卷积神经网络

> [卷积网络(convolutional network)/卷积神经网络(convolutional neural network,CNN)](https://gwfp.github.io/深度学习算法/2019/06/07/CNN.html)

![avatar](https://gwfp.github.io/static/images/19/06/07/CNN.jpeg){:width='400px' height="200px"}

## 参考资料

[1] 伊恩古德费洛.深度学习[M].北京:人民邮电出版社.2017:34-51
[2] [深度学习中的常见名词术语(图像方向)](https://www.zhihu.com/lives/904295186979508224)
