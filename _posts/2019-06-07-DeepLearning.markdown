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
