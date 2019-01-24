---
layout: post
category: 机器学习概念
tags: [VC-Dimension,VC-维度]
---


VC-维度 Vapnik-Chervonenkis Dimension
================

## 霍夫丁不等式 hoffding's Inequality

> 对整体进行抽样，当样本N足够大的时候，样本数据中的比例 ν 与整体数据中比例 μ 会很接近。其中 ε 表示容忍误差（这个参数人为规定的 0<ε<1，你能容忍多少误差就写多少）
$$
	P\left [ \left | \nu - \mu  \right | > \epsilon  \right ] \leqslant 2 e^{-2 \epsilon N}
$$

## 霍夫丁不等式 与 机器学习

> for any fix h,in N large data, Ein(h) is probably close to Eout(h), (Ein(h) is in-sample error, Eour(h) is out-of-sample error)
$$
	P\left [ \left | E_{in}(h) - E_{out}(h)  \right | > \epsilon  \right ] \leqslant 2 e^{-2 \epsilon N}
$$ 
> valid for all N and \\epsilon , no need to know Eout(h), 'Ein(h)=Eout(h)'is probably approximately correct(PAC)

  
## 当数据整体是有限的

> 有M个训练样本(hypothesis)，对于抽样的D中(D1表示对每个Hypothesis进行的第一次抽样),存在BAD数据的概率为P[BAD D]
$$
	P_{D} = [D_{BAD}]	\\
	      = P_{D}[D1_{BAD}^{H_{1}} or D2_{BAD}^{H_{2}} or ... or DM_{BAD}^{H_{M}}]	\\
	      \leqslant  P_{D}[D1_{BAD}^{H_{1}}]+P_{D}[D2_{BAD}^{H_{1}}]+...+P_{D}[DM_{BAD}^{H_{M}}]	\\
	      \leqslant  2e^{-2\epsilon ^{2}N}+2e^{-2\epsilon ^{2}N}+...+2e^{-2\epsilon ^{2}N} 		\\
	      = 2Me^{-2\epsilon ^{2}N}	(1) 
$$

> 样本M情况下，接受一定忍受度情况下，需要最小的N(样本)
$$
	忍受度为\delta，错误差为\epsilon，样本数为M	\\
	N=\frac{1}{2\epsilon^{2}}ln\frac{2M}{\delta } 
$$

## 当数据整体是无限的

> 1.公式(1)使用了union bound方法，缺点是没有考虑到每个抽样中BAD重叠的问题。当整体数据量无限大时，会导致上限过分大。\\
  2.如果两个hypothesis非常接近(h1,h2),那么有很大几率Ein(h1)=Ein(h2) \\
  3.将这些hypothesis归类。

### effective number of line

> 一般情况下有效的分类线会小于M，所以现在我们可以做出一个重要的式子变换，用effective(N)来代替公式(1)中的M。
$$
	P[|E_{in}(g)-E_{out}(g)|>\epsilon ]\leqslant 2\cdot M\cdot e^{-2\epsilon ^{2}N}	\\
	=> P[|E_{in}(g)-E_{out}(g)|>\epsilon ]\leqslant 2\cdot effective(N)\cdot e^{-2\epsilon ^{2}N}	\\
	\\
	effective(N) < 2^{N}
$$
当N足够大时候
$$
	2\cdot effective(N)\cdot e^{-2\epsilon ^{2}N}
$$
趋向于0.说明
$$
	P[|E_{in}(g)-E_{out}(g)|>\epsilon]
$$
发生坏事的几率很低。

### 成长函数  Growth Function
 
> 1.假设平面上所有的线都是一个hypothesis。（H' = {all lines in R^2}），此时有无限多条线，也就有无限多个hypothesis。对平面上N个点进行分类，将Ein=Eout的hypothesis进行合并形成dichotomy，dichotomies根据平面上点的不同进行取值，此时dochotomies上限为2^N.	\\
  2.使用mH'(N)替换公式（1）中的M \\
  3.mH'(N) 就是成长函数，hypothesis在N个点上能产生多少dichotomy。

#### growth function
> 1.从点出发来看，得到线的的种类是有限的(maximum kinds of lines with respect to N inputs x1, x2,...,xn <=> effective number of lines)	\\
  2.某个数量的点可能在不同情况下，出现不同的分割种类的数量（三个点不共线时一条线可有8种不同的情形将点分割为两个部分，但共线情况下有两种情况无法优一条线分割，所以只有6种情况），以最多的那种情况计算(三个点时记8种)。
$$
	m_{H'}(N)=\underset{x_{1},x_{2}...x_(N)}{max}\left | H'(x_{1},x_{2}...x_(N)) \right |
$$

### 成长函数中的Break Point

> 从第k个点开始，无法做出所有的dichotomy,即开始出现某种点排列的情形无法用一个dichotomy分割(if no k inputs can be shattered by H',call k a break posint for H).
$$
	m_{H'}(k) < 2^{k}
$$

> break point 与 成长函数的复杂度有关:从这个点开始，增长函数增速开始放缓了。
$$
	m_{H'}(N) = O(N^{k-1})
$$

### Recap:More on Grow Function
> when break point = k.	\\
$$
	\sum_{i=0}^{k-1}C_{n}^{i}\leqslant N^{k-1}
$$

> For any g=A(D) in H' and 'statistical' large D.if k exists and k>3:
$$
	P_{D}[|E_{in}(g)-E_{out}(g)|>\epsilon ]	\\
	\leqslant P[\exists h \in h's.t.|E_{in}(h)-E_{out}(h)|>\epsilon]  \\
	\leqslant 4m_{h'}(2N)\cdot e^{-\frac{1}{8}\epsilon ^{2}N}	\\
	\leqslant 4(2N)^{d_{VC}}\cdot e^{-\frac{1}{8}\epsilon ^{2}N}	\\	
$$

### Bounding Function

> 成长函数最多有多少种可能(maximum possible m_(H')(N) when break point = k).
$$
	B(N,k) \leqslant \sum_{i=0}^{k-1}C_{n}^{i}	(2)
$$

### VC-Dimension

> 最大的no-break point点（theformal name of maximum non-break point,the most inputs H' that can shatter.)
$$
	d_{VC} = 'k_{minimum}'-1
$$

> Vapnik-Cervonenkis(VC) bound: When N large enough, BAD BOUND for Grneral H‘.
$$
	P[\exists h \in h's.t.|E_{in}(h)-E_{out}(h)|>\epsilon ]\leqslant 4m_{h'}(2N)\cdot e^{-\frac{1}{8}\epsilon ^{2}N}
$$

> VC GOOD probability:set
$$
	\delta = 4(2N)^{d_{VC}}\cdot e^{-\frac{1}{8}\epsilon ^{2}N}	\\
      =>\frac{\delta }{4(2N)^{d_{VC}}}=e^{-\frac{1}{8}\epsilon ^{2}N}	\\
      =>ln(\frac{4(2N)^{d_{VC}}}{\delta})=\frac{1}{8}\epsilon ^{2}N	\\
      =>\epsilon =\sqrt{\frac{8}{N}ln(\frac{(2N)^{d_{VC}}}{\delta })}	\\	
$$
gen .error
$$
	|E_{in}(g)-E_{out}(g)|\leqslant \sqrt{\frac{8}{N}ln(\frac{(2N)^{d_{VC}}}{\delta })}	\\
      => E_{in}(g)-\sqrt{\frac{8}{N}ln(\frac{(2N)^{d_{VC}}}{\delta })}\leqslant E_{out}	\leqslant E_{in}(g)+\sqrt{\frac{8}{N}ln(\frac{(2N)^{d_{VC}}}{\delta })}	\\
$$
with a high probability.
$$
	E_{out}(g)\leqslant E_{in}(g)+\underset{\Omega (N,H'\delta )}{\sqrt{\frac{8}{N}ln(\frac{(2N)^{d_{VC}}}{\delta })}}	\\
$$
![avatar](https://gwfp.github.io/static/images/190118/VCmessage.png)

### VC Bound Rephrase

> 知道容忍度：
$$
	\epsilon 	
$$
,VC dimension：
$$
	d_{VC}
$$
,保持坏事件发生几率小于
$$
	\delta 	
$$
的情况下，求最少需要多少笔元素（N的值取多少）?	\\
$$
	4(2N)^{d_{VC}}e^{(-\frac{1}{8}\epsilon ^{2}N)}\leqslant \delta	\\ 
$$
theroy: need
$$
	N \approx 10000 d_{VC}
$$
in theory.practice:need
$$
	N \approx 10 d_{VC}
$$
in practice.

### 不同类型的成长函数、grow function、VCdimension

#### positive raysVC-Dimension

> 成长函数
$$
	m_{H'}(N) = N + 1
$$

> break point : 2	\\
  当N=1的时候，1+1=2^1,两边相等；当N=2的时候，2+1<2^2两边不等，所以break point就是2。

> 成长复杂度：O(N^{2-1})=O(N)

> VC dimension: 1 \\
$$
	d_{VC}=k-1=2-1=1
$$

#### positive intervals

> 成长函数
$$
	m_{H'}(N) = C_{n+1}^{2} + 1 	\\
		  = \frac{1}{2}N^{2}+\frac{1}{2}N+1
$$

> break point : 3

#### convex set

>成长函数
$$
	m_{H'}(N) = 2^{N}
$$

> break point ： no break point

#### 2D perceptrons

> 成长函数，in some cases(N<4)。
$$
	m_{H'}(N) < 2^{N}
$$ 

> break point : 4

## Error Measure

### Pointwise Error Measure

> in-sampple:
$$
	E_{in}(g)=\frac{1}{N}\sum_{n=1}^{N}err(g(x_{n}),f(x_{n})) 
$$

> out-of-sample:
$$
	E_{out}(g)=\underset{x~p}{\varepsilon}err(g(x),f(x))
$$

### Two important Pointwise Error Measures

> 0/1 error,often for Classication
$$
	err(\tilde{y},y)=\left |\tilde{y}-y  \right |
$$

> squared error,often for Regrassion
$$
	err(\tilde{y},y)=(\tilde{y}-y)^2
$$

> Linear Regression for Binary Classification
$$
	err_{0/1} \leqslant err_{sqr}	\\
	=> classification_E_{out}(W)\overset{VC}{\leqslant}classification_E_{in}(W)+\sqrt{...}	\\
	{\leqslant}regaression_E_{in}(W)+\sqrt{...} 
$$

## Weighted Classification


