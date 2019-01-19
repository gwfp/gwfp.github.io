---
layout: post
category: 机器学习基础
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

### 成长函数  Growth Function
 
> 1.假设平面上所有的线都是一个hypothesis。（H' = {all lines in R^2}），此时有无限多条线，也就有无限多个hypothesis。对平面上N个点进行分类，将Ein=Eout的hypothesis进行合并形成dichotomy，上限为2^N.	\\
  2.使用mH'(N)替换公式（1）中的M \\
  3.mH'(N) 就是成长函数上限为 2^N
$$
	m_{H'}(N)=\underset{x_{1},x_{2}...x_(N)}{max}\left | H'(x_{1},x_{2}...x_(N)) \right |
$$



 

