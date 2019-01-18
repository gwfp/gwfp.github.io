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
	      \leqslant  2e^{-2\epsilon ^{2}N}+2e^{-2\epsilon ^{2}N}+...+2e^{-2\epsilon ^{2}N}	\\
	      = 2Me^{-2\epsilon ^{2}N} 
$$


