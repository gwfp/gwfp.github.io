---
layout: post
category: 金融
tags: [金融分析]
---

Finance Analysis
====================


## rate of return

	import numpy as np
	import pandas as pd
	import pandas_datareader.data as wb
	import matplotlib.pyplot as plt

	PG = wb.DataReader('PG', data_source="yahoo", start="2018-1-1")

### Simple rate of return/HPR,holding-perid return 

> 同时处理多个资产时，推荐使用

$$
	r = \frac{p_{t}-p_{0}-d_{t}}{p_{0}}
$$

	PG["simple_return"] = (PG["Adj Close"]/PG["Adj Close"].shift(1))-1
	# 计算简单收益的年化平均收益
	avg_returns = PG["simple_return"].mean()*250
	print(round(avg_returns,5) ,'%')

### Logarithmic rate iof return

> 处理单个长期资产时，推荐使用

$$
	r =  log \frac{p_{t}}{p_{0}}
$$

	PG["log_return"] = np.log(PG["Adj Close"]/PG["Adj Close"].shift(1))
	# 计算对数收益率的年华平均收益
	avg_log_returns = PG["log_return"].mean()*250
	print(str(round(avg_log_returns,5)), '%')


