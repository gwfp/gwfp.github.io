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

## return of multiple invesetment

	tickers = ["PG", "MSFT", "F", "GE"]
	mydata = pd.DataFrame()
	for t in tickers:
    		mydata[t] = wb.DataReader(t, data_source="yahoo", start="2018-1-1")['Adj Close']

### 计算多种投资组合的收益率

#### 计算投资组合权重

$$
	w = w_{1} + w_{2}+...+w_{n}
$$

w : 整个投资组合在总投资中的权重
	
	weights = np.array([0.4,0.4,0.15,0.05])

$$
	p = w_{1}p_{1} + w_{2}p_{2} + ... +w_{n}p_{n}
$$

#### 计算投资组合年化收益率

	# 计算各组合成员年化收益率
	annual_return = simple_returns.mean()*250
	# 计算投资组合年化收益率
	pfolio = str(round(np.dot(annual_return, weights)*100,3))+ "%"	
	print(pfolio)

### 比较各投资组合的收益率

#### Normalization

$$
	p_{normal} = \frac{p_{n}}{p_{0}} * 100
$$

	p_normal = mydata/ mydata.iloc[0] * 100

#### 比较调整后的净收益收盘价

	p_normal.plot(figsize=(8,6))
	plt.show()
	
![avatar](https://gwfp.github.io/static/images/20/04/23/rateofreturn.png){:width='450px' height="350px"}


## 计算证券风险 Calculating the Risk of Security

	tickers = ["PG", "BEI.DE"]
	sec_data = pd.DataFrame()
	for t in tickers:
    		sec_data[t] = wb.DataReader(t, data_source="yahoo", start="2018-1-1")['Adj Close']

### 风险的标准差衡量(Statistical measures to quantify risk)

$$
	\sqrt{\frac{\sum (X-\bar{X})^{2}}{N-1}}
$$

	sec_return[["PG", "BEI.DE"]].std()*250**0.5
	result:
		PG        0.253991
		BEI.DE    0.203546
		dtype: float64


### 证券组合的相关性

#### 计算方差和协方差 Calculation Covariance and Correlation

协方差计算公式

$$
	 \rho_{xy}=\frac{(x-\bar{x})*(y-\bar{y})}{\sigma _{x}\sigma _{y}}
$$

	# 计算协方差
	sec_return.cov()*250
	# 计算收益率的相关性(不同于股票净值的相关性)
	sec_return.corr()

### 计算投资组合的风险 Calculation Protfolio Risk

	# PG 和 BEI.DE 各占50% 权重
	weights = np.array([0.5,0.5])

#### 计算两类风险投资组合的风险

$$
	\sigma_{p} = \sqrt{(w_{1}\sigma _{1}+w_{2}\sigma _{2})^{2}}        \\
        =\sqrt{w_{1}^2\sigma _{1}^{2}+2w_{1} \sigma _{1} w_{2} \sigma_{2} \rho_{1,2}+ w_{2}^2\sigma _{2}^{2}}
$$
	
	# 投资组合的风险
	pfolio_vol = (np.dot(weights.T,np.dot(sec_return.cov()*250, weights)))**0.5
	print(str(round(pfolio_vol ,5)*100)+'%')


#### 计投资组合的系统风险和非系统风险 Calculate Diversifiable and Non-Diversifiable Risk of a Protfolio

系统性风险

$$	
	r_{d}=\sigma_{p}^{2}-w_{1}^{2}*\sigma_{1}^{2}-w_{2}^{2}*\sigma_{2}^{2}+...+w_{n}^{2}*\sigma_{n}^{2}
$$

	# 计算各类投资的年度方差
	BEI_var_a = sec_return['BEI.DE'].var()*250
	PG_var_a = sec_return['PG'].var()*250
	# 计算总投资方差
	pfolio_var = (np.dot(weights.T,np.dot(sec_return.cov()*250, weights)))
	# 投资组合的系统性风险
	dr =  pfolio_var - (weights[0]**2*PG_var_a)-(weights[1]**2*BEI_var_a)
		


