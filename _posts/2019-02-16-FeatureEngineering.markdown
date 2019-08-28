---
layout: post
category: 机器学习概念
tags: [特征工程]
---


特征工程  FeatureEngineering
===============

## 概念

> 最大限度地从原始数据中提取特征，以供机器学习算法和模型使用, 数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。

## 机器学习的步骤:

>	1. 提出问题
	2. 理解数据
	3. 数据清洗
	4. 数据预处理
	5. 特征选择
	6. 模型构建
	7. 对测试数据集进行预测 

![avatar](https://gwfp.github.io/static/images/19/02/16/step.png){:width='500px' height="200px"}

使用sklearn进行虚线框内的工作

## 提出问题

## 理解数据

### 采集数据

### 导入数据

#### CSV

	import pandas as pd
	
	train = pd.read_csv("data.csv")

### 查看数据

#### head

	train.head() ## 默认查看前5行

#### describe 描述统计信息

	train.describe()

#### isnull 统计空值的个数

	miss = train.isnull().sum()	# 统计空值的个数
	miss[miss>0].sort_values(ascending=False)	# 由高到低排列，accending=True 由高到低排列
	
#### info 发现缺失数据

	train.info()

#### 探索性数据分析 (EDA，Exploratory Data Analysis）

##### pandas_profiling
	
	import pandas_profiling as ppf

	ppf.ProfileReport(train)

#### 查看异常值
	
##### 通过boxplot 查看异常值

	plt.figure(figsize=(10,8))
	sns.boxplot(train.YearBuilt, train.SalePrice)
	
## 数据清洗 (Data cleaning method)

### 缺失值数据处理

#### 删除多列

	train.drop("SalePrice",axis=1,inplace=True)##删除SalePrice列

#### 检查缺失数据

#### fillna 缺失值填充

##### 字符型数据 （info ：object）

	# None 值填充
	train[col].fillna("None",inplace=True)

##### 数值型数据 （info ：int64 / float64）

	# 0 值填充
	train[col].fillna(0, inplace=True)
	# 均值填充
	full["LotFrontage"].fillna(np.mean(full["LotFrontage"]),inplace=True)
	# 众数填充 
	full[col].fillna(full[col].mode()[0], inplace=True)

### 数据类型转换

#### 将列从 字符型 转为 数值型

	from sklearn.preprocessing import LabelEncoder
	
	lab = LabelEncoder()
	full["Alley"] = lab.fit_transform(full.Alley)

#### 多项式转换

	from sklearn.preprocessing import PolynomialFeatures

	ploy = PolynomialFeatures()
	ploy.fit_transform(iris.data)

#### 自定义函数（对数）

	from numpy import log1p
	from sklearn.preprocessing import FunctionTransformer	

	log = FunctionTransformer(log1p).fit_transform(data.data)

#### 转换时间戳（从字符串到日期时间格式）

### 数据标准化/归一化

#### 标准化(z-score standaedization)

$$
	z=\frac{x-\mu }{\sigma }
$$

\\(\mu \\) 是均值，\\(\sigma \\) 是标准差

	from sklearn.preprocessing import StandardScaler	

	std = StandardScaler()
	train_scale = std.fit_transform(train)

#### min-max 归一化

$$
	m=\frac{x-x_{min}}{x_{max}-x_{min}}
$$

	from sklearn.preprocessing import MinMaxScaler

	min_max =  MinMaxScaler()
	train_scale = min_max.fit_transform(train)

#### 范数归一化

对行向量进行处理

	from sklearn.preprocessing import Normalizer
	import pandas as pd

	normalizer = Normalizer()
	train_normal = pd.DataFrame(normalizer.fit_transform(train))

### 二值化

> 设置阈值，大于阈值设置为1，小于阈值设置为0
$$
	x'=\left\{\begin{matrix}
	   1, x>threshold \\  0, x\leqslant threshold
	   \end{matrix}\right.
$$

	from sklearn.preprocessing import Binarizer

	# 二值化，阈值设置为3
	binarizer = Binarizer(threshold=3).fit_transform(train_scale)	

### 哑编码

#### 独热编码(OneHotEncoder)

	from sklearn.preprocessing import OneHotEncoder

	data_onehot = OneHotEncoder().fit_transform(data.target.reshape((-1,1)))

### 无量纲化

## 特征选择

### 特征工程

#### 特征提取

##### map 函数 (Series)

#### 特征选择

##### 相关系数法

## 构建模型

## 模型评估

## 方案实施

## 参考资料

[1] 一文带你探索性数据分析(EDA) [html](https://www.jianshu.com/p/9325c9f88ee6)
