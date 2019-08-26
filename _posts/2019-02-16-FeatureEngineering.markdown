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
	4. 特征

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

#### 异常值处理

### 缺失值数据处理

#### fillna 缺失值填充

##### 字符型数据 （info ：object）

##### 数值型数据 （info ：int64 / float64）

	# 0 值填充
	train[col].fillna(0, inplace=True)
	# 均值填充
	full["LotFrontage"].fillna(np.mean(full["LotFrontage"]),inplace=True)
	# 众书填充 
	full[col].fillna(full[col].mode()[0], inplace=True)

#### dropna 删除缺失值

## 数据预处理

### 选择子集

### 列名重命名

### 数据类型转换

#### 将列从 字符型 转为 数值型

	from sklearn.preprocessing import LabelEncoder
	
	lab = LabelEncoder()
	full["Alley"] = lab.fit_transform(full.Alley)

### 数据排序

## 数据标准化

### 标准化

### 极差法

### L1 与 L2 正则化

## 特征选择

### 特征工程

#### 特征提取

##### one-hot 编码 （get_dummies）

##### map 函数 (Series)

#### 特征选择

##### 相关系数法

## 构建模型

## 模型评估

## 方案实施

## 参考资料

[1] 一文带你探索性数据分析(EDA) [html](https://www.jianshu.com/p/9325c9f88ee6)
