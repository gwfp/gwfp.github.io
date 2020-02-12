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

>	1. 提出问题(了解场景和目标，了解评估准则)
	2. 理解数据
	3. 数据预处理（清洗，调权）
	4. 特征选择
	5. 模型构建（调参、状态分析、融合）
	6. 对测试数据集进行预测 

![avatar](https://gwfp.github.io/static/images/19/02/16/step.png){:width='500px' height="200px"}

使用sklearn进行虚线框内的工作

## 提出问题

## 理解数据

### 采集数据

### 导入数据

#### CSV

	import pandas as pd
	train = pd.read_csv("data.csv")

#### head

	train.head() ## 默认查看前5行

### 查看数据类型

少量
	
	train.dtypes

大量
	
	for col in df.columns:
    		print(col + ':' + str(df[col].dtype))

只看字符类型

	for col in df.columns:
    		if str(df[col].dtype) == 'object':
        		print(col)

### 查看数据类型及列数量

	train.info()

#### describe 描述统计信息

	train.describe()

#### isnull 统计空值的个数

	miss = train.isnull().sum()	# 统计空值的个数
	miss[miss>0].sort_values(ascending=False)	# 由低到高排列，accending=True 由高到低排列
	## 查看缺失值
	train.isnull().any()

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


#### Imputer 填充缺失值 (输出结果为 numpy.array)

	from sklearn.preprocessing import Imputer

	# 使用平均数 mean 填充，如需填充众数，将mean 换为 median
	imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
	data_imp = imp.fit_transform(data)

### 数据类型转换

#### 将列从 字符型 转为 数值型

	from sklearn.preprocessing import LabelEncoder
	
	lab = LabelEncoder()
	full["Alley"] = lab.fit_transform(full.Alley)

#### 将列从 数字型 转换为 字符型

	full["Alley"] = full["Alley"].astype(str)

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

sklearn

	from sklearn.preprocessing import StandardScaler	

	std = StandardScaler()
	train_scale = std.fit_transform(train)

pandas 只归一化 numerical 类型

	# 选择所有非 object 类型
	numeric_cols = df.columns[df.dtypes != 'object']
	# 归一化
	numeric_col_mean = df.loc[:, numeric_cols].mean()
	numeric_col_std = df.loc[:, numeric_cols].std()
	df.loc[:,numeric_cols] = (df.loc[:,numeric_cols] - numeric_col_mean) / numeric_col_std

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

pandas

	pd.get_dummies( , prefix=)

sklearn

	from sklearn.preprocessing import OneHotEncoder
	data_onehot = OneHotEncoder().fit_transform(data.target.reshape((-1,1)))

### 无量纲化

## 特征选择

### Filter : 过滤法

#### 方差选择法

> 先计算各个特征的方差，然后根据阈值，选择方差大于阈值的特征.默认情况下，删除零方差的特征，例如那些只有一个值的样本。例子，假设我们有一个有布尔特征的数据集，然后我们想去掉那些超过80%的样本都是0（或者1）的特征。布尔特征是伯努利随机变量，方差为 p(1-p)。

	from sklearn.feature_selection import VarianceThreshold

	X = [[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]]
	sel = VarianceThreshold(threshold=(.8*(1-.8)))
	sel.fit_transform(X)
	array([[0, 1],
       	      [1, 0],
       	      [0, 0],
      	      [1, 1],
   	      [1, 0],
   	      [1, 1]])

第一列里面0的比例为5/6,被去掉

#### 单变量特征选择（Univariate feature selection)
	
	from sklearn.feature_selection import SelectKBest, SelectPercentile

##### SelectBest 只保留K个最高的特征

	sklearn.feature_selection.SelectKBest(score_func=<function f_classif>, k=10）

##### SelectPercentile 只保留用户指定百分比的最高分的特征
	
	sklearn.feature_selection.SelectPercentile(score_func=<function f_classif>, percentile=10)

> score_func:
	For regression: f_regression, mutual_info_regression
	For classification: chi2, f_classif, mutual_info_classif

###### 卡方检验(chi2)

$$
	x^{2}=\sum \frac{(A-E)^2}{E}
$$

\\(A \\) 是实值，\\(E \\) 是理论值

###### 互信息和最大信息系数（mutual_info_classif,mutual information and maximal information）

互信息方法可以捕捉任何一种统计依赖，但是作为非参数方法，需要更多的样本进行准确的估计。

	from minepy import MINE
	from sklearn.feature_selection import SelectKBest
	
	#由于MINE的设计不是函数式的，定义mic方法将其为函数式的，返回一个二元组，二元组的第2项设置成固定的P值0.5 
	def mic(x, y):
    		m = MINE()
    		m.compute_score(x, y)
    		return (m.mic(), 0.5)
	#选择K个最好的特征，返回特征选择后的数据
	SelectKBest(lambda X, Y: array(map(lambda x:mic(x, Y), X.T)).T, k=2).fit_transform(iris.data, iris.target)

##### pearson相关系数（pearson correlaction）

$$
	p_{x,y}=\frac{cov(X,Y)}{\sigma(X)\sigma (Y)}
$$
	
\\(cov(X,Y) \\) 是X,Y的协方差，\\(\sigma(X) \sigma (Y) \\) 是X,Y个字标准差的乘积

适用于回归问题（y值连续）

	from scipy.stats import pearsonr

	pearsonr(x, y) # 输入为特征矩阵和目标向量
		       #输出为二元组(sorce, p-value)的数组


##### 距离相关系数

距离相关系数是为了克服Pearson相关系数的弱点而生的。即便Pearson相关系数是0，我们 也不能断定这两个变量是独立的(有可能是非线性相关);但如果距离相关系数是0，那么我们就可以说这两个变量 是独立的。

### Wrapper : 包装法

> 根据目标函数(通常是预测效果评分)，每次选择若干特征，或者排除若干特征。也可以将特征 子集的选择看作是一个搜索寻优问题，生成不同的组合，对组合进行评价，再与其他的组合进行比较。这样就将子集 的选择看作是一个是一个优化问题，这里有很多的优化算法可以解决，尤其是一些启发式的优化算法，如GA， PSO，DE，ABC等

#### 递归特征消除法

使用一个基模型来进行多轮训练，每轮训练后，消除若干权值系数的特征，再基于新的特征集进行下一轮训练.

	from sklearn.feature_selection import RFE
	from sklearn.linear_model import LogisticRegression	

	#参数n_features_to_select为选择的特征个数
	rfe = RFE(estimator=LogisticRegression(), n_features_to_select=2).fit_transform(iris.data, iris.target)
	ranking = rfe.ranking_
	# 查看特征排名，数值越大，说明重要性越高
	print (ranking) 

### Embedded : 嵌入法 

> 使用某些机器学习的算法和模型进行训练，得到各个特征的权值系数，根据系数从大到小选择特征。

#### 基于惩罚的特征选择法

	

#### 基于树模型的特征选择法

	from sklearn.feature_selection import SelectFromModel
	from sklearn.ensemble import GradientBoostingClassifier

	#GBDT作为基模型的特征选择
	SelectFromModel(GradientBoostingClassifier()).fit_transform(iris.data, iris.target)

### 降维

> PCA是为了让映射后的样本具有最 大的发散性;而LDA是为了让映射后的样本有最好的分类性能。

#### 主成分分许法（PCA）

	from sklearn.decomposition import PCA

	#主成分分析法，返回降维后的数据
	#参数n_components为主成分数目
	PCA(n_components=2).fit_transform(iris.data)

#### 线性判别分析法(LDA) 

	from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

	#线性判别分析法，返回降维后的数据
	#参数n_components为降维后的维数
	LDA(n_components=2).fit_transform(iris.data, iris.target)

## 构建模型

### 交叉验证 Cross Validation

#### Holdout 验证

#### K-fold cross-validation

#### 留一验证（Leave-One-Out Cross Validation, LOOCV）

### 集成学习([Ensemble Learning](https://gwfp.github.io/机器学习算法/2019/09/14/EnsembleLearning.html)) (模型融合)

![avatar](https://gwfp.github.io/static/images/19/09/14/EnsembleLearning.jpg){:width='350px' height="200px"}

#### Bagging

sklearn bagging [html](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html#sklearn.ensemble.BaggingClassifier)

	sklearn.ensemble.BaggingClassifier

#### boosting

#### Staking

#### Blending

## 模型评估



## 参考资料

[1] 一文带你探索性数据分析(EDA) [html](https://www.jianshu.com/p/9325c9f88ee6)
[2] feature selection [html](https://scikit-learn.org/stable/modules/feature_selection.html)
