---
layout: post
category: 机器学习基础
tags: [回归分析,简单线性回归,算法]
---

LinearRegression
===========

## 基本形式
$$
	f(x)=\omega ^{T}x+b	(0)
$$

## 简单线性回归 Simple Linear Regression

> y has to be explained/forecasted on the basis of one single independent variable, x.(y and x both are interval variables)

### 简单线性回归函数推导

> 假设我们找到了最佳拟合直线方程 
$$
    y_{i} = ax_{i} + b, (1.0)
$$

> 对于每个样本点
$$
  (x^{i})     (1.1)
$$

>，线性试图回归学得
$$
        \hat{y}_{i} = a x_{i} +b 使得 \hat{y}_{i} \cong y_{i}  (1.2)	
$$

> 使用“最小二乘法” 使得均方差最小化
$$
	\sum_{i=1}^{m}(y_{i} - \hat{y}_{i})^2   (1.3)
$$
> 尽可能小

>  将（1.2）带入 （1.3）
$$
	J(a,b) = \sum_{i=1}^{m}(y_{i} - ax_{i}+b)^2  (1.4)
$$
> 即找到a,b使得(1.4)最小

> 分别对a和b进行求导

> 对b求导
$$
	\frac{\delta J(a,b)}{\delta b} = 0	(1.5)	\\
	=> \sum_{i=1}^{m}2(y_{(i)}-ax_{(i)}-b)(-1) = 0  \\
	=> \sum_{i=1}^{m}(y_{(i)}-ax_{(i)}-b) = 0        \\
	=> \sum_{i=1}^{m}(y_{(i)})-a\sum_{i=1}^{m}(x_{(i)})-\sum_{i=1}^{m}b = 0 \\	=> mb = \sum_{i=1}^{m}(y_{(i)})-a\sum_{i=1}^{m}(x_{(i)})  \\
	=> b = \bar{y}-a\bar{x}  （1.6）
$$

> 对a求导
$$
	\frac{\delta J(a,b)}{\delta a} = 0	(1.7)   \\
	=>  \sum_{i=1}^{m}2(y_{(i)}-ax_{(i)}-b)(-x_{(i)}) = 0	\\
	=> \sum_{i=1}^{m}(y_{(i)}-ax_{(i)}-b)x_{(i)} = 0	 （1.8）\\
	=> 将（1.6） 带入 （1.8）   \\
	=> \sum_{i=1}^{m}(y_{(i)}-ax_{(i)}-\bar{y}-a\bar{x})x_{(i)} = 0  \\
	=> a = \frac{\sum_{i=1}^{m}(x_{i}y_{(i)}-x_{(i)}\bar{y})}{\sum_{i=1}^{m}((x_{(i)})^2-\bar{x}x_{(i)})} 	(1.9.1) \\
	=> \sum_{i=1}^{m}x_{(i)}\bar{y} = \bar{y}\sum_{i=1}^{m}x_{(i)}=m\bar{y}\bar{x}= \bar{x}\sum_{i=1}^{m}y_{(i)}=\sum_{i=1}^{m}y_{(i)}\bar{x} \\
	=> m\bar{y}\bar{x}=\sum_{i=1}^{m}\bar{x}\bar{y} \\
	=> a = \frac{\sum_{i=1}^{m}(x_{i}y_{(i)}-x_{(i)}\bar{y}-\bar{x}y_{(i)}+\bar{x}\bar{y})}{\sum_{i=1}^{m}((x_{(i)})^2-\bar{x}x_{(i)}-\bar{x}x_{(i)}+\bar{x}^2)}  \\
	=> a = \frac{\sum_{i=1}^{m}(x_{i}-\bar{x})(y_{(i)}-\bar{y})}{\sum_{i=1}^{m}(x_{(i)}-\bar{x})^2} 	(1.9.2) 
$$

### 简单线性回归实现

	import numpy as np
	
	from metrics import r2_score
	
	class SimpleLinearRegression:
	
		def __init__(self):
			'''初始化Simple Linear Regression 模型'''
			self.a_ = None
			self.b_ = None

		def fit(self, x_train, y_train):
			assert x_train.ndim ==1, \
				"Simple Linear Regressor can only solve single feature taining data."
			assert len(x_train) == len(y_train), \
				"the size of x train must be equal to the size of y train"
	
			x_mean = np.mean(x_train)
			y_mean = np.mean(y_train)

			#向量积的方法, 带了性能的提升
			num = (x_train - x_mean).dot(y_train - y_mean)
			d = (x_train - x_mean).dot(x_train - x_mean)
		
			self.a_ = num / d
			self.b_ = y_mean - self.a_ * x_mean

			return self

		def predict(self, x_predict):
			'''给定待预测数据集x_predict, 返回表示x_predict的结果向量'''
			assert x_predict.ndim == 1,  \
				"Simple Linear Regressior can only solve single feature training data."
			assert self.a_ is not None and self.b_ is not None, \
				"must fit before predict!"
		
			return np.array([self._predict(x) for x in x_predict])

		def _predict(self, x_single):
			'''给定单个待预测数据x_single, 返回x的预测结果值'''
			return self.a_ * x_single + self.b_
		
		def score(self, x_test, y_test):
                	y_predict = self.predict(x_test)
                	return r2_score(y_test, y_predict)

		def __repr__(self):
			return "SimpleLinearRegression()"

	
	def main():
		x = np.array([1., 2., 3., 4., 5.])
		y = np.array([1., 3., 2., 3., 5.])
	
		reg1 = SimpleLinearRegression()
		reg1.fit(x, y)

		print("a=",reg1.a_,",b=",reg1.b_)
		x_predict = 6	
		print(reg1.predict(np.array([x_predict])))

	if __name__ == '__main__':
		main()


## 多元线性回归

> 存在
$$
	\hat{y} = X_{b}\cdot \theta 
$$

> 使得
$$
        \sum_{i=1}^{m}(y_{i} - \hat{y}_{i})^2   (2.1)
$$
> 尽可能小

> 多元线性回归方程的正规方程解 （Mormal Equation）
$$
\theta = (X_{b}^{T}X_{b})^{-1}X^{T}y
$$

> 问题：时间复杂度高O（n^3）,优化后O（n^2.4）

> 优点：不需要对数据做归一化处理

### 多元线性回归实现

	import numpy as np
	from  metrics import r2_score

	class LinearRegression_normal:

	class LinearRegression_normal:
	        def __init__(self):
        	        '''初始化linear Regression模型'''
	       	        self.coef_ = None
	                self.interception_ = None
	                self._theta = None

	        def fit_nomal(self, X_train, y_train):
	                '''根据训练数据集X_train, y_train，使用标准化方程训练模型'''
	                assert X_train.shape[0] == y_train.shape[0], \
	                        "the size of X_train must be equal to the size of y_train"
	                #X矩阵前加入全1的列
	                X_b = np.hstack([np.ones((len(X_train), 1)), X_train])
	                self._theta = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y_train)
	                self.interception_ = self._theta[0]
	                self.coef_ = self._theta[1:]

	                return self

		def predict(self, X_predict):
	                assert self.interception_ is not None and self.coef_ is not None, \
	                  "must fit before predict!"
	                assert X_predict.shape[1] == len(self.coef_), \
	                  "the feature number of X_predict must be equal to X_train"

	                X_b = np.hstack([np.ones((len(X_predict), 1)), X_predict])
	                return X_b.dot(self._theta)

	        def score(self, X_test, y_test):
	                '''根据测试集X_test 和 y_test 确定当前模型的准确度'''

	                y_predict = self.predict(X_test)
	                return r2_score(y_test, y_predict)

	        def __repr__(self):
	                return "Linear Regression()"


## 回归函数的准确度

### 均方误差 MSE （Mean Squared Error）
$$
	MSE = \frac{1}{m}\sum_{i=1}^{m}(y_{test}^{(i)}-\hat{y}_{test}^{(i)})^{2}
$$

### 均方根误差 RMSE （Root Mean Squared Error）
$$
	RMSE = \sqrt{MSE} = \sqrt{\frac{1}{m}\sum_{i=1}^{m}(y_{test}^{(i)}-\hat{y}_{test}^{(i)})^{2}}
$$

### 平均绝对误差 MAE （Mean Absolute Error）
$$
	MAE = \frac{1}{m}\sum_{i=1}^{m}\left | y_{test}^{(i)}-\hat{y}_{test}^{(i)} \right |
$$

### R Squared
$$
	R^2 = 1-\frac{SS_{residual}}{SS_{total}} \\
	= 1-\frac{\sum_{i=1}^{m}(\hat{y}_{(i)}-y_{(i)})^2}{\sum_{i=1}^{m}(\bar{y}-y_{(i)})^2} \\
	= 1-\frac{\sum_{i=1}^{m}(\hat{y}_{(i)}-y_{(i)})^2/m}{\sum_{i=1}^{m}(\bar{y}-y_{(i)})^2/m} \\
	= 1-\frac{MSE(\hat{y}, y)}{Var(y)} 
$$

1. R^2 <= 1。R^2 越大越好, 当预测的模型完全拟合数据时,R^2得到最大值1。
2. R^2 = 0 当模型的差错等于基准模型的差错时R^2为0。
3. R^2 < 0 很有可能我们的数据不存在任何线性关系。

### 准确度计算的实现

	def mean_squared_error(y_true, y_predict):
        	'''计算 MSE'''
        	assert len(y_true) == len(y_predict), \
	          "the size of y_true must be equal to the size of y_predict"

	        return np.sum((y_true - y_predict) ** 2) / len(y_true)

	def root_mean_squared_error(y_true, y_predict):
	        '''计算 RMSE'''
	        return sqrt(mean_squared_error(y_true, y_predict))

	def mean_absolute_error(y_true, y_predict):
	        assert len(y_true) == len(y_predict), \
	          "the size of y_true must be equal to the size of y_predict"

	        return np.sum(np.absolute(y_true - y_predict)) / len(y_true)

	def r2_score(y_true, y_predict):
	        '''计算y_true和y_predict之间的R Square'''

	        return 1 - mean_squared_error(y_true, y_predict)/np.var(y_true)

## 线性回归对数据的可解释性

