---
layout: post
category: 机器学习工具
tags: [scikit-learn,函数库]
--- 

scikit-learn
==============

## scikit-learn algorithm cheat sheet [html](http://scikit-learn.org/stable/tutorial/machine_learning_map/)

![avatar](https://gwfp.github.io/static/images/18/04/24/ml_map.png){:width='220px' height="130px"}

## 基础操作

### train_test_split

	from sklearn.model_selection import train_test_split
	...
	X_train, X_test, y_train, y_test = train_test_split(X, y, /
	test_size=0.5, random_state=666) #test_size默认0.2,seed:666

### 数据归一化StandardScaler

> 把所有数据映射到0-1之间
$$
	X_{scale} = \frac{x-x_{min}}{x_{max}-x_{min}}
$$

	from sklearn.preprocessing import StandardScaler
	
	'''数据归一化'''
        standardScaler = StandardScaler()
        standardScaler.fit(X_train)
        X_train = standardScaler.transform(X_train)
        X_test_standard = standardScaler.transform(X_test)

### 数据标准化 

> 经过处理的数据符合标准正态分布,即均值为1,方差为0.\\(\mu \\)为样本数据均值，\\( \sigma \\)为样本数据标准差.

$$
	x^{*} = \frac{x-\mu }{\sigma }
$$

### 降维

#### PCA

> 用PCA (Principal Component Analysis) 方法，对数据进行降维

	from  sklearn.decomposition import PCA

	# 保留90%的维度信息
	pca = PCA(0.9)
	pca.fit(X)
	X_reduction = pca.transform(X)

### 准确度

	from sklearn.metrics import r2_score
	r2_score(y_test, y_predict)	

## 算法实现

### KNN算法

	训练集: X_train , 结果集:　y_train, 投票个数:k, 输入:Ｘ
	#初始化
	kNN_classifier = KNeighborsClassifier(n_neighbors=3)
	#数据拟合
	kNN_classifier.fit(group, labels)
	#预测
	predict_y = kNN_classifier.predict(X)
	#准确度
	kNN_classifier.score(X_test, predict_y)
	
### 超参数 和 模型参数

> 超参数： 在算法运行前需要决定的参数

> 模型参数: 算法过程中学习的参数

#### KNN 的超参数 hyperparameter	

	best_socre = 0.0
        best_k = -1
        for k in range(1, 11):
                knn_clf = KNeighborsClassifier(n_neighbors=3)
                knn_clf.fit(X_train, y_train)
                score = knn_clf.score(X_test, y_test)
                if score > best_socre:
                        best_k = k
                        best_score = score

#### KNN 的网格搜索

	from sklearn.model_selection import GridSearchCV

	param_grid = [
                {
                        'weights': ['uniform'],
                        'n_neighbors': [i for i in range(1, 11)],
                },
                {
                        'weights': ['distance'],
                        'n_neighbors': [i for i in range(1, 11)],
                        'p': [i for i in range(1, 6)]
                }
        ]

	knn_clf = KNeighborsClassifier()
        grid_search = GridSearchCV(knn_clf, param_grid, n_jobs=-1， verbose=2)#n_jobs多核运算 , verbose 现实运行过程
        grid_search.fit(X_train, y_train)
	
	#KNN最佳所有参数
        print(grid_search.best_estimator_)
	#最佳得分
        print(grid_search.best_score_)
        #最佳参数值
        print(grid_search.best_params_)

	#导入最佳预测参数
        knn_clf = grid_search.best_estimator_

	knn_clf.prdict(X_test)
        knn_clf.score(X_test, y_test)
	

#### PCA降维后使用KNN
	
	# -*- coding: utf-8 -*-

	import numpy as np
	from sklearn import datasets
	import matplotlib
	matplotlib.use('TkAgg')
	import matplotlib.pyplot as plt
	from sklearn.model_selection import train_test_split
	from sklearn.decomposition import PCA
	from sklearn.neighbors import KNeighborsClassifier


	def main():
		digits = datasets.load_digits()
		X = digits.data
		y = digits.target
		X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=666)

		'''
		pca 对数据降维 并保持 95% 的维度信息
		'''
		pca = PCA(0.95)
		pca.fit(X_train)
		# 计算保留95%维度信息时最多降维到多少维度
		print(pca.n_components_)
		X_train_reduction = pca.transform(X_train)
		X_test_reduction = pca.transform(X_test)

		# 解释方差相应的比例，表示数据能表示百分之多少的数据维持的方差
		print(pca.explained_variance_ratio_)
    
		knn_clf = KNeighborsClassifier()
		knn_clf.fit(X_train_reduction, y_train)

		score = knn_clf.score(X_test_reduction, y_test)
		print(score)

		'''
		将数据降到2维进行可视化
		'''
		pca_2 = PCA(n_components=2)
		pca_2.fit(X)
		X_reduction =pca_2.transform(X)

		for i in range(10):
			plt.scatter(X_reduction[y==i,0], X_reduction[y==i,1], alpha=0.8)
		plt.show()


	if __name__ == '__main__':
		main()



### LogisticRegression

	
### SVM


## 参考资料

[1] 伊恩古德费洛.深度学习[M].北京:人民邮电出版社.2017:34-51
[2] sciklearn document [html](http://scikit-learn.github.io/stable)
	

