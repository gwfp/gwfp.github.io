---
layout: post
category: 机器学习基础
tags: [scikit-learn,　函数库]
--- 

scikit-learn
==============

##　基础操作

### train_test_split

	from sklearn.model_selection import train_test_split
	...
	X_train, X_test, y_train, y_test = train_test_split(X, y, /
	test_size=0.5, random_state=666) #test_size默认0.2,seed:666

##　算法实现

###　KNN算法

	训练集: X_train , 结果集:　y_train, 投票个数:k, 输入:Ｘ
	#初始化
	kNN_classifier = KNeighborsClassifier(n_neighbors=3)
	#数据拟合
	kNN_classifier.fit(group, labels)
	#预测
	predict_y = kNN_classifier.predict(X)
	

