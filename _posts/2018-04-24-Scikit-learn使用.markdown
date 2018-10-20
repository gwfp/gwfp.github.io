---
layout: post
category: 机器学习基础
tags: [scikit-learn,函数库]
--- 

scikit-learn
==============

## 基础操作

### train_test_split

	from sklearn.model_selection import train_test_split
	...
	X_train, X_test, y_train, y_test = train_test_split(X, y, /
	test_size=0.5, random_state=666) #test_size默认0.2,seed:666

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
	

#### LogisticRegression

	

	
	

