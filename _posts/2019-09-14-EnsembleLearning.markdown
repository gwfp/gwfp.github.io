---
layout: post
category: 机器学习算法
tags: [集成学习,Bagging,Boosting,Stacking,Blending,AdaBoost,GBDT,XGBoost,LightGBM,CatBoost]
---


集成学习  Ensemble Learning
===============

## 集成学习的概念

> 将几种机器学习技术组合成一个预测模型的元算法，以达到减小方差（bagging）、偏差（boosting）或改进预测（stacking）的效果。

![avatar](https://gwfp.github.io/static/images/19/09/14/EnsembleLearning.jpg){:width='350px' height="200px"}

## 基础集成学习

### 最大投票

### 平均

### 加权平均法

## 高级集成学习

### Bagging (bootstrap aggregating/自助采样法)

#### 随机森林(RF, Random Forest)

### Boosting

> 可以将弱学习器提升为强学习器的算法。
  步骤：
	
	1. 先从初识训练集训练出一个基学习器
	2. 根据基学习器的表现对训练样本分布进行调整，使得先前基学习器做错的样本在后续受到更多关注。
	3. 然后基于调整后的样本分许训练下一个基学习器。
	4. 如此重复进行，直到基学习器的数目达到事先制定的值 T ，最终将这 T 个基学习器进行加权结合。

#### AdaBoosting 
	
> 基于“加性模型”，即基学习器的线形组合

#### GBDT

#### XGBoost

#### LightGBM

#### CatBoost

### 堆叠 (Stacking)

### 混合 (Blending)




## 参考资料

[1] 周志华.机器学习[M].清华大学出版社.2017:171-195 
