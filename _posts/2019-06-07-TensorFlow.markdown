---
layout: post
category: 深度学习框架
tags: [深度学深度学习框架,Tensorflow2.0]
---

TensorFlow2.0
===============

## 安装

### Linux, Mac OSX, Windows

	(sudo -H) pip install tensorflow==2.0.0-beta1

### Check installation and version

	import tensorflow as tf
	tf.__version__

### 升级

	pip install --upgrade tensorflow

## Tensorflow Workflow

![avatar](https://gwfp.github.io/static/images/19/06/07/tensorflow_workflow.png){:width='600px' height="200px"}

## Data Ingestion and Transformation

## Modle Building

## Training

### Eager Execution（动态图）模式

#### 作用

> 它是一个命令式、由运行定义的接口，一旦从 Python 被调用，其操作立即被执行。
  无需 Session.run() 就可以把它们的值返回到 Python。

#### 用法

> TF2.0默认为动态图，即eager模式

## Saving

### SavedModel

## 参考资料

[1] [Tensorflow 源码](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python)
[2] [Tensorflow Tutorials](https://tensorflow.google.cn/beta)
[3] [keras](https://keras.io)
	


