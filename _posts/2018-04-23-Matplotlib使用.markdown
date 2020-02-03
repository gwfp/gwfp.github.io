---
layout: post
category: 机器学习工具
tags: [mathplotlib,函数库]
--- 

mathplotlib
==========

##　载入

> import matplotlib as mpl
  import matplotlib.pyplot as plt #部分载入

## 绘制窗口

### plt.show()

	plt.plot(np.random.rand(100))
	plt.show()

### matplotlib inline 

	%matplotlib inline ## 开头添加一次即可
	x = np.random.randn(100)
	y = np.random.randn(100)
	plt.scatter(x,y)

### matplotlib notebook

	%matplotlib notebook ## 生成动态图，可编辑
	s = pd.Series(np.random.randn(100))
	s.plot(style='k--o', figsize=(10,5))

### matplotlib qt5 （新建窗口，生成动态图）

	%matplotlib qt5
	df = pd.DataFrame(np.random.rand(50,2),columns=['A','B'])
	df.hist(figsize=(12,5),color='g',alpha=0.8)

	plt.close() #关闭窗口
	plt.gcf.clear() #清空窗口内容

## 绘图

### x =np.linspace(0, 10 ,20)

### siny=np.sin(x)

### cosy=np.cos(x)

### 绘制折线图

> plt.plot(x, siny)
  plt.show()

### 同时绘制两条折线图

> plt.plot(x, siny)
  plt.plot(x, cosy)
  plt.show()

### 对绘制曲线选择颜色

> plt.plot(x, cosy, color="red")

### 对绘制曲线选择线条样式

> plt.plot(x, cosy, color="red", linestyle="--")

### 对ｘ轴取值范围调节

> plt.xlim(-5, -15)

### 对ｘ, y 同时调节

> plt.axis([-1,11,-2,2]) # x轴取值(-1~11),ｙ轴取值(-2,2)

### ｘ, y 轴上添加字符串

> plt.xlabel("x axis")
  plt.ylabel("y value")
  
### 为绘制的折线加上图示

> plt.plot(x, siny, label="sin(x)")
  plt.plot(x, cosy, label="cos(x)")
  plt.legend()	
  plt.show()

### 为绘制的折线加上标题

> plt.title("this is title")

### 绘制散点图

> plt.scatter(x,siny)
  plt.show()

### 改变散点图的不透明度

> plt.scatter(x, siny, alpha=0.5)
