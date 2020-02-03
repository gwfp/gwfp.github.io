---
layout: post
category: 机器学习工具
tags: [matplotlib,函数库]
--- 

matplotlib
==========


## 载入

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

## 图表的基本元素

### 作图大小

	fig = df.plot(figsize=(6,4)) 

### 添加名称

#### tittle添加

	plt.title('aa')

#### x轴、y轴名称

	plt.xlabel('x')
	plt.ylabel('y')

### 添加图例

	plt.legend(loc=0)

	'''
	loc 对应数值： 
	'best',	-- 0 (自适应方式)
	'upper right',	-- 1 
	'upper left', -- 2
	'lower left', -- 3
	'lower right', -- 4
	'right', 	-- 5
	'center left', 	-- 6
	'center right', -- 7
	'lower center', -- 8
	'upper center', -- 9
	'center'-- 10

### 控制x、y轴边界

	plt.xlim([0,12])
	plt.ylim([0,1.5])

### 设置信x、y轴刻度

#### 设置刻度

	plt.xticks(range(10))
	plt.yticks([0,0.2,0.4,0.6,0.8,1.0,1.2])

#### 设置刻度显示方式

	# 浮点数显示刻度
	fig.set_xticklabels("%1f" %i for i in range(10))
	fig.set_yticklabels("%2f" %i for i in [0,0.2,0.4,0.6,0.8,1.0,1.2])

### 显示设置

#### 添加网格

	plt.grid()

#### 不显示坐标轴

	plt.axis('off')

#### x、y轴单独不可见

	frame = plt.gca()
	frame.axes.get_xaxis().set_visible(False)
	frame.axes.get_yaxis().set_visible(False)

### 图标样式

#### 线型 (linestyle)

	'-'       solid line style
	'--'      dashed line style
	'-.'      dash-dot line style
	':'       dotted line style

#### 点型（marker）

	'.'       point marker
	','       pixel marker
	'o'       circle marker
	'v'       triangle_down marker
	'^'       triangle_up marker
	'<'       triangle_left marker
	'>'       triangle_right marker
	'1'       tri_down marker
	'2'       tri_up marker
	'3'       tri_left marker
	'4'       tri_right marker
	's'       square marker
	'p'       pentagon marker
	'*'       star marker
	'h'       hexagon1 marker
	'H'       hexagon2 marker
	'+'       plus marker
	'x'       x marker
	'D'       diamond marker
	'd'       thin_diamond marker
	'|'       vline marker
	'_'       hline marker

#### 颜色 （color）

	plt.hist(np.random.randn(100), color='red', alpha=0.8)
	'''
	 alpha: 0-1 透明度
	 color: red-r,green-g, black-k, blue-b, yellow-u
	'''
#### 颜色渐变 （colormap）
	
	df=pd.DataFrame(np.random.randn(50,4), columns=list('ABCD'))
	df = df.cumsum()
	df.plot(style='--', alpha=0.8, colormap='afmhot')  # afmhot_r 颜色反向

#### 样式（style）

	import matplotlib.style as psl
	print(psl.available)
	psl.use('fast') #一旦样式选择，除非重启，所有图表都将有样式
	ts=pd.Series(np.random.randn(1000).cumsum(),index=pd.date_range('1/1/2000',periods=1000))
	ts.plot(style='--g', grid=True, figsize=(10,6))
	 '''
         style = ['seaborn-dark', 'seaborn-darkgrid', 'seaborn-ticks', 'fivethirtyeight', 'seaborn-whitegrid', 'classic', '_classic_test', 'fast', 'seaborn-talk', 'seaborn-dark-palette', 'seaborn-bright', 'seaborn-pastel', 'grayscale', 'seaborn-notebook', 'ggplot', 'seaborn-colorblind', 'seaborn-muted', 'seaborn', 'Solarize_Light2', 'seaborn-paper', 'bmh', 'tableau-colorblind10', 'seaborn-white', 'dark_background', 'seaborn-poster', 'seaborn-deep']
	'''

### 刻度

#### 设置刻度 与 刻度格式

	from matplotlib.ticker import MultipleLocator, FormatStrFormatter

	ax = plt.subplot(111)
	plt.plot(np.random.rand(100))

	ax.xaxis.set_major_locator(MultipleLocator(40))# 将 x 主刻度标签设置为40的倍数
	ax.xaxis.set_major_formatter(FormatStrFormatter('%.0f')) # 设置刻度格式
	ax.xaxis.set_minor_locator(MultipleLocator(20))

	ax.xaxis.grid(True, which='both') # 'major' 主刻度显示网格
	ax.yaxis.grid(True, which='minor')

#### 设置注释

	plt.text(40,0.6,'text',fontsize=15)

#### 保存图片

	plt.savefig('./pic.png',
          	   dpi=400,
           	   bbox_inched='tight' , # 不保留边沿部分
           	   )

## 生成子图

### figure

	fig1 =plt.figure(num=1, figsize=(4,2)) # 创建子图1
	plt.plot(np.random.rand(100).cumsum(),'k--')  
	fig2 =plt.figure(num=2, figsize=(4,2)) # 创建子图2
	plt.plot(np.random.rand(50).cumsum(),'k--') # plot会自动向上寻找最近的figure

### 常用子图的生成方式

#### 先建figure后再在figure里画图

	fig = plt.figure(figsize=(10,6),facecolor='grey')
	ax1 = fig.add_subplot(2,2,1) # (2,2,1) ->  2*2 图像中第一行左图
	plt.plot(np.random.rand(50).cumsum(),'k--')
	plt.plot(np.random.rand(50).cumsum(),'b--')

	ax2 = fig.add_subplot(2,2,2) # 第一行右图
	plt.hist(np.random.rand(50), alpha=0.5)

	ax4 = fig.add_subplot(2,2,4) # 第二行右图
	df = pd.DataFrame(np.random.rand(10,4), columns=['a','b','c','d'])
	ax4.plot(df, alpha=0.8, linestyle='--',marker='.')


#### 先创建新figure，并返回一个subpolt对象的numpy数组，在生成的图里画内容

代码：

	fig,axes =plt.subplots(2,3,figsize=(10,4)) # 生成2行3列的图像
	ts = pd.Series(np.random.randn(1000).cumsum())

	ax1 =axes[0,1] #在1行2列上画图
	ax1.plot(ts)

	axes[1,2].plot(np.random.randn(100)) #在2行3列上画图

带参数代码：

	fig,axes =plt.subplots(2,3,sharex=True,sharey=True) # sharex 共享x轴
	for i in range(2):
    	    for j in range(3):
            	axes[i,j].hist(np.random.randn(500), color='k')
	plt.subplots_adjust(wspace=1,hspace=1) 调整间距

#### 分别绘制多图

	df = pd.DataFrame(np.random.randn(1000,4),columns=list('ABCD'))
	df.plot(style='--',subplots=True)  # subplots = True 分别显示

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


Seaborn
==============


