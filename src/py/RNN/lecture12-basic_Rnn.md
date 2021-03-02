[TOC]



# 基础RNN

## 总述

dense/deep 	稠密网络  ====  FC  全连接



 用来处理带有序列模式的数据、先后时间序列、数据共享概念、权重共享机制



==序列、numLayers==

![image-20201121110808393](images/image-20201121110808393.png)



## 下雨例子

![image-20201230204153385](images/image-20201230204153385.png)

x1，2，3表示三天数据，每个x包含温度、气压、是否有雨三个特征，拼接后作为输入数据



## RNN Cell

RNN Cell 本质：线性层，维度映射，改变维度。区别就是



![image-20201115154833799](images/image-20201115154833799.png)

RNN Cell就是一个线性层

==把3维度映射为5维度==

红色箭头的输入就是上一个输出的h，h被称为隐层

h0表示先验，可以初始化为0，或者使用CNN+FC转为将图像转为多维

这些绿色的格子（RNN Cell），其实是一个格子。即右边图示是左边图示的解释。

下面图是代码实现：使用一个循环解决，不同的是x和h的值。



![image-20201121092111177](images/image-20201121092111177.png)





### RNN Cell内部结构图示



![image-20201115155411482](images/image-20201115155411482.png)

input_size：输入X的维度

hidden_size: 隐层维度

![image-20201230205628744](images/image-20201230205628744.png)



### 过程

自定义RNN Cell，需要参数：输入维度，隐层维度

![image-20201115155724292](images/image-20201115155724292.png)

维度要求

![image-20201115155912752](images/image-20201115155912752.png)

​			经过x~1~和h~0~得到隐层h~1~

​	

各变量具体含义：	

![image-20201121094942550](images/image-20201121094942550.png)

![image-20201121095040353](images/image-20201121095040353.png)

### demo

![image-20201121095432901](images/image-20201121095432901.png)

参数、初始化

设置序列数据

隐层全0，batch_size和hidden_size

循环遍历dataset

hidden=上一级的input和hidden

![image-20201121095634311](images/image-20201121095634311.png)



## RNN

把多个RNN Cell连接循环算出隐层，并且可能有多层（横向）的就是RNN。

![image-20201115161112800](images/image-20201115161112800.png)

### 维度要求

![image-20201115161210443](images/image-20201115161210443.png)

numLayer：初始隐层的层数。这里是3层。一种颜色表示一个线性层

![image-20201121100135403](images/image-20201121100135403.png)



下对角是input

![image-20201121100433601](images/image-20201121100433601.png)



**主要是多了numLayers**，xxSize是表示相应维度

![image-20201121100234013](images/image-20201121100234013.png)



### demo

![image-20201121100528289](images/image-20201121100528289.png)

![image-20201121100713652](images/image-20201121100713652.png)

![image-20201121100722937](images/image-20201121100722937.png)

设置参数

初始化RNN

设置input和hidden

直接调用cell，不用写循环



![image-20201121100742416](images/image-20201121100742416.png)

### 其他参数

batch_first：

![image-20201115161736100](images/image-20201115161736100.png) 

![image-20201121100847707](images/image-20201121100847707.png)