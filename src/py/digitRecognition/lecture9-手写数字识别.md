[TOC]

@[TOC]



# 手写数字识别



## 通道channel

![image-20201111171117154](images/image-20201111171117154.png)

单通道

三通道：rgb（红绿蓝）



## PIL-->pytorch 变换

WxHxC   ===> CxWxH(pytorch中)

![image-20201111171229799](images/image-20201111171229799.png)



## transform

![image-20201111171728856](images/image-20201111171728856.png)

![image-20201111171915004](images/image-20201111171915004.png)

先转为tensor，再归一化，两个参数是均值和标准差

## 模型

![image-20201111172021865](images/image-20201111172021865.png)

把每个图像转成向量，得到矩阵N x 1\*28\*28

![image-20201111172545551](images/image-20201111172545551.png)

## 代码实现

### prepare dataset

![image-20201111174727930](images/image-20201111174727930.png)



![image-20201111172848613](images/image-20201111172848613.png)



### loss  optimal

![image-20201111200913770](images/image-20201111200913770.png)

momentum 冲量

### train

![image-20201111172929189](images/image-20201111172929189.png)



### test

![image-20201111173116343](images/image-20201111173116343.png)

不做梯度计算

![image-20201111173342999](images/image-20201111173342999.png)

最大值下标，行是维度0，列是维度1

注意   `-，`

![image-20201111173634203](images/image-20201111173634203.png)

### run

![image-20201111173652638](images/image-20201111173652638.png)

一轮训练一轮测试



### 结果

![image-20201111173757432](images/image-20201111173757432.png)



## 总结

全连接

局部信息利用不好

使用的是原始特征

可以自动提取特征，比如CNN



## 练习

![image-20201111174127912](images/image-20201111174127912.png)



