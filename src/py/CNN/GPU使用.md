[TOC]



### 使用GPU训练模型

~~~python
# 定义设备，下标0表示使用第0个显卡
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# 将模型置于设备中，参数、缓存、权重等所有都放入指定device中
model.to(device)
~~~



将输入放入device中

![image-20201114102837101](images/image-20201114102837101.png)



![image-20201114102938892](images/image-20201114102938892.png)



