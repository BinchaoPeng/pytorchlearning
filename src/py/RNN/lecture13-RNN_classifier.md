[TOC]



# 循环网络分类器：名字分类（名字对应国家）



## model

![image-20201121195040487](images/image-20201121195040487.png)



![image-20201121194907460](images/image-20201121194907460.png)



## 主要循环

![image-20201121195451858](images/image-20201121195451858.png)

N_CHARS:字符数量。英文字符转成独热向量

HIDDEN_SIZE:隐层维度，GRU输出的h的维度

N_COUNTRY:国家数目，即分类器数目

N_LAYER:GRU层数



start：训练时间

![image-20201121200202313](images/image-20201121200202313.png)



## 准备数据

字符集，使用ASCII码

![image-20201121200403727](images/image-20201121200403727.png)

用0填充长度，每个数字都是一个向量，构成一个张量

![image-20201121200719838](images/image-20201121200719838.png)

y：

![image-20201121200919197](images/image-20201121200919197.png)

### 数据集构造

![image-20201121200940192](images/image-20201121200940192.png)

getitem：name字符串，country是索引



![image-20201121201639945](images/image-20201121201639945.png)

### 数据加载

![image-20201121201805488](images/image-20201121201805488.png)



## model design

![image-20201121201843103](images/image-20201121201843103.png)

![image-20201121202920148](images/image-20201121202920148.png)

**embedding过程：**

![image-20201121203207596](images/image-20201121203207596.png)

input_size:

hidden_size:

output_size:

n_layers:

bidirectional:双向还是单项循环网络

![image-20201121202103335](images/image-20201121202103335.png)



![image-20201121202125657](images/image-20201121202125657.png)



pack up：要求排序

![image-20201121203755960](images/image-20201121203755960.png)







**双向循环网路**

正向和反向得到的隐层做拼接

![image-20201121202535455](images/image-20201121202535455.png)



![image-20201121202614353](images/image-20201121202614353.png)

![image-20201121202622113](images/image-20201121202622113.png)





## name to tensor

字符、ASCII码数字、填充、转置、排序

![image-20201121204421431](images/image-20201121204421431.png)

![image-20201121204120863](images/image-20201121204120863.png)



## train

![image-20201121204446461](images/image-20201121204446461.png)



## test

![image-20201121204602671](images/image-20201121204602671.png)



## 作业

![image-20201121204841859](images/image-20201121204841859.png)





# 总结

解决的是带有序列关系的数据