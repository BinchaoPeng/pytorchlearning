[TOC]

# RNN实现hello--ohlol



## 准备数据



### 构造字典

![image-20201121101248068](images/image-20201121101248068.png)

有多少个不同的词，词向量就有多少长，每个词向量只有一个1，其余是0，称为独热向量。

在这里是一个词有4个不同的字母，故向量长4.



独热向量的每行就是x1，x2，x3，x4，x5，输入维度input_size就是4



![image-20201121101417274](images/image-20201121101417274.png)



## 设计模型



### 结构

【接交叉熵，产生一个分布，分类问题】

是为了判断输出属于helo四个字母的那个字母

![image-20201121101601849](images/image-20201121101601849.png)

![image-20201121101740911](images/image-20201121101740911.png)

输入x为独热向量，结合h，输入RNN Cell

再接入softmax得到各字母的预测概率

再通过y的独热向量，计算NLLLoss

得到最后loss



![image-20201121101836634](images/image-20201121101836634.png)

CrossEntropyLoss：交叉熵损失



## 实现

### 参数

![image-20201121101913980](images/image-20201121101913980.png)

### 数据



![image-20201121101930113](images/image-20201121101930113.png)

索引转独热向量

![image-20201121102136017](images/image-20201121102136017.png)



![image-20201121102147910](images/image-20201121102147910.png)

-1：表示自动处理

![image-20201121102157924](images/image-20201121102157924.png)



### model

![image-20201121102226027](images/image-20201121102226027.png)

![image-20201121102315476](images/image-20201121102315476.png)

![image-20201121102356477](images/image-20201121102356477.png)



![image-20201121102428152](images/image-20201121102428152.png)

![image-20201121102450859](images/image-20201121102450859.png)

batch_size只有在构造h0时才需要



### loss、optimizer

![image-20201121102615131](images/image-20201121102615131.png)

交叉熵



### train

![image-20201121102905051](images/image-20201121102905051.png)



**注意loss：**是个计算图，所以不用`.item()`

![image-20201121102823743](images/image-20201121102823743.png)

维度：

![image-20201121102942934](images/image-20201121102942934.png)

![image-20201121103000036](images/image-20201121103000036.png)

预测：

![image-20201121103055068](images/image-20201121103055068.png)

四维的h，找最大的那个字母



### result

![image-20201121103128045](images/image-20201121103128045.png)



## RNN训练方式

![image-20201121103201601](images/image-20201121103201601.png)





## model优化

### RNN方式

![image-20201121103412399](images/image-20201121103412399.png)



![image-20201121103451386](images/image-20201121103451386.png)

![image-20201121103704348](images/image-20201121103704348.png)

两维好处：使用交叉熵时变成了矩阵

输出：

![image-20201121103750541](images/image-20201121103750541.png)



### 结果

![image-20201121103829694](images/image-20201121103829694.png)



## 独热向量（one-hot）

缺点：

- 纬度高，多少种就是多少维。【high-dimension】
- 向量过于稀疏。【sparse】
- 硬编码，不是学习出来的。【hardcoded】



## 嵌入层（embedding）

EMBEDDING：把一个高维的稀疏的样本映射到低维的稠密的空间里【即降维】

![image-20201121104604244](images/image-20201121104604244.png)

特点：

- low-dimension
- dense
- learned from data

### 示例【4 --》 5】

![image-20201121104953991](images/image-20201121104953991.png)

​			输入索引2：转置、使用矩阵乘法索引

![image-20201121105050093](images/image-20201121105050093.png)

4行5列做转置，然后与【0 0 1 0】的转置相乘，得到索引2的那行



### 用法

![image-20201121105242609](images/image-20201121105242609.png)

线性层是为了保证输出的隐层（h）的维数和预测的分类数一致

x必须是长整形张量

### 参数

![image-20201121105401909](images/image-20201121105401909.png)

num_embeddings:input的独热向量维数（embedding的高度）

embedding_dim:embedding的宽度

![image-20201231144822432](images/image-20201231144822432.png)





### 代码

![image-20201121105607375](images/image-20201121105607375.png)

注意点：

![image-20201121105623127](images/image-20201121105623127.png)

![image-20201121105643089](images/image-20201121105643089.png)

![image-20201121105705985](images/image-20201121105705985.png)

![image-20201121105731737](images/image-20201121105731737.png)





![image-20201121105833361](images/image-20201121105833361.png)

![image-20201121105849761](images/image-20201121105849761.png)

![image-20201121105857437](images/image-20201121105857437.png)

![image-20201121105909984](images/image-20201121105909984.png)



## LSTM

![image-20201121110357986](images/image-20201121110357986.png)



![image-20201121110444072](images/image-20201121110444072.png)



## GRU

RNN 和 LSTM 的折中，LSTM时间复杂度高，学习能力强。

![image-20201121110541493](images/image-20201121110541493.png)

![image-20201121110611870](images/image-20201121110611870.png)

