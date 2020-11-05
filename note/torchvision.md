[TOC]

# torchvision

是pytorch用于深度学习的计算机视觉包



## 数据	ETL

提取（extract）、转换（transform）、加载（load）



```python
import torch
import torchvision
import torchvision.transforms as transforms

train_set = torchvision.datasets.FashionMNIST(
    root=r'../../data/FashionMNIST',
    train=True,
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor()]
    )
)

# trainSet is loaded in train loader
train_loader = torch.utils.data.DataLoad(train_set)
```



# class:Dataset,DataLoader



