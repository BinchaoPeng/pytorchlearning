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
train_loader = torch.utils.data.DataLoader(train_set, batch_size=10)

import numpy as np
import matplotlib.pyplot as plt

torch.set_printoptions(linewidth=120)


