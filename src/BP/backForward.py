import torch

import matplotlib.pyplot as plt

train_set_x = [1, 2, 3]
train_set_y = [2, 4, 6]

w = torch.Tensor([1.0])
# compute grad
w.requires_grad = True


def forward(w, x):
    return w * x


def loss(x, y):
    y_pred = forward(w, x)
    return (y_pred - y) ** 2


for epoch in range(100):
    for x, y in zip(train_set_x, train_set_y):
        # l represents a tensor,means computation graph
        l = loss(x, y)
        l.backward()
        print('\tgrad:', x, y, w.grad.item())
        # w.grad is a tensor
        w.data = w.data - 0.01 * w.grad.data

        w.grad.data.zero_()
        # can use l.item() to get the loss value, a scare value
    print("process:", epoch, l.item())
