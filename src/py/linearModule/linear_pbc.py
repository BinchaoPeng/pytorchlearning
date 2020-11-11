import numpy as np

import matplotlib.pyplot as plt

import math


def dataloader(train_set):
    train_X = train_set[:, 0:-1]
    train_y = train_set[:, -1]
    print("x: \n", train_X)
    print("y: \n", train_y)
    return train_X, train_y


def forward(w, x):
    return np.dot(w, x)


def cost(weight, train_X, train_y):
    loss = np.sum(math.pow(forward(weight, train_X) - train_y, 2))

    return loss / len(train_y)


if __name__ == '__main__':
    train_set = np.array([[1, 2], [2, 4], [3, 6]])
    train_X, train_y = dataloader(train_set)

    loss = []
    weight = []
    for w in np.arange(0.1, 2.1, 0.1):
        w = np.array(w)
        cost_temp = cost(w, train_X, train_y)
        print(w)
        print(cost_temp)
        loss.append(cost_temp)
        weight.append(w)

    print(loss)
    print(weight)
    # polt
    plt.plot(weight, loss)
    plt.ylabel("loss")
    plt.xlabel("w")
    plt.show()
