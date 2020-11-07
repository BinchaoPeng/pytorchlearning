import math

import numpy as np
import matplotlib.pyplot as plt

train_set_x = [1, 2, 3]
train_set_y = [2, 4, 6]


def forward(w, x):
    return np.dot(w, x)


def J(w, train_set_x, train_set_y):
    loss = np.sum(np.power(forward(w, train_set_x) - train_set_y, 2))

    return loss / len(train_set_y)


def gradient(w, train_set_x, train_set_y):
    grad = 1 / len(train_set_y) * np.sum((forward(w, train_set_x) - train_set_y) * train_set_x)
    return grad


def updateW(w, grad, rate):
    return w - rate * grad


if __name__ == '__main__':
    w = 1
    epochs = []
    loss = []
    for epoch in range(100):
        grad_val = gradient(w, train_set_x, train_set_y)
        cost_val = J(w, train_set_x, train_set_y)
        epochs.append(epoch)
        loss.append(cost_val)
        w = updateW(w, grad_val, 0.01)

        print("epoch:", str(epoch), "\tweight:" + str(w), "\tgrad:", str(grad_val), "\tloss:", str(cost_val))

    # polt
    plt.plot(epochs, loss)
    plt.ylabel("loss")
    plt.xlabel("epoch")

    plt.show()
