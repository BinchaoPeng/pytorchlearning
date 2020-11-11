import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# prepare dataset
xy = np.loadtxt(r"data/digits.csv.gz", delimiter=",", dtype=np.float32)
# get x_data except the last collum which represents y
x_data = xy[:, :-1]
# use [-1] to make y_data to be a matrix, not a vector
y_data = xy[:, [-1]]

print(np.shape(x_data))
print(np.shape(y_data))

x_data = torch.as_tensor(x_data)
y_data = torch.as_tensor(y_data)


# design model
class MutiDimLogisticModule(nn.Module):
    def __init__(self):
        super(MutiDimLogisticModule, self).__init__()
        # construct linear model
        # (8, 1) represents the dim of X and y
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 4)
        self.linear4 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        # the last x means your predicted value y_pred
        return x


model = MutiDimLogisticModule()

# construct loss function and optimizer
criterion = nn.BCELoss(size_average=True)  # herit nn.module
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# train cycle
for epoch in range(1000):
    # forward
    y_pred = model(x_data) # here using all data, not using mini-batch
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    # backward
    optimizer.zero_grad()
    loss.backward()
    # update
    optimizer.step()

print("w=", model.linear1.weight)
print("w=", model.linear2.weight)
print("w=", model.linear3.weight)
print("w=", model.linear4.weight)
print("b=", model.linear1.bias)
print("b=", model.linear2.bias)
print("b=", model.linear3.bias)
print("b=", model.linear4.bias)
# # predict
# x_test = torch.Tensor([4])
# y_test = model(x_test)
# print("y_pred=", y_test.data.item())
