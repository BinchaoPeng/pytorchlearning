import torch
import torch.nn as nn
import torch.nn.functional as F

# prepare dataset
x_data = [[1.],
          [2.],
          [3.],
          [4.]]
y_data = [[0],
          [0],
          [1],
          [1]]

x_data = torch.as_tensor(x_data)
y_data = torch.as_tensor(y_data, dtype=torch.float)


# design model
class LinearModule(nn.Module):
    def __init__(self):
        super(LinearModule, self).__init__()
        # the dim of X and y
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = F.sigmoid(self.linear(x))
        print("data:", y_pred)
        return y_pred


model = LinearModule()

# construct loss and optimizer
criterion = nn.BCELoss(size_average=False)  # herit nn.module
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# train cycle
for epoch in range(1000):
    y_pred = model(x_data)
    loss = criterion(y_pred, y_data)
    print(epoch, loss.item())

    optimizer.zero_grad()
    loss.backward()
    # update
    optimizer.step()

print("w=", model.linear.weight.item())
print("b=", model.linear.bias.item())

# predict
x_test = torch.Tensor([1])
y_test = model(x_test)
print("y_pred=", y_test.data.item())
