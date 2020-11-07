import torch
import torch.nn as nn

# prepare dataset
x_data = [[1.],
          [2.],
          [3.],
          [4.]]
y_data = [[2.],
          [4.],
          [6.],
          [8.]]

x_data = torch.as_tensor(x_data)
y_data = torch.as_tensor(y_data)


# design model
class LinearModule(nn.Module):
    def __init__(self):
        super(LinearModule, self).__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


model = LinearModule()

# construct loss and optimizer
criterion = nn.MSELoss(size_average=False)
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
x_test = torch.Tensor([4])
y_test = model(x_test)
print("y_pred=", y_test.data.item())
