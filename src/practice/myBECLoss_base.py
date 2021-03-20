import torch
import torch.nn as nn

batch_size = 2
m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(batch_size, requires_grad=True)
target = torch.empty(batch_size,).random_(2)
print(input)
print(target)
output = loss(m(input), target)
print(output)
output.backward()
