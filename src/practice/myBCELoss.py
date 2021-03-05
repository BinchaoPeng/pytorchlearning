import torch
import torch.nn as nn

m = nn.Sigmoid()
loss = nn.BCELoss()
input = torch.randn(3, requires_grad=True)
target = torch.empty(3).random_(2)
out = m(input)
output = loss(out, target)
output.backward()

print("input:", input)
print("target:", target)
print("out:", out)
