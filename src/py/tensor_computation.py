from __future__ import print_function
import torch

x = torch.ones(5, 3, dtype=torch.float)
y = torch.ones(5, 3, dtype=torch.float)
print(x)
print(y)
print(x + y)
print(torch.add(x, y))

result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

y.add_(x)
print(y)

print(x[(1, 2), :])

x = torch.randn(4, 4)
y = x.view(4 * 4)
z = x.view(-1, 2)
print(x)
print(y)
print(z)
print(x)

x = torch.randn(1, dtype=torch.double)
print(x)
print(x.item())
