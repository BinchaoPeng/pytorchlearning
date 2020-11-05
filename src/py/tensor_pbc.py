from __future__ import print_function
import torch

"""
Tensor
"""

x = torch.empty(5, 3)
print(x)

x = torch.zeros(5, 3, dtype=torch.long)
print(x)

x = torch.rand(5, 3)
print(x)

x = torch.tensor([[1, 2], [3, 4]])
print(x)

# overload attribute
y = x.new_ones(5, 3, dtype=torch.double)
print(x)
print(y)

print(x.size())

