import torch
import torch.nn as nn

batch_size = 3

criterion = nn.CrossEntropyLoss()

y_pred = torch.randn(batch_size, 5)
y = torch.empty(batch_size, dtype=torch.long).random_(5)
print(y_pred)

soft = nn.Softmax(dim=1)
y_pred = soft(y_pred)
print(y_pred)

print(y)

loss = criterion(y_pred, y)
print(loss)

"""
tensor([[-0.9051, -0.4489, -0.5590, -0.1007,  1.1984],
        [-0.0395,  0.4527,  0.0423, -0.1724,  0.4925],
        [ 0.4064,  1.5080,  1.1995,  0.1651,  0.2300]])
tensor([[0.0693, 0.1094, 0.0980, 0.1550, 0.5682],
        [0.1588, 0.2597, 0.1723, 0.1390, 0.2703],
        [0.1275, 0.3836, 0.2818, 0.1002, 0.1069]])
tensor([2, 1, 3])
tensor(1.6658)
"""