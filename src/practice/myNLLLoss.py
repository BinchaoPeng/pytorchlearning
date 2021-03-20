import torch
import torch.nn as nn

batch_size = 4

m = nn.LogSoftmax(dim=1)
criterion = nn.NLLLoss()
y_pred = torch.randn(batch_size, 5)
print(y_pred)
y = torch.empty(batch_size, dtype=torch.long).random_(5)
y_pred = m(y_pred)
loss = criterion(y_pred, y)

print(y_pred)
print(y)
print(loss)
"""
tensor([[-0.0823, -1.9247, -0.0603,  0.1532, -1.3969],
        [ 0.4716,  1.2842, -0.9967,  1.2505, -0.5792],
        [ 0.1119,  1.1941, -0.9118,  2.0806, -0.1346],
        [ 0.2026,  0.6298, -0.2217,  0.7229, -0.1681]])
tensor([[-1.3123, -3.1547, -1.2903, -1.0768, -2.6269],
        [-1.7939, -0.9813, -3.2621, -1.0150, -2.8447],
        [-2.5058, -1.4236, -3.5295, -0.5371, -2.7523],
        [-1.7156, -1.2884, -2.1399, -1.1953, -2.0863]])
tensor([3, 1, 1, 3])
tensor(1.1692)
"""