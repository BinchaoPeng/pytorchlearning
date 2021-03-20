import torch
import torch.nn as nn

epoch = 4
batch_size = 3

m = nn.Sigmoid()
loss = nn.BCELoss(reduction="mean")
loss1 = nn.BCELoss(reduction="sum")

input = torch.randn(epoch, batch_size, requires_grad=True)
target = torch.empty(epoch, batch_size, ).random_(2)

print(input)
print(target)

"""
the way of mean
"""
i = 0
total_loss = 0
while i < epoch:
    out = m(input[i])
    output = loss(out, target[i])
    output.backward()

    total_loss += output.item()
    # print(f"input[{i}]:", input[i])
    # print(f"target[{i}]:", target[i])
    print("output:", output)
    i += 1
print("total_loss:", total_loss)
print("loss:", total_loss / epoch)
"""
the way of sum
"""
i = 0
total_loss = 0
while i < epoch:
    out = m(input[i])
    output1 = loss1(out, target[i])
    output1.backward()
    total_loss += output1.item()
    # print(f"input[{i}]:", input[i])
    # print(f"target[{i}]:", target[i])
    print("output1:", output1)
    i += 1
print("total_loss:", total_loss)
print("loss:", total_loss / (epoch * batch_size))
