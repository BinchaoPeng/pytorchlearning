import torch
import torch.nn as nn

batch_size = 3

m = nn.Sigmoid()
loss = nn.BCELoss(reduction="mean")
loss1 = nn.BCELoss(reduction="sum")

input = torch.randn(batch_size,3, requires_grad=True)
target = torch.empty(batch_size,3).random_(2)

i = 0
total_loss = 0
while i < batch_size:
    out = m(input[i])
    output = loss(out, target[i])
    output.backward()

    total_loss + output.item()
    # print(f"input[{i}]:", input[i])
    # print(f"target[{i}]:", target[i])
    print("output:", output)

    i += 1

i = 0
total_loss = 0
while i < batch_size:
    out = m(input[i])
    output1 = loss(out, target[i])
    output1.backward()
    total_loss + output.item()
    # print(f"input[{i}]:", input[i])
    # print(f"target[{i}]:", target[i])
    print("output1:", output1)
    i += 1