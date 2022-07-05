import torch
import torch.nn as nn

B = 10

input = torch.empty(B, 4, 3000).random_(2)

conv2d = nn.Conv1d(in_channels=4, out_channels=300, kernel_size=(4, 40))

out = conv2d(input)

print(input.shape)
print(out.shape)
