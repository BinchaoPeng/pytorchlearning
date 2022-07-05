import torch
import torch.nn as nn

# input = torch.empty(3000, 4).random_(2)
#
# conv2d = nn.Conv2d(in_channels=1, out_channels=300, kernel_size=(40, 4))
#
# out = conv2d(input)
#
# print(input.shape)
# print(out.shape)

# With square kernels and equal stride
m = nn.Conv2d(16, 33, 3, stride=1)
# # non-square kernels and unequal stride and with padding
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2))
# # non-square kernels and unequal stride and with padding and dilation
# m = nn.Conv2d(16, 33, (3, 5), stride=(2, 1), padding=(4, 2), dilation=(3, 1))
input = torch.randn(20, 16, 50, 100)
output = m(input)
print(output.shape) # 20 33 48 98