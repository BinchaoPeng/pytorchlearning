import torch.nn as nn
import torch

torch.random.manual_seed(0)
input = torch.randn(2, 3, 3, 4)
# With Learnable Parameters
m = nn.BatchNorm2d(3)
# Without Learnable Parameters
# m = nn.BatchNorm1d(100, affine=False)
output = m(input)
print(input)
print(m.weight)
print(m.bias)
print(m.running_mean)
print(output)
##########################################################################################################
## wrong
##########################################################################################################
print('=' * 100)
input_c = (input[0][0] + input[1][0]) / 2
# input_c = input[0][0]
print(input_c)
firstDimenMean = torch.Tensor.mean(input_c)
firstDimenVar = torch.Tensor.var(input_c, False)  # false表示贝塞尔校正不会被使用
print(firstDimenMean)
print(firstDimenVar)
batchnormone = ((input_c[0][0] - firstDimenMean) / (torch.pow(firstDimenVar, 0.5) + m.eps)) \
               * m.weight[0] + m.bias[0]
print(batchnormone)
#######################################################################################################
## the computation is correct!!!
#######################################################################################################
print('=' * 100)
input_c1 = input[:, 0, :, :]
print(input_c1)
firstDimenMean = torch.Tensor.mean(input_c1)
firstDimenVar = torch.Tensor.var(input_c1, False)  # false表示贝塞尔校正不会被使用

print(firstDimenMean)
print(firstDimenVar)
batchnormone = ((input_c1 - firstDimenMean) / (torch.pow(firstDimenVar, 0.5) + m.eps)) \
               * m.weight[0] + m.bias[0]
print(batchnormone)

########################################################################################################
print('=' * 100)
print(input_c1.flatten())
print(torch.mean(input_c1.flatten()))
print(torch.var(input_c1.flatten()))
print(torch.mean(input_c1))
print(torch.var(input_c1))
