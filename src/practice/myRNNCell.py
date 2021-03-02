import torch
import torch.nn as nn
import torch.nn.functional as F

input_size = 2  # input_size means the lengths of one-hot encode vector, for example, the code [... 128 dim ...] of 'o' in "hello"
batch_size = 1
seq_len = 10    # it means the length of the whole sequence  rather than one-hot encode vector

hidden_size = 5

# the vector dimension of input and output for every sample x
Cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)

data = torch.randn(seq_len, batch_size, input_size)     # (3,2,4)
print(data)
hidden = torch.zeros(batch_size, hidden_size)   # (2,4)
print(hidden)

for idx, input in enumerate(data):
    print("=" * 20, idx, "=" * 20)
    print("input shape:", input.shape)
    print(input)

    print(hidden)

    hidden = Cell(input, hidden)

    print("hidden shape:", hidden.shape)
    print(hidden)
