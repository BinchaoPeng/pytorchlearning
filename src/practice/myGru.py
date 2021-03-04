import torch
import torch.nn as nn

batch_size = 3
seq_lens = 4
input_size = 2
hidden_size = 5
num_layers = 1
bidirectional = True

n_direction = 2 if bidirectional else 1

gru = nn.GRU(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, bidirectional=bidirectional)

input = torch.randn(seq_lens, batch_size, input_size)
hidden = torch.zeros(num_layers * n_direction, batch_size, hidden_size)

out, hn = gru(input, hidden)

if n_direction == 2:
    hidden_cat = torch.cat([hn[-1], hn[-2]], dim=1)
else:
    hidden_cat = hidden[-1]

print(gru.parameters())

print("input:", input)
print("input shape:", input.shape)

print("hidden:", hidden)
print("hidden shape:", hidden.shape)

print("out:", out)
print("out shape:", out.shape)
print("hn:", hn)
print("hn shape:", hn.shape)
print("hidden_cat:", hidden_cat)
print("hidden_cat shape:", hidden_cat.shape)
