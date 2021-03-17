import torch.nn as nn
import torch

encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
src = torch.rand(5, 3, 512)  # [seq_len,Batch_size,num_feature]  S B I
out = transformer_encoder(src)

print(out.shape)    #torch.Size([5, 3, 512])
