import torch
import torch.nn as nn

num_embeddings = 7 # 至少要是7
embedding_dim = 4

# 其实是[2,5],[5,1],[3,4],[1,6]
# and every single number is a one-hot vector,its length is num_embeddings
# finally input becomes a 3-dim vector, the third-dim length is embedding_dim
# out shape : (2,4,4)
input = torch.LongTensor([[2, 5, 3, 1], [5, 1, 4, 6]])
embedding = nn.Embedding(num_embeddings=num_embeddings, embedding_dim=embedding_dim)
out = embedding(input)

print("input:", input)
print("embedding parms:", embedding.weight)
print("out:", out)
