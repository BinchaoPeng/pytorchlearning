import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import gzip
import csv
import matplotlib.pyplot as plt

datasetDir = '../../../data/nameDataset'
trainPath = datasetDir + '/names_train.csv.gz'
testPath = datasetDir + '/names_test.csv.gz'


# In[]:
# STEP1 prepare dataset
class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        super(NameDataset, self).__init__()
        filename = trainPath if is_train_set else testPath
        with gzip.open(filename, 'rt') as file:
            reader = csv.reader(file)  # get a iteration of csv file
            rows = list(reader)
            self.names = [row[0] for row in rows]  # save the str of name
            self.len = len(self.names)
            self.countries = [row[1] for row in rows]  # save the index in countryDict

            self.country_list = list(sorted(set(self.countries)))

            self.country_dict = self.getCountryDict()
            self.country_num = len(self.country_list)

    def __getitem__(self, item):
        return self.names[item], self.country_dict[self.countries[item]]

    def __len__(self):
        return self.len

    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict

    def get_num_country(self):
        return self.country_num

    def index2countryName(self, index):
        return self.country_dict[index]


# In[]: set paras when we need
BATCH_SIZE = 512
NUM_WORKS = 12
HIDDEN_SIZE = 100
N_LAYER = 2
N_CHARS = 128
USE_GPU = False
N_EPOCHS = 100
OUTPUT_SIZE = NameDataset().get_num_country()

# In[]:
# step1.1  load Dataset
"""
use ASCII to encode char 
the final length is the longest name, other name is used 0 to make full
use one-hot encoding every char
embedding
"""
trainSet = NameDataset()
trainLoader = DataLoader(dataset=trainSet, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKS)
testSet = NameDataset(is_train_set=False)
testLoader = DataLoader(dataset=testSet, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKS)


# In[]:
# STEP2 design RNN module
class RNNClassifierNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifierNet, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1

        """
        embedding Args:
        num_embeddings (int): size of the dictionary of embeddings
        embedding_dim (int): the size of each embedding vector
        """
        self.embedding = nn.Embedding(input_size, hidden_size)  # input char length and hidden_length
        """
        gru Args:
        input_size: The number of expected features in the input `x`
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two GRUs together to form a `stacked GRU`,
            with the second GRU taking in outputs of the first GRU and
            computing the final results. Default: 1
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False``
        bidirectional: If ``True``, becomes a bidirectional GRU. Default: ``False``
        """
        self.gru = nn.GRU(hidden_size,  # input
                          hidden_size,  # output
                          n_layers,
                          bidirectional=True)
        self.fc = nn.Linear(hidden_size * self.n_directions,  # if bi_direction, there are 2 hidden
                            output_size)

    def forward(self, input, sql_lengths):
        input = input.t()  # B x S -> S x B
        batch_size = input.size(1)

        hidden = self._init_hidden(batch_size)
        embedding = self.embedding(input)  # the input shape is S x B

        gru_input = pack_padded_sequence(embedding, sql_lengths)

        output, hidden = self.gru(gru_input, hidden)

        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]

        fc_output = self.fc(hidden_cat)

        return fc_output

    def _init_hidden(self, batch_size):
        return torch.zeros(self.n_layers * self.n_directions,
                           batch_size,
                           self.hidden_size)


# STEP4 train and test
def name2list(name):
    """
    name to ASCII by ord()
    :param name:
    :return:
    """
    arr = [ord(c) for c in name]
    return arr, len(arr)


def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor


def make_tensors(names, countries):
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [sl[0] for sl in sequences_and_lengths]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    countries = countries.long()

    # make tensor of name : B x S
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        seq_tensor[idx, : seq_len] = torch.LongTensor(seq)

    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    countries = countries[perm_idx]

    return create_tensor(seq_tensor), \
           create_tensor(seq_lengths), \
           create_tensor(countries)


def time_since(start):
    s = time.time() - start
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def trainModel(epoch):
    total_loss = 0

    for i, (names, countries) in enumerate(trainLoader, 1):
        # print(countries)
        # print(type(countries))
        inputs, seq_lengths, target = make_tensors(names, countries)
        y_pred = module(inputs, seq_lengths)
        loss = criterion(y_pred, target)

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        total_loss += loss.item

        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch {epoch} ', end='')
            print(f'[{i * len(inputs)}/{len(trainSet)}]', end='')
            print(f'loss = {total_loss / (i * len(inputs))}')
    return total_loss


def testModel():
    correct = 0
    total = len(testSet)
    with torch.no_grad():
        for i, (names, countries) in enumerate(testLoader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = module(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()
        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')

    return correct / total


if __name__ == '__main__':
    module = RNNClassifierNet(N_CHARS, HIDDEN_SIZE, OUTPUT_SIZE, N_LAYER)

    if USE_GPU:
        device = torch.device("cuda:0")
        module.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(module.parameters(), lr=0.001)

    start = time.time()
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        trainModel(epoch)
        acc = testModel()
        acc_list.append(acc)

    # polt
    x = range(1, N_EPOCHS + 1)
    plt.plot(x, acc_list)
    plt.ylabel("acc")
    plt.xlabel("epoch")
    plt.show()
