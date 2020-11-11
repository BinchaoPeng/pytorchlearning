import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import DataLoader


class DiabetesDataset(Dataset):
    def __init__(self, filepath):
        """
        load data;two ways:
        1: all data
        2: batch
        """
        # prepare dataset
        xy = np.loadtxt(filepath, delimiter=",", dtype=np.float32)
        # get x_data except the last collum which represents y
        self.x_data = torch.as_tensor(xy[:, :-1])
        # use [-1] to make y_data to be a matrix, not a vector
        self.y_data = torch.as_tensor(xy[:, [-1]])
        # example num
        self.len = xy.shape[0]

    def __getitem__(self, index):
        """
        a implement of data index
        :param index:
        :return:
        """
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        """
        a implement of get length of dataset
        :return:
        """
        return self.len


dataset = DiabetesDataset(r"data/digits.csv.gz")

train_loader = DataLoader(dataset=dataset,
                          batch_size=32,
                          shuffle=True,
                          num_workers=2)

if __name__ == '__main__':
    for epoch in range(100):
        for i, data in enumerate(train_loader, 0):
            pass
