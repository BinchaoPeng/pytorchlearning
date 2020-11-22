import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


# step1  prepare dataset
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


# step2 design model using Class
class LogisticModel(nn.Module):
    def __init__(self):
        super(LogisticModel, self).__init__()
        # construct linear model
        # (8, 1) represents the dim of X and y
        self.linear1 = nn.Linear(64, 32)
        self.linear2 = nn.Linear(32, 16)
        self.linear3 = nn.Linear(16, 4)
        self.linear4 = nn.Linear(4, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.sigmoid(self.linear1(x))
        x = self.sigmoid(self.linear2(x))
        x = self.sigmoid(self.linear3(x))
        x = self.sigmoid(self.linear4(x))
        # the last x means your predicted value y_pred
        return x


model = LogisticModel()

# step3 construct loss function and optimizer
criterion = nn.BCELoss(size_average=True)  # inherit nn.module
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# step4 Training cycle [mini-batch]
if __name__ == '__main__':
    for epoch in range(100):
        """
        0 represents the start index
        data is a tuple (x,y),that is data = (X,y)
        inputs is X
        labels is y
        """
        for i, data in enumerate(train_loader, 0):
            # 1.prepare data
            inputs, labels = data
            # 2.forward
            y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            print(epoch, i, loss.item())
            # 3.backward
            optimizer.zero_grad()
            loss.backward()
            # 4.update
            optimizer.step()
