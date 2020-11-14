import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST

# prepare dataset
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = MNIST(root="../../../data",
                      transform=transform,
                      train=True,
                      download=True)
# print(type(train_dataset))
# print(len(train_dataset))  # 60000
# img_tensor = train_dataset[0]  # a tuple (tensor,5)
# print(img_tensor)  # a tensor of the first img
# print(type(img_tensor[0]))  # a tensor of the first img
# print((img_tensor[0].shape))  # torch.Size([1, 28, 28])

train_dataloader = DataLoader(dataset=train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=6)
# print(train_dataloader.__sizeof__())  # 32
# print(len(train_dataloader))  # 938 = total(60000) / batch_size(64),that  is iteration

test_dataset = MNIST(root="../../../data",
                     transform=transform,
                     train=False,
                     download=True)
test_dataloader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=6)

# design module
in_channel = 1
out_channel = 10
kernel_size = 5
padding = 0


class CnnNet(nn.Module):
    def __init__(self):
        super(CnnNet, self).__init__()
        # design the cnn net
        self.conv1 = nn.Conv2d(in_channels=in_channel,
                               out_channels=10,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=padding)
        # print(self.conv1.weight.shape)  # torch.Size([10, 1, 5, 5])
        self.conv2 = nn.Conv2d(in_channels=10,
                               out_channels=20,
                               kernel_size=kernel_size,
                               stride=1,
                               padding=padding)
        # print(self.conv2.weight.shape)  # torch.Size([20, 10, 5, 5])
        self.pooling = nn.MaxPool2d(2)
        self.fc = nn.Linear(320, out_channel)

    def forward(self, x):
        # design fp
        # print("1:", x.size())  # torch.Size([64, 1, 28, 28])
        batch_size = x.size(0)
        x = self.pooling(F.relu(self.conv1(x)))
        # print("conv1:", x.size())  # torch.Size([64, 10, 12, 12])
        x = self.pooling(F.relu(self.conv2(x)))
        # print("conv2:", x.size())  # torch.Size([64, 20, 4, 4])
        x = x.view(batch_size, -1)  # flatten
        x = self.fc(x)
        # print("fc:", x.size())
        return x


module = CnnNet()

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimal = optim.SGD(module.parameters(), lr=0.01, momentum=0.5)


# train
def train(epoch):
    for i, (x, y) in enumerate(train_dataloader, 0):
        # print("iter:", i + 1)
        # FP
        y_pred = module(x)
        # get loss
        loss = criterion(y_pred, y)
        # BP
        optimal.zero_grad()
        loss.backward()
        # update
        optimal.step()
        if i % 300 == 299:
            print("epoch:%d\t,iter:%d\t,loss:%lf\t" % (epoch + 1, i + 1, loss.item()))


# test
def test(epoch):
    correct = 0
    total = 0
    for i, (x, y) in enumerate(test_dataloader, 0):
        y_pred = module(x)
        # print(type(y_pred)) # <class 'torch.Tensor'>
        # print("y_pred:", y_pred.size())  # y_pred: torch.Size([64, 10])
        _, predicted = torch.max(y_pred, dim=1)
        total += predicted.size(0)
        correct += (predicted == y).sum().item()
    print("Epoch:%d\t,Accuracy:%f" % (epoch + 1, correct / total))


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch)
        test(epoch)
