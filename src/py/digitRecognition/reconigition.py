import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# for constructing DataLoader
from torchvision import transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from torchvision.datasets import MNIST

# step1 prepare dataset
batch_size = 64
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# MNIST inherit Dataset,just use directly
train_dataset = MNIST(root="../../../data/",
                      train=True,
                      transform=transform,
                      download=True)
test_dataset = MNIST(root="../../../data/",
                     train=False,
                     transform=transform,
                     download=True)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=6)
test_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=6)

print(type(train_dataset))
print(type(train_loader))
print(len(train_dataset))

# can get the number of train_dataset
print(len(train_loader))


# step2 design model


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear1 = nn.Linear(784, 512)
        self.linear2 = nn.Linear(512, 256)
        self.linear3 = nn.Linear(256, 128)
        self.linear4 = nn.Linear(128, 64)
        self.linear5 = nn.Linear(64, 32)
        self.linear6 = nn.Linear(32, 16)
        self.linear7 = nn.Linear(16, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1 * 28 * 28)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        x = self.relu(x)
        x = self.linear3(x)
        x = self.relu(x)
        x = self.linear4(x)
        x = self.relu(x)
        x = self.linear5(x)
        x = self.relu(x)
        x = self.linear6(x)
        x = self.relu(x)
        x = self.linear7(x)
        # x = F.softmax(x) // auto softmax in CE , not need.
        print(x)
        return x


model = Net()
# step3 construct loss and optimal
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# step4 train
def train(epoch, train_loader):
    running_loss = 0
    iteration = 0
    for i, (x, y) in enumerate(train_loader, 0):
        print(y, y.dtype)
        # forward===========>to get predict y
        y_pred = model.forward(x)
        # compute loss======================>use y_pred and y to get loss by criterion,that is loss function
        loss = criterion(y_pred, y)
        if i % 300 == 299:
            print("epoch:", epoch + 1, "index:", i + 1, "loss:", loss.item())

        # backward=======================>set grad to 0,then loss does backward
        optimizer.zero_grad()
        loss.backward()
        # update==============================>update
        optimizer.step()
        running_loss += loss.item()
        iteration = i + 1
    print("EPOCH:", epoch + 1, "iteration:", iteration, "loss:", running_loss / iteration)


# step5 test
def test(test_loader):
    correct = 0
    total = 0
    with torch.no_grad():
        for (x, y) in test_loader:
            # a row in the prediction of y represents a prediction of a example x.
            y_pred = model(x)
            # <class 'torch.Tensor'>
            # print(type(y_pred))
            _, predicted = torch.max(y_pred, dim=1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
        print("Accuracy:", correct / total)


if __name__ == '__main__':
    for epoch in range(10):
        train(epoch, train_loader)
        test(test_loader)
