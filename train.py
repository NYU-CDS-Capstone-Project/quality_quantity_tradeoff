import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


cat_idx = np.where(np.array(trainset.train_labels)==3)[0]
dog_idx = np.where(np.array(trainset.train_labels)==5)[0]

from torch.utils.data.dataset import Dataset

cat_array = trainset.train_data[cat_idx]
dog_array = trainset.train_data[dog_idx]
train_data = np.concatenate([cat_array, dog_array])
train_label = np.array([0]*cat_array.shape[0] + [1]*dog_array.shape[0])

class Cifar2(Dataset):
    def __init__(self, data, label, transform):
        super(Cifar2, self).__init__()
        self.data = data
        self.label = label
        self.transform = transform
    def __getitem__(self, index):
        return self.transform(self.data[index]), self.label[index]
    def __len__(self):
        return self.label.shape[0]


train_set = Cifar2(train_data, train_label, transform)


batch_size = 4
valid_size = 0.2
random_seed = 10

num_train = len(train_set)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

np.random.seed(random_seed)
np.random.shuffle(indices)

train_idx, valid_idx = indices[split:], indices[:split]
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, sampler=train_sampler,
    num_workers=2)
valid_loader = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, sampler=valid_sampler,
    num_workers=2)


import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, layers=[0, 0]):
        super(Net, self).__init__()
        conv1 = [nn.Conv2d(3, 6, 5)]
        for i in range(layers[0]):
            conv1.append(nn.Conv2d(6, 6, 3, padding=1))
        self.conv1 = nn.Sequential(*conv1)
        self.pool = nn.MaxPool2d(2, 2)
        conv2 = [nn.Conv2d(6, 16, 5)]
        for i in range(layers[1]):
            conv2.append(nn.Conv2d(16, 16, 3, padding=1))
        self.conv2 = nn.Sequential(*conv2)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


criterion = nn.BCEWithLogitsLoss()


def train(model, train_loader, optimizer):
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(train_loader)


def valid(model, valid_loader):
    valid_loss = 0.0
    correct = 0
    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), labels.float())
        valid_loss += loss.item()
        pred = outputs.view(-1)>0.5
        correct += (pred.long()==labels).float().mean()

    return valid_loss/len(valid_loader), correct/len(valid_loader)


def train_valid_model(model, num_epoch, optimizer, verbose = False):
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        best_acc = 0
        train_loss = train(model, train_loader, optimizer)
        if verbose:
            print('Train [%d] loss: %.3f' %
                  (epoch + 1, train_loss))
        valid_loss, valid_acc = valid(model, valid_loader)
        if valid_acc > best_acc:
            best_acc = valid_acc
        if verbose:
            print('Valid [%d] loss: %.3f -- accuracy: %.3f' %
                  (epoch + 1, valid_loss, valid_acc))
    return best_acc.numpy()
