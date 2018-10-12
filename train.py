import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F

from data import generate_cifar_loaders

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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


def train(model, train_loader, optimizer, criterion):
    train_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)  # torch.autograd.Variable
        loss = criterion(outputs.view(-1), labels.float())
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    return train_loss/len(train_loader)


def valid(model, valid_loader, criterion):
    valid_loss = 0.0
    correct = 0
    for i, data in enumerate(valid_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs.view(-1), labels.float())
        valid_loss += loss.item()
        pred = outputs.view(-1)>0.5
        correct += (pred.long()==labels).float().mean()

    return valid_loss/len(valid_loader), correct/len(valid_loader)


def train_valid_model(data_dir, batch_size, model, num_epoch, optimizer, early_stopping_limit=10, verbose = False):

    train_loader, valid_loader = generate_cifar_loaders(0)

    criterion = nn.BCEWithLogitsLoss()
    best_acc = 0
    count_no_improv = 0
    for epoch in range(num_epoch):  # loop over the dataset multiple times
        train_loss = train(model.to(device), train_loader, optimizer, criterion)
        if verbose:
            print('Train [%d] loss: %.3f' %
                  (epoch + 1, train_loss))
        valid_loss, valid_acc = valid(model, valid_loader, criterion)
        if valid_acc > best_acc:
            best_acc = valid_acc
            count_no_improv = 0
        else:
            count_no_improv += 1
        if verbose:
            print('Valid [%d] loss: %.3f -- accuracy: %.3f' %
                  (epoch + 1, valid_loss, valid_acc))
        if count_no_improv > early_stopping_limit:
            break
    return best_acc.cpu().numpy()
