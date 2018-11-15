import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data.dataset import Dataset
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## Simple CNN
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
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


## VGG adapted
class VGG_net(nn.Module):
    def __init__(self, conv_params, n_features, init_weights=True):
        super(VGG_net, self).__init__()
        self.convnet = make_layers(conv_params)
        self.classifier = nn.Sequential(
            nn.Linear(n_features, n_features),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(n_features, 1),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        x = self.convnet(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

def make_layers(conv_params, batch_norm=False):
    """ conv_params (list): each element corresponds to a layer component (pooling or conv & non-lin)
            - 'M' = maxpooling, stride of 2
            - integer = nb of kernels to apply = nb of output channels
    """
    layers = []
    in_channels = 3
    for v in conv_params:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)






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


def train_valid_model(model, num_epoch, optimizer, train_loader, valid_loader,
                    savepath, num_iters, early_stopping_limit=10, verbose = False):
    criterion = nn.BCEWithLogitsLoss()
    best_acc = 0
    for i in range(num_iters):
        best_acc_iter = 0
        count_no_improv = 0
        model._initialize_weights()
        for epoch in range(num_epoch):  # loop over the dataset multiple times
            train_loss = train(model.to(device), train_loader, optimizer, criterion)
            if verbose:
                print('Train [%d] loss: %.3f' %
                      (epoch + 1, train_loss))
            valid_loss, valid_acc = valid(model, valid_loader, criterion)
            if valid_acc > best_acc_iter:
                best_acc_iter = valid_acc
                count_no_improv = 0
            else:
                count_no_improv += 1
            if valid_acc > best_acc:
                best_acc = valid_acc
                model.save_state_dict(savepath)
            if verbose:
                print('Valid [%d] loss: %.3f -- accuracy: %.3f' %
                      (epoch + 1, valid_loss, valid_acc))
            if count_no_improv > early_stopping_limit:
                break
    return best_acc.cpu().numpy()
