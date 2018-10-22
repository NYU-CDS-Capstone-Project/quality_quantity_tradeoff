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
    def __init__(self, data, label, transform, label_error_rate=0):
        super(Cifar2, self).__init__()
        self.data = data
        self.label = label
        self.label_error_rate = label_error_rate

        self.transform = transform
    def perturb_label(self):
        num_false_labels = int(np.floor(self.label_error_rate * len(labels)))
        indices = list(range(len(labels)))
        np.random.shuffle(indices)
        self.false_indices = indices[:num_false_labels]
        self.label[self.false_indices] = 1 - self.label[self.false_indices]

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.label[index]
    def __len__(self):
        return self.label.shape[0]


train_set = Cifar2(train_data, train_label, transform)


def generate_cifar_loaders(training_size, label_error_rate):
    train_set = Cifar2(train_data, train_label, transform)

    batch_size = 4
    valid_size = 0.2
    random_seed = 10

    num_train = len(train_set)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    np.random.seed(random_seed)
    np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:][:training_size], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler,
        num_workers=2)
    valid_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=valid_sampler,
        num_workers=2)

    return train_loader, valid_loader
