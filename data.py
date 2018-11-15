import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from torch.utils.data.dataset import Dataset

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Cifar2(Dataset):
    def __init__(self, data, label, transform, label_error_rate=0):
        super(Cifar2, self).__init__()
        self.data = data
        self.label = label
        self.label_error_rate = label_error_rate
        self.transform = transform
        if label_error_rate > 0:
            self.perturb_labels()

    def perturb_labels(self):
        num_false_labels = int(np.floor(self.label_error_rate * len(self.label)))
        indices = list(range(len(self.label)))
        np.random.shuffle(indices)
        false_indices = indices[:num_false_labels]
        self.label[false_indices] = 1 - self.label[false_indices]

    def __getitem__(self, index):
        return self.transform(self.data[index]), self.label[index]

    def __len__(self):
        return self.label.shape[0]


def generate_train_data_label(train_dataset, cat_id, dog_id):
    cat_idx = np.where(np.array(train_dataset.train_labels)==cat_id)[0]
    dog_idx = np.where(np.array(train_dataset.train_labels)==dog_id)[0]
    cat_array = train_dataset.train_data[cat_idx]
    dog_array = train_dataset.train_data[dog_idx]
    train_data = np.concatenate([cat_array, dog_array])
    train_label = np.array([0]*cat_array.shape[0] + [1]*dog_array.shape[0])
    return train_data, train_label

def generate_test_data_label(test_dataset, cat_id, dog_id):
    cat_idx = np.where(np.array(test_dataset.test_labels)==cat_id)[0]
    dog_idx = np.where(np.array(test_dataset.test_labels)==dog_id)[0]
    cat_array = test_dataset.test_data[cat_idx]
    dog_array = test_dataset.test_data[dog_idx]
    test_data = np.concatenate([cat_array, dog_array])
    test_label = np.array([0]*cat_array.shape[0] + [1]*dog_array.shape[0])
    return test_data, test_label

def generate_cifar_loaders(training_size, label_error_rate):
    cat_id = 3
    dog_id = 5
    train_data, train_label = generate_train_data_label(trainset, cat_id, dog_id)
    test_data, test_label = generate_test_data_label(testset, cat_id, dog_id)
    train_set = Cifar2(train_data, train_label, transform, label_error_rate)
    test_set = Cifar2(test_data, test_label, transform)
    batch_size = 4
    valid_size = 0.2
    random_seed = 10

    num_train = len(train_set)
    indices = list(range(num_train))

    np.random.seed(random_seed)
    np.random.shuffle(indices)
    
    indices = indices[:training_size]
    split = int(np.floor(valid_size * training_size))
    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=train_sampler,
        num_workers=2)
    valid_loader = torch.utils.data.DataLoader(
        train_set, batch_size=batch_size, sampler=valid_sampler,
        num_workers=2)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
        num_workers=2)
    return train_loader, valid_loader, test_loader
