from train import train_valid_model, Net
from plot import plot_results
import torch.optim as optim
import argparse
import pickle

from data import generate_cifar_loaders

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./../data', help='directory for data')
parser.add_argument('--max_layer', type=int, default=2, help='possible number of layers in the CNN')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--epochs', type=int, default=10, help='num epoch')

opt = parser.parse_args()


max_layer = opt.max_layer
results = []

for quantity in [8000, 10000]:
    for quality in [0, 0.05]:
        net = Net(layers=[1, 1])
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        train_loader, valid_loader = generate_cifar_loaders(quantity, quality)
        acc = train_valid_model(net, opt.epochs, optimizer, train_loader, valid_loader, verbose=True)
        results.append({'quality':quality, 'quantity': quantity, 'accuracy': acc})


with open("isoerror_results.np", "wb") as fp:
    pickle.dump(results, fp)

plot_results("isoerror_results.np", 'test.png')
