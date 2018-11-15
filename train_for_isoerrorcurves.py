from train import train_valid_model, Net, valid
from plot import plot_results
import torch
import torch.optim as optim
import argparse
import pickle
import datetime
datestr = datetime.datetime.now().strftime("%m_%d-%H_%M")

from data import generate_cifar_loaders

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./../data', help='directory for data')
parser.add_argument('--max_layer', type=int, default=2, help='possible number of layers in the CNN')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='num epoch')

opt = parser.parse_args()


max_layer = opt.max_layer
results = []

quantities = [8000]
qualities = [0]
num_iters = 1

for quantity in quantities:
    for quality in qualities:
        net = Net(layers=[1, 2])
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        train_loader, valid_loader, test_loader = generate_cifar_loaders(quantity, quality)
        acc = train_valid_model(net, opt.epochs, optimizer, train_loader,
        valid_loader, datestr + 'best_model.pt', num_iters, verbose=True)

        net.load_state_dict(torch.load(datestr + 'best_model.pt'))
        test_acc = valid(net, test_loader,test=True)
        results.append({'quality':quality, 'quantity': quantity, 'accuracy': test_acc})


with open(datestr + "isoerror_results.np", "wb") as fp:
    pickle.dump(results, fp)

#plot_results("isoerror_results.np", 'test.png')
