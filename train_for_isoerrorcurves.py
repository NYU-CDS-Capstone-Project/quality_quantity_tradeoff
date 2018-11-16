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

quantities = [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000]
qualities = [0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.10, 0.05, 0]
num_iters = 4
class1 = 3
class2 = 5
prep_str = datestr + '-class_{}_{}'.format(class1, class2)

for quantity in quantities:
    for quality in qualities:
        net = Net(layers=[1, 2])
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        train_loader, valid_loader, test_loader = generate_cifar_loaders(quantity, quality)
        acc = train_valid_model(net, opt.epochs, optimizer, train_loader,
        valid_loader, prep_str + 'best_model.pt', num_iters, verbose=True)

        net.load_state_dict(torch.load(prep_str + 'best_model.pt'))
        test_acc = valid(net, test_loader,test=True)
        results.append({'quality':quality, 'quantity': quantity, 'accuracy': test_acc})
        print('quality: {} -- quantity: {} -- test_acc: {:.5f}'.format(quality, quantity, test_acc))

with open(prep_str + "isoerror_results.np", "wb") as fp:
    pickle.dump(results, fp)

#plot_results("isoerror_results.np", 'test.png')
