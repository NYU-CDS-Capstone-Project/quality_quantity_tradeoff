from train import train_valid_model, Net
from plot import plot_results
import torch.optim as optim
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./../data', help='directory for data')
parser.add_argument('--max_layer', type=int, default=2, help='possible number of layers in the CNN')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--epochs', type=int, default=10, help='num epoch')

opt = parser.parse_args()


max_layer = opt.max_layer
layers = [[i, j] for i in range(max_layer) for j in range(max_layer)]
print(layers)

results = []

for layer in layers:
    net = Net(layers=layer)
    print(net)
    nb_params = sum(p.numel() for p in net.parameters())

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    acc = train_valid_model(opt.data_dir, opt.batch_size, net, opt.epochs, optimizer, True)
    results.append({'layers':layer, 'num_params': nb_params, 'accuracy': acc})


with open("test.np", "wb") as fp:
    pickle.dump(results, fp)

plot_results("test.np", 'test.png')
