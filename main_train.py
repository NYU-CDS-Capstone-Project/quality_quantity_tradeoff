from train import train_valid_model, Net
from plot import plot_results
import torch.optim as optim

max_layer = 2
layers = [[i, j] for i in range(max_layer) for j in range(max_layer)]
print(layers)

results = []

for layer in layers:
    net = Net(layers=layer)
    print(net)
    nb_params = sum(p.numel() for p in net.parameters())

    optimizer = optim.Adam(net.parameters(), lr=0.001)
    acc = train_valid_model(net, 10, optimizer, True)
    results.append({'layers':layer, 'num_params': nb_params, 'accuracy': acc})


with open("test.txt", "wb") as fp:
    pickle.dump(results, fp)

plot_results("test.txt")
