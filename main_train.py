
from train import train_valid_model, Net, VGG_net
from plot import plot_results
import torch.optim as optim
import argparse
import pickle


parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', default='./../data', help='directory for data')
parser.add_argument('--model', default='cnn', help='experiment with either simple cnn or vgg')
parser.add_argument('--max_layer', type=int, default=2, help='possible number of layers in the CNN')
parser.add_argument('--batch_size', type=int, default=4, help='batch size')
parser.add_argument('--epochs', type=int, default=50, help='num epoch')
parser.add_argument('--early_stop', type=int, default=10, help='nb of epochs with no improvements before stopping')

opt = parser.parse_args()
model_choice = opt.model

results = []

if model_choice == 'cnn':
    max_layer = opt.max_layer
    layers = [[i, j] for i in range(max_layer) for j in range(max_layer)]
    print(layers)

    for layer in layers:
        net = Net(layers=layer)
        print(net)
        nb_params = sum(p.numel() for p in net.parameters())

        optimizer = optim.Adam(net.parameters(), lr=0.001)
        acc = train_valid_model(opt.data_dir, opt.batch_size, net, opt.epochs, optimizer, verbose=True)
        results.append({'layers':layer, 'num_params': nb_params, 'accuracy': acc})

        
elif model_choice == 'vgg':
    """ Images are 3*32*32
            - with 5 maxpooling, features obtained are of dimension 1
            - with 4 maxpooling, features obtained are of dimension 4, etc.
            - the dimension to feed to the classifier is specified by the param "n_features"
    """
    architectures = {'ex_1': [16, 'M', 32, 'M', 64, 'M', 128, 'M', 256, 'M'],
                     'ex_2': [32, 'M', 64, 'M', 128, 'M', 256, 'M', 512, 'M'],
                     'ex_3': [32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M'],
                     'ex_4': [16, 'M', 32, 'M', 64, 'M', 128, 'M'],
		     'ex_5': [16, 16, 'M', 32, 32, 'M', 64, 64, 'M', 128, 128, 'M', 256, 256, 'M'],
		     'ex_6': [16, 16, 16, 'M', 32, 32, 32, 'M', 64, 64, 64, 'M', 128, 128, 128, 'M', 256, 256, 256, 'M'],
		     'ex_7': [16, 16, 16, 16, 'M', 32, 32, 32, 32, 'M', 64, 64, 64, 64, 'M', 128, 128, 128, 128, 'M', 256, 256, 256, 256, 'M']
		     }
    nb_features = {'ex_1':256, 'ex_2':512, 'ex_3':512, 'ex_4':512, 'ex_5':256, 'ex_6':256, 'ex_7':256}
    
    for ex in architectures.keys():
        net = VGG_net(architectures[ex], nb_features[ex])
        print(architectures[ex])
        nb_params = sum(p.numel() for p in net.parameters())
        
        optimizer = optim.Adam(net.parameters(), lr=0.001)
        acc = train_valid_model(opt.data_dir, opt.batch_size, net, opt.epochs, optimizer, opt.early_stop, verbose=True)
        results.append({'architecture': architectures[ex], 'num_params': nb_params, 'accuracy': acc})
    


with open("test.np", "wb") as fp:
    pickle.dump(results, fp)

plot_results("test.np", 'test.png')
