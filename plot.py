from matplotlib import pyplot as plt
import pickle

def plot_results(datadir, figdir):
    with open(datadir, 'rb') as fd:
        data = pickle.load(datadir)
    plt.figure(figsize=(10, 10))
    data = [(result['num_params'], result['accuracy']) for result in results]
    data = np.array(sorted(data, key=lambda x:x[0]))
    plt.plot(data[:, 0], data[:, 1])

    plt.savefig(savedir)
