import os
import pickle
import numpy as np
import torch

import plotly
plotly.tools.set_credentials_file(username='rubenstern', api_key='tsHqUIBaeDLvOjJAiWMI')

import plotly.plotly as py
import plotly.graph_objs as go
import plotly.io as pio

class_list = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

def generate_isoerror_plot(data):
    acc_array = []
    qualities = np.sort(np.unique(data[:,0]))
    quantities = np.sort(np.unique(data[:,1]))
    for q in qualities:
        selected_quality = data[data[:,0]==q]
        line_acc = selected_quality[selected_quality[:,1].argsort()][:,2]
        acc_array.append(list(line_acc))
    iso_data=[
        go.Contour(
            z=acc_array,
            x=quantities,
            y=qualities,
            contours=dict(
                coloring ='heatmap',
                showlabels = True,
                labelfont = dict(
                    family = 'Raleway',
                    size = 12,
                    color = 'white',
                )
            )
        )
    ]
    return iso_data

def get_iso(results_file):
    with open(results_file, 'rb') as fd:
        data = pickle.load(fd)

    data = np.array([[100*(1-result['quality']), result['quantity'], float(result['accuracy'])] for result in data])


    #data[58,2] = (data[57,2]+data[59,2])/2
    
    iso_data = generate_isoerror_plot(data)
    
    return iso_data

def plot_save(results_file, iso_data, save_dir):
    class1 = int(results_file.split('class_')[1].split('_')[0])
    class1 = class_list[class1]
    class2 = int(results_file.split('class_')[1].split('_')[1])
    class2 = class_list[class2]
    
    layout=go.Layout(title="Error heatmap for {} vs {} classification".format(class1, class2), xaxis={'title':'quantity'}, yaxis={'title':'quality'})
    f = go.Figure(iso_data, layout)
    figure_file = os.path.join(save_dir, 'heatmap_{}_{}.png'.format(class1, class2))
    pio.write_image(f, figure_file)
    print('Heatmap figure generated and saved for {} and {} at {}'.format(class1, class2, figure_file))
    
    
    
def generate_plot_save(results_file, save_dir=''):
    iso_data = get_iso(results_file)
    plot_save(results_file, iso_data, save_dir)
