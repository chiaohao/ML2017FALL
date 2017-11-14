from keras.models import load_model
from keras.utils import np_utils
import keras.backend as K
import argparse
from termcolor import colored,cprint
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
import os
import sys
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def deprocess_image(x):
    x -= x.mean()
    x /= x.std()

    x = np.clip(x, 0, 1)

    return x

with open(sys.argv[2]) as csvfile:
    r = list(csv.reader(csvfile))
    r = r[1:]
    ys = np_utils.to_categorical([int(i[0]) for i in r])
    xs = np.asarray([[int(j) for j in i[1].split(' ')] for i in r])
    xs = xs.reshape(-1,48,48,1) / 255.0
    x = xs[103]
    y = ys[103]

model = load_model(sys.argv[1])
layer_dict = dict([layer.name, layer] for layer in model.layers)
input_img = model.input

layers = ['conv2d_1', 'conv2d_2', 'conv2d_3', 'conv2d_4']
collect_layers = [K.function([input_img, K.learning_phase()], [layer_dict[l].output]) for l in layers]

for cnt, fn in enumerate(collect_layers):
    im = fn([x.reshape(1,48,48,1),0])
    fig = plt.figure(figsize=(56,24))
    nb_filter = im[0].shape[3]
    for i in range(nb_filter):
        ax = fig.add_subplot(nb_filter/16,16,i+1)
        ax.imshow(im[0][0,:,:,i],cmap='PuBu')
        plt.xticks(np.array([]))
        plt.yticks(np.array([]))
        plt.tight_layout()
    fig.suptitle('Output of layer conv2d_{} (Given image{})'.format(cnt,103))
    fig.savefig(os.path.join('filters/layer{}'.format(cnt)))