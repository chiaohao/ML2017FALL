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
    d = 20
    xs_train = xs[100:d+100]
    ys_train = ys[100:d+100]

model = load_model(sys.argv[1])
input_img = model.input

for i in range(len(xs_train)):

    plt.figure()
    plt.imshow(xs_train[i].reshape((48,48)), cmap='gray')
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('saliency_map/origin{}.png'.format(i))

    val_proba = model.predict(xs_train[i].reshape((1,48,48,1)))
    pred = val_proba.argmax(axis=-1)
    target = K.mean(model.output[:, pred])
    grads = K.gradients(target, input_img)[0]
    fn = K.function([input_img, K.learning_phase()], [grads])

    g = fn([xs_train[i].reshape((1,48,48,1)), 0])
    heatmap = deprocess_image(np.array(g).reshape((48,48)))

    thres = 0.2
    see = xs_train[i].reshape(48, 48)
    see[np.where(heatmap <= thres)] = np.mean(see)

    plt.figure()
    plt.imshow(heatmap, cmap=plt.cm.jet)
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('saliency_map/color{}.png'.format(i))

    plt.figure()
    plt.imshow(see,cmap='gray')
    plt.colorbar()
    plt.tight_layout()
    fig = plt.gcf()
    plt.draw()
    fig.savefig('saliency_map/gray{}.png'.format(i))