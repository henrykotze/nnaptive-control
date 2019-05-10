#!/usr/bin/env python3
# PYTHON_ARGCOMPLETE_OK


from __future__ import absolute_import, division, print_function

import tensorflow as tf

tf.enable_eager_execution()

import pickle
import shelve
import argcomplete, argparse
import pandas as pd
import tensorflow as tf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np




parser = argparse.ArgumentParser(\
        prog='Plot the data from the event file',\
        description=''
        )


parser.add_argument('-path', default = './', help='path to eventfile')

args = parser.parse_args()
path=str(vars(args)['path'])

print('--------------------PATH------------------------')
print(path)
print('-----------------------------------------------')


log = tf.data.TFRecordDataset(path)




### A method


# tf_size_guidance = {
#         'compressedHistograms': 10,
#         'images': 0,
#         'scalars': 100,
#         'histograms': 1
# }
#
#
# event_acc = EventAccumulator(path, tf_size_guidance)
# event_acc.Reload()
#
# training_accuracies =   event_acc.Scalars('training-accuracy')
# validation_accuracies = event_acc.Scalars('validation_accuracy')
#
# steps = 10
# x = np.arange(steps)
# y = np.zeros([steps, 2])
#
# for i in xrange(steps):
#     y[i, 0] = training_accuracies[i][2] # value
#     y[i, 1] = validation_accuracies[i][2]
#
# plt.plot(x, y[:,0], label='training accuracy')
# plt.plot(x, y[:,1], label='validation accuracy')
#
# plt.xlabel("Steps")
# plt.ylabel("Accuracy")
# plt.title("Training Progress")
# plt.legend(loc='upper right', frameon=True)
# plt.show()
