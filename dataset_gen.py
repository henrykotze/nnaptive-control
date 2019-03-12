#!/usr/bin/env python3


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from second_order import second_order
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import random as rand
import argparse
from single_pendulum import pendulum
import os
import pickle


parser = argparse.ArgumentParser(\
        prog='create data 2nd order',\
        description='Creates .npz files of step responses with given damping ratio\
                    frequency'
        )


parser.add_argument('-loc', default='./train_data/', help='location to store responses, default: ./learning_data')
parser.add_argument('-filename', default="response-0.npz", help='filename, default: response-0.npz')
parser.add_argument('-system', default='pendulum', help='type of system to generate data, default: pendulum')


args = parser.parse_args()

dir = vars(args)['loc']
filename = vars(args)['loc']+'/'+vars(args)['filename']
system = vars(args)['system']


# Add a Readme file in directory to show selected variables that describe the
# responses

print('----------------------------------------------------------------')
print('Fetching training info from: ', str(dir+'/training_info'))
print('----------------------------------------------------------------')

with open(str(dir+'/training_info'),'rb') as filen:
    system,t,numberSims,initial,zeta,wn,numberSims,randomMag,inputRange,inputTime= pickle.load(filen)

os.system("./info.py -loc="+str(dir+'/training_info'))


# def loadData(dir,filename,features=[],labels=[]):
    # in the directory, dir, determine how many *.npz files it contains


if __name__ == '__main__':

    N = 4
    # Pre-creating correct sizes of arrays
    features = np.zeros( (t*numberSims,N) )
    labels = np.zeros( (t*numberSims,1) )


    for numFile in range(numberSims):
        with np.load(filename) as data:
            print('Loading Data from: ', filename, '\n')

            response_y = data['y_'] # inputs from given file
            input = data['input']

            for step in range(0,t-N):
                labels[step+t*numFile] = response_y[step]
                for n in range(0,N):
                    features[step+t*numFile,n] = response_y[N-n+step]

            # to ensure array size is correct when stacking them
            # if(numFile == 0):
            #     features = temp_features
            #     labels = temp_labels
            # else: # stack all files features and labels on top of eachother
            #     features = np.vstack( ( features, temp_features ) )
            #     labels = np.vstack( ( labels, temp_labels ) )

            # fetch next name of *.npz file to be loaded
            filename = filename.replace(str(numFile),str(numFile+1))


    # each row of `features` corresponds to the same row as `labels`.
    assert features.shape[0] == labels.shape[0]
    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

    print(features_placeholder)
    print(labels_placeholder)

    # returns:
    # dataset with correct size and type to match the features and labels
    # features from all files loaded
    # labels from all files loaded
    dataset = [tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder)),features,labels]



    #
