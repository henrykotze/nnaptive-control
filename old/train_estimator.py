#!/usr/bin/env python3


# Estimator as a Neural Network for the rotary wing UAV
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
import argparse
import pickle

parser = argparse.ArgumentParser(\
        prog='Trains the Neural Network ',\
        description=''
        )


parser.add_argument('-loc', default='./learning_data/', help='location of stored responses, default: ./learning_data/')
parser.add_argument('-filename', default="response-0.npz", help='filename, default: response-*')
parser.add_argument('-epochs', default=1, help='Number of Epochs, default: 1')
parser.add_argument('-mdl_loc', default='./learning_data/', help='Location to save model: ./learning_data')
parser.add_argument('-mdl_name', default='nn_mdl', help='Name of model, default: nn_mdl')
parser.add_argument('-reg_w', default='0', help='Regularization of weight, default: 0')
parser.add_argument('-lr', default='0', help='learning rate, default: 0')





args = parser.parse_args()
dir = vars(args)['loc']
filename = vars(args)['loc']+'/'+vars(args)['filename']
epochs = int(vars(args)['epochs'])
mdl_name = str(vars(args)['mdl_name'])
mdl_loc = str(vars(args)['mdl_loc'])
weight_reg = float(vars(args)['reg_w'])
learning_rate = float(vars(args)['lr'])


with open(str(dir+'/training_info'),'rb') as filen:
    system,t,numberSims,initial,zeta,wn,numberSims,randomMag,inputMag = pickle.load(filen)
# Base name for data files:
# filename='./learning_data/response-0.npz'
# data_directory='./learning_data/'

# Building model
def build_model(dataset):

    model = keras.Sequential([
    # layers.Flatten(input_shape=(4,)),\
    layers.Dense(20,kernel_regularizer=keras.regularizers.l2(weight_reg),input_shape=dataset.output_shapes[0] ), \
    layers.ReLU(),\
    # layers.Dense(4,kernel_regularizer=keras.regularizers.l2(weight_reg)),\
    # layers.ReLU(),\
    layers.Dropout(0.4),\
    # layers.ReLU(),\
    # layers.Dropout(0.2),\
    layers.Dense(3,kernel_regularizer=keras.regularizers.l2(weight_reg))])

    optimizer = tf.keras.optimizers.Adam(lr=learning_rate)

    model.compile(loss='mean_squared_error',    \
                    optimizer=optimizer,        \
                    metrics=['mean_absolute_error', 'mean_squared_error', 'accuracy'])

    return model


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],label='Train Error')

    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],label = 'Val Error')
    # plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],label = 'Val Error')
    # plt.ylim([0,20])
    plt.legend()
    plt.show()





# Getting data
# dir: location of directory containing the *.npz file
# filename: Base filaname to be used
# features: np.array that contains all features
# labels: np.array that contians all labels
def loadData(dir,filename,features=[],labels=[]):
    # in the directory, dir, determine how many *.npz files it contains
    path,dirs,files = next(os.walk(dir))

    for numFile in range(numberSims):
        with np.load(filename) as data:
            print('Loading Data from: ', filename, '\n')
            temp_features = data["features"] # inputs from given file
            temp_labels = data["labels"] # outputs from given file
            # to ensure array size is correct when stacking them
            if(numFile == 0):
                features = temp_features
                labels = temp_labels
            else: # stack all files features and labels on top of eachother
                features = np.vstack( ( features, temp_features ) )
                labels = np.vstack( ( labels, temp_labels ) )

            # fetch next name of *.npz file to be loaded
            filename = filename.replace(str(numFile),str(numFile+1))


    # each row of `features` corresponds to the same row as `labels`.
    assert features.shape[0] == labels.shape[0]
    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

    # returns:
    # dataset with correct size and type to match the features and labels
    # features from all files loaded
    # labels from all files loaded
    return [tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder)),features,labels]


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')



if __name__ == '__main__':

    # Setting up an empty dataset to load the data into

    [dataset,features,labels] = loadData(dir,filename)

    # print(dataset.output_shapes[0])



    model = build_model(dataset)

    model.summary()

    with open(str(mdl_loc + '/'+mdl_name+'_info'),'wb+') as filen:

        print('Saving training info to:', str(mdl_loc+'/'+mdl_name))
        pickle.dump([system,epochs,mdl_name,mdl_loc,weight_reg,learning_rate],filen)

    filen.close()

    history = model.fit(features, labels, epochs=epochs, \
    validation_split = 0.2, verbose=1)

    plot_history(history)

    print('\n Model Saved at: ' , str(mdl_loc + '/' + mdl_name))
    model.save(mdl_loc+'/'+mdl_name)
