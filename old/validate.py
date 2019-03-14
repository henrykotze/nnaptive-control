#!/usr/bin/env python3


import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from second_order import second_order
from single_pendulum import pendulum
import argparse
import pickle

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
        prog='Validation of Trained Neural Network',\
        description=''
        )


parser.add_argument('-loc', default='./test_data/', help='location of saved unseen data, default: ./test_data')
parser.add_argument('-filename', default="response-0.npz", help='filename, default: response-*')
parser.add_argument('-mdl_loc', default='./learning_data', help='Location of saved model: ./learning_data')
parser.add_argument('-mdl_name', default='nn_mdl', help='Name of saved model: nn_mdl')
parser.add_argument('-inputMag', default=0.5, help='Step input size, default: 0.5')
parser.add_argument('-init', default=0, help='Initial conditions, default = 0')

args = parser.parse_args()
dir = vars(args)['loc']
filename = vars(args)['loc']+'/'+vars(args)['filename']
mdl_name = str(vars(args)['mdl_name'])
mdl_loc = str(vars(args)['mdl_loc'])
inputMag = float(vars(args)['inputMag'])
initial = float(vars(args)['init'])

print('Fetching training info from: ', str(mdl_loc+'/training_info'))


with open(str(mdl_loc+'/training_info'),'rb') as filen:
    system,t,numberSims,initial,zeta,wn,numberSims,randomMag,inputRange= pickle.load(filen)


# Getting data
# dir: location of directory containing the *.npz file
# filename: Base filaname to be used
# features: np.array that contains all features
# labels: np.array that contians all labels
def loadData(dir,filename,features=[],labels=[]):
    for numFile in range(numberSims):
        with np.load(filename) as data:
            print('Loading Data from: ', filename)
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




def compare2model(ann,input,system,time,wn,zeta):

    transferModel = 0

    if(system == 'pendulum'):
        transferModel = pendulum(wn,zeta,input=input,y=initial)
    elif(system == 'second'):
        transferModel = second(wn,zeta,input=input,y=initial)

    ann_input = np.array([input,0,0,0])
    ann_response = np.zeros(4)
    model_response = np.zeros(4)


    for t in range(0,time):
        model_output = transferModel.getAllStates()
        ann_output = ann.predict(np.array([model_output]))
        ann_output = np.append(inputMag,ann_output)
        transferModel.step()
        model_response = np.vstack( (model_response, model_output ) )
        ann_response = np.vstack( ( ann_response, ann_output ) )
    return [model_response, ann_response]


if __name__ == '__main__':

    # Setting up an empty dataset to load the data into

#     [dataset,test_features,test_labels] = loadData(dir,filename)
#
#     # print(test_features[0])
#
#     # print(test_features.shape)
#     # print(test_features[0])
#     # print(test_features[1])
# #
# #
    print('loading model from: ', str(mdl_loc+'/'+mdl_name))
    model = keras.models.load_model(str(mdl_loc+'/'+mdl_name))
    # model.summary
# #
#     loss, mean_asb_error, meas_squared_error, acc = model.evaluate(test_features, test_labels)
# #
# # #
#     print("Restored model, accuracy: {:5.2f}%".format(100*acc))
# #
#     test_prediction = model.predict(test_features).flatten()
# #
# #
#     error = test_prediction - test_labels.flatten()
#     plt.hist(error, bins = 50)
#     plt.xlabel("Prediction Error [MPG]")
#     _ = plt.ylabel("Count")
# #
#     plt.show()

    print('Using model saved at: ', str(mdl_loc+'/'+mdl_name))
    [model_output, ann_output] = compare2model(model,inputMag,system,t,wn,zeta)



    fig, ax = plt.subplots()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=12)

    ax.plot(ann_output,'b-',label='Neural Network',mew=1, ms=8,mec='w')

    ax.plot(model_output,'r--', label='Mathematical Model', mew=1, ms=8,mec='w')
    ax.grid()
    ax.legend()
    plt.show()
