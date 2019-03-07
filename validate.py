#!/usr/bin/env python3


import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os
from second_order import second_order
from single_pendulum import pendulum





filename = './test_data/response-0.npz'
data_directory='./test_data/'


# Getting data
# dir: location of directory containing the *.npz file
# filename: Base filaname to be used
# features: np.array that contains all features
# labels: np.array that contians all labels
def loadData(dir,filename,features=[],labels=[]):
    # in the directory, dir, determine how many *.npz files it contains
    path,dirs,files = next(os.walk(dir))

    for numFile in range(len(files)):
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

# Read the system properties from the readme.txt
def determineSystemProps(dir):
    # f = open(,'r')
    pass



def compare2model(ann,inputMag):

    transferModel = second_order(1,0.8,input=inputMag)

    ann_input = np.array([inputMag,0,0,0])

    ann_response = np.zeros(4)
    model_response = np.zeros(4)


    for t in range(0,8000):


        model_output = transferModel.getAllStates()
        ann_output = ann.predict(np.array([model_output]))


        ann_output = np.append(inputMag,ann_output)
        transferModel.step()




        model_response = np.vstack( (model_response, model_output ) )
        ann_response = np.vstack( ( ann_response, ann_output ) )


    return [model_response, ann_response]


if __name__ == '__main__':

    # Setting up an empty dataset to load the data into

#     [dataset,test_features,test_labels] = loadData(data_directory,filename)
#
#     # print(test_features[0])
#
#     # print(test_features.shape)
#     # print(test_features[0])
#     # print(test_features[1])
# #
# #
    model = keras.models.load_model('./trained_models/model_2nd_order')
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

    [model_output, ann_output] = compare2model(model,0.8)



    fig, ax = plt.subplots()

    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=12)

    ax.plot(ann_output,'b-',label='Neural Network',mew=1, ms=8,mec='w')

    ax.plot(model_output,'r--', label='Mathematical Model', mew=1, ms=8,mec='w')
    ax.grid()
    ax.legend()
    plt.show()
