# Estimator as a Neural Network for the rotary wing UAV
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import os


# Base name for data files:
filename='./learning_data/response-0.npz'
data_directory='./learning_data/'

# Building model
def build_model(dataset):

    # model = Sequential()
    # model.add(Dense(22,input_shape=dataset.output_shapes[0]))
    model = keras.Sequential([
    layers.Dense(22, activation=tf.nn.relu, input_shape=dataset.output_shapes[0] ), \
    layers.Dense(18)])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',    \
                    optimizer=optimizer,        \
                    metrics=['mean_absolute_error', 'mean_squared_error'])

    return model


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mean_absolute_error'],label='Train Error')

    plt.plot(hist['epoch'], hist['val_mean_absolute_error'],label = 'Val Error')
    plt.ylim([0,5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mean_squared_error'],label='Train Error')
    plt.plot(hist['epoch'], hist['val_mean_squared_error'],label = 'Val Error')
    plt.ylim([0,20])
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

    for numFile in range(len(files)):
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

    [dataset,features,labels] = loadData(data_directory,filename)



    model = build_model(dataset)

    model.summary()


    EPOCHS = 50

    history = model.fit(features, labels, epochs=EPOCHS, \
    validation_split = 0.2, verbose=0,callbacks=[PrintDot()])

    plot_history(history)
