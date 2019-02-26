# Estimator as a Neural Network for the rotary wing UAV


import tensorflow as tf
import numpy as np
import os



# Base name for data files:
filaname='./learning_data/response-0.npz'


# Building model
def build_model(train_dataset):
    model = keras.Sequential([
    layers.Dense(22, activation=tf.nn.relu, input_shape=[len(train_dataset.keys())]),
    layers.Dense(18, activation=tf.nn.relu),
    layers.Dense(18)])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mean_squared_error',
                optimizer=optimizer,
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
def loadData(dir):

    # in the directory, dir, determine how many data file it contains
    path,dirs,files = next(os.walk(dir))

    for numFile in range(len(files)):
        with np.load(filename) as data:
            features = data["features"]
            labels = data["labels"]


            filename = filename.replace(str(numFile),str(numFile+1))


    # each row of `features` corresponds to the same row as `labels`.
    assert features.shape[0] == labels.shape[0]
    features_placeholder = tf.placeholder(features.dtype, features.shape)
    labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

    return tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))







if __name__ == '__main__':
    pass
