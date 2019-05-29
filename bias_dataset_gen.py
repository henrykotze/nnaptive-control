#!/usr/bin/env python3


import tensorflow as tf
# from tensorflow import keras
# from tensorflow.keras import layers
from second_order import second_order
import numpy as np
# import scipy.integrate as spi
# import matplotlib.pyplot as plt
import random as rand
import argparse
from single_pendulum import pendulum
import os
import pickle
import shelve


parser = argparse.ArgumentParser(\
        prog='create data 2nd order',\
        description='Creates .npz files of step responses with given damping ratio\
                    frequency'
        )


parser.add_argument('-loc', default='./biased_train_data/', help='location to stored responses, default: ./train_data')
parser.add_argument('-Nt', default=5, help='number of previous output timesteps used, default: 5')
parser.add_argument('-dataset_name', default='biased_dataset0', help='name of your dataset, default: dataset*')
parser.add_argument('-dataset_loc', default='./datasets/', help='location to store dataset: ./datasets/')


args = parser.parse_args()

dir = vars(args)['loc']

# Getting information from readme file of training data

print('----------------------------------------------------------------')
print('Fetching training info from: ', str(dir+'/readme'))
print('----------------------------------------------------------------')
with shelve.open( str(dir+'/readme')) as db:
    system = db['system']
    t = int(db['t'])
    dt = float(db['dt'])
    numberSims = int(db['numSim'])
    filename = db['filename']
    bias_activated = int(db['biases'])
    maxInput = float(db['maxInput'])
db.close()

if(not bias_activated):
    raise Exception('The data being loaded does not contain an bias')



N_t = int(vars(args)['Nt'])
timeSteps = int(t/dt)
nameOfDataset = str(vars(args)['dataset_name'])
dataset_loc = str(vars(args)['dataset_loc'])



with shelve.open( str(dataset_loc + '/'+nameOfDataset+'_readme') ) as db:
    for arg in vars(args):
        db[arg] = getattr(args,arg)

    with shelve.open(str(dir+'/readme')) as data_readme:
        for key in data_readme:
            db[key] = data_readme[key]

    data_readme.close()
db.close()

if __name__ == '__main__':

    # Pre-creating correct sizes of arrays
    features = np.zeros( (timeSteps*numberSims,3*N_t) )   # +1 is for the input
    labels = np.zeros( (timeSteps*numberSims,1) )
    max_input = 0

    for numFile in range(numberSims):
        with np.load(str(dir+'/'+filename)) as data:
            print('Loading Data from: ', filename)

            biased_response_y = data['biased_y'] # inputs from given file
            response_y = data['y_'] # inputs from given file
            input = data['input']
            bias = data['bias']

            if(np.amax(input) > max_input):
                 max_input = np.amax(input)

            for step in range( N_t, timeSteps- N_t ):
                # print(bias[step])
                labels[step+t*numFile,0] = bias[step]

                for n in range(0,N_t):
                    features[step+timeSteps*numFile,n] = input[step-n]/maxInput
                for n in range(0,N_t):
                    features[step+timeSteps*numFile,N_t+n] = np.sin(response_y[step-n])
                for n in range(0,N_t):
                    features[step+timeSteps*numFile,2*N_t+n] = np.sin(biased_response_y[step-n])

            # fetch next name of *.npz file to be loaded
            filename = filename.replace(str(numFile),str(numFile+1))



    with open(dataset_loc + '/'+nameOfDataset,'wb+') as filen:

        print('\n--------------------------------------------------------------')
        print('Saving features and labels to:', nameOfDataset)
        print('\n--------------------------------------------------------------')

        pickle.dump([features,labels],filen)


    filen.close()
