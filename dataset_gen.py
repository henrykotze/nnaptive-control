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


parser.add_argument('-loc', default='./train_data/', help='location to stored responses, default: ./train_data')
parser.add_argument('-filename', default="response-0.npz", help='filename, default: response-0.npz')
parser.add_argument('-system', default='pendulum', help='type of system to generate data, default: pendulum')
parser.add_argument('-Nt', default=5, help='number of previous output timesteps used, default: 5')
parser.add_argument('-Ni', default=5, help='number of previous input timesteps used, default: 5')
parser.add_argument('-numSim', default=0, help='number of responses to add, default: all')
parser.add_argument('-name', default='dataset0', help='name of your dataset, default: dataset*')
parser.add_argument('-data_loc', default='./datasets/', help='location to store dataset: ./datasets/')


args = parser.parse_args()

dir = vars(args)['loc']
filename = vars(args)['loc']+'/'+vars(args)['filename']
system = vars(args)['system']
N_t = int(vars(args)['Nt'])
N_i = int(vars(args)['Ni'])
numSim = int(vars(args)['numSim'])
nameOfDataset = str(vars(args)['name'])
dataset_loc = str(vars(args)['data_loc'])

# Add a Readme file in directory to show selected variables that describe the
# responses

print('----------------------------------------------------------------')
print('Fetching training info from: ', str(dir+'/training_info'))
print('----------------------------------------------------------------')

with open(str(dir+'/readme'),'rb') as filen:
    system,t,numberSims,initial,zeta,wn,randomMag,inputRange,inputTime= pickle.load(filen)

os.system("./info.py -loc="+str(dir+'/readme'))

if(numSim != 0):
    numberSims = numSim




if __name__ == '__main__':

    # Pre-creating correct sizes of arrays
    features = np.zeros( (t*numberSims,N_t+N_i) )   # +1 is for the input
    labels = np.zeros( (t*numberSims,1) )


    for numFile in range(numberSims):
        with np.load(filename) as data:
            print('Loading Data from: ', filename)

            response_y = data['y_'] # inputs from given file
            response_ydot = data['y_dot'] # inputs from given file
            input = data['input']

            for step in range( np.maximum(N_t,N_i), t- np.maximum(N_t,N_i) ):

                labels[step+t*numFile] = response_y[step+1]

                for n in range(0,N_i):
                    features[step+t*numFile,n] = input[step-n]
                for n in range(0,N_t):
                    features[step+t*numFile,N_i+n] = response_y[step-n]
                # for n in range(0,N_t):
                    # features[step+t*numFile,N_i+n+N_t] = response_ydot[step-n]


            # fetch next name of *.npz file to be loaded
            filename = filename.replace(str(numFile),str(numFile+1))


    with open(dataset_loc + '/'+nameOfDataset,'wb+') as filen:

        print('\n-----------------------------------------')
        print('Saving features and labels to:', nameOfDataset)
        print('-----------------------------------------')

        pickle.dump([features,labels],filen)


    filen.close()
