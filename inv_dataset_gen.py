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
import shelve


parser = argparse.ArgumentParser(\
        prog='create data 2nd order',\
        description='Creates .npz files of step responses with given damping ratio\
                    frequency'
        )


parser.add_argument('-loc', default='./train_data/', help='location to stored responses, default: ./train_data')
parser.add_argument('-Nt', default=5, help='number of previous output timesteps used, default: 5')
parser.add_argument('-j', default=1, help='')
parser.add_argument('-dataset_name', default='dataset0', help='name of your dataset, default: dataset*')
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
    biases = db['biases']
db.close()


Nt = int(vars(args)['Nt'])
nameOfDataset = str(vars(args)['dataset_name'])
dataset_loc = str(vars(args)['dataset_loc'])
timeSteps = int(t/dt)
j = int(vars(args)['j'])


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
    features = np.zeros( (timeSteps*numberSims,Nt+Nt) )   # +1 is for the input
    labels = np.zeros( (timeSteps*numberSims,2) )
    max_y = 0
    max_ydotdot = 0


    for numFile in range(numberSims):
        with np.load(str(dir+'/'+filename)) as data:
            print('Loading Data from: ', filename)

            response_y = data['y_'] # inputs from given file
            response_ydotdot = data['y_dotdot'] # inputs from given file
            input = data['input']
            bias = data['bias']

            if(np.amax(response_ydotdot) > max_ydotdot):
                 max_ydotdot = np.amax(response_ydotdot)


            if(np.amax(response_y) > max_y):
                 max_y = np.amax(response_y)


            for step in range( Nt, timeSteps - Nt ):

                labels[step+timeSteps*numFile,0] = input[step]
                labels[step+timeSteps*numFile,1] = bias[step]

                for n in range(0,Nt):
                    features[step+timeSteps*numFile,n] = np.sin(response_y[step-n+1])

                for n in range(0,Nt):
                    features[step+timeSteps*numFile,Nt+n] = response_ydotdot[step-n+1]


            # fetch next name of *.npz file to be loaded
            filename = filename.replace(str(numFile),str(numFile+1))

    features[:,Nt:Nt+Nt] = features[:,Nt:Nt+Nt]/max_ydotdot
    with open(dataset_loc + '/'+nameOfDataset,'wb+') as filen:

        print('\n--------------------------------------------------------------')
        print('Saving features and labels to:', nameOfDataset)
        print('\n--------------------------------------------------------------')

        pickle.dump([features,labels],filen)


    filen.close()

    print('max_ydotdot :', max_ydotdot)
    print('max_y :', max_y)
