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
        prog='Plot the response of the neural network against the true response',\
        description=''
        )


parser.add_argument('-loc', default='./train_data/', help='location to stored responses, default: ./train_data')
parser.add_argument('-filename', default="response-0.npz", help='filename, default: response-0.npz')
parser.add_argument('-Nt', default=5, help='number of previous output timesteps used, default: 5')
parser.add_argument('-Ni', default=5, help='number of previous input timesteps used, default: 5')
parser.add_argument('-n', default='0', help='number of response to plot: 0')
parser.add_argument('-mdl_loc', default='./', help='Location to save model: ./learning_data')
parser.add_argument('-mdl_name', default='nn_mdl', help='Name of model, default: nn_mdl')


args = parser.parse_args()

dir = vars(args)['loc']
filename = vars(args)['loc']+'/'+vars(args)['filename']
N_t = int(vars(args)['Nt'])
N_i = int(vars(args)['Ni'])
n = int(vars(args)['n'])
mdl_name = str(vars(args)['mdl_name'])
mdl_loc = str(vars(args)['mdl_loc'])


# Get information regarding chosen response
with open(str(loc+'/readme'),'rb') as filen:
    system,t,numberSims,initial,zeta,wn,numberSims,randomMag,inputRange,inputTime= pickle.load(filen)


os.system("./info.py -loc="+str(dir+'/readme'))


# Still need to get data regarding dataset: Ni Nt


filename = filename.replace(str('0',str(n))

def getFeaturesAndResponse(filename,N_t,N_i):
    # Load Data from created response

    features = np.zeros( (t,N_t+N_i) )   # +1 is for the input
    with np.load(filename) as data:
        print('Loading Data from: ', filename)

        data = np.load(filename)

        y = data['y_']
        ydot = data['y_dot']
        ydotdot = data['y_dotdot']
        input = data['input']

        response_ydot = data['y_dot'] # inputs from given file
        input = data['input']

        for step in range( np.maximum(N_t,N_i), t- np.maximum(N_t,N_i) ):
            for n in range(0,N_i):
                features[step+t*numFile,n] = input[step-n]
            for n in range(0,N_t):
                features[step+t*numFile,N_i+n] = y[step-n]

    return [features,y,ydot,ydotdot,input]


def getResponseFromNN(model,features,steps,N_t,N_i):

    predictions = np.zeros( (steps - np.maximum(N_t,N_i),1) )
    for t_steps in range( np.maximum(N_t,N_i), steps - np.maximum(N_t,N_i) ):
        predictions[t_steps] = model.predict(features[t_steps,:])


    return predictions



if __name__ == '__main__':


    [features,y,ydot,ydotdot,input] = getFeaturesAndResponse(filename,N_t=N_t,N_i=N_i)
    print('loading model from: ', str(mdl_loc+'/'+mdl_name))
    model = keras.models.load_model(str(mdl_loc+'/'+mdl_name))
    predictions = getFeaturesAndResponse(filename,N_t=N_t,N_i=N_i)




    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rc('font', size=12)

    plt.figure(1)
    plt.plot(input,'-', mew=1, ms=8,mec='w')
    plt.plot(y,'-', mew=1, ms=8,mec='w')
    plt.plot(predictions,'-', mew=1, ms=8,mec='w')
    plt.legend(['$input$','$y$', '$\hat y'])
    plt.grid()


    plt.show()