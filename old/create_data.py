#!/usr/bin/env python3


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


parser.add_argument('-zeta', default = 1, help='the damping ratio the response, default: 1')
parser.add_argument('-wn', default= 2, help='the natural frequency of the response, default: 2')
parser.add_argument('-loc', default='./learning_data/', help='location to store responses, default: ./learning_data')
parser.add_argument('-filename', default="response-0.npz", help='filename, default: response-0.npz')
parser.add_argument('-t', default=1000, help='time lenght of responses, default: 1000ms')
parser.add_argument('-numSim', default=2, help='number of responses to generate, default: 1')
parser.add_argument('-inputMag', default=1, help='magnitude of input given to system, default: +-1')
parser.add_argument('-system', default='pendulum', help='type of system to generate data, default: pendulum')
parser.add_argument('-init', default=0, help='Initial Condition, default: 0')
parser.add_argument('-rand', default=0, help='pick from normal distribution the input, default: 0')


args = parser.parse_args()

zeta=float(vars(args)['zeta'])
wn=float(vars(args)['wn'])
time=int(vars(args)['t'])
numberSims = int(vars(args)['numSim'])
dir = vars(args)['loc']
filename = vars(args)['loc']+'/'+vars(args)['filename']
inputMag = float(vars(args)['inputMag'])
system = vars(args)['system']
initial = float(vars(args)['init'])
randomMag = int(vars(args)['rand'])


# Add a Readme file in directory to show selected variables that describe the
# responses


# Write information to the readme file in the directory of the response
with open(str(dir + '/training_info'),'wb+') as filen:
    print('Saving training info to')
    pickle.dump([system,time,numberSims,initial,zeta,wn,numberSims,randomMag,inputMag],filen)
filen.close()

def determine_system(system,wn,zeta,initial_condition):
    if(system == 'pendulum'):
        response = pendulum(wn,zeta,y=initial_condition*np.pi/180)
    elif(system =='second'):
        response = second_order(wn,zeta,y=initial_condition)

    return response



if __name__ == '__main__':
    print('Creating the response of ', str(system))
    print('Writing responses to:',system,'\n','zeta:',zeta,'\n','wn:', wn)



    for numSim in range(0,numberSims):

        print('Number of simulation: ', numSim)
        response = determine_system(system,wn,zeta,initial)

        if(randomMag == 0):
            response.update_input(inputMag)
        else:
            response.update_input(np.random.uniform(-inputMag,inputMag))


        input = np.zeros(4)
        output = np.zeros(3)


        for t in range(0,time):

            input = np.vstack( (input, response.getAllStates()))
            response.step()
            output = np.vstack( (output, response.getEstimatedStates() ) )


        # Saves response in *.npz file
        np.savez_compressed(filename,features=input,labels=output)

        # Change number on filename to correspond to simulation number
        filename = filename.replace(str(numSim),str(numSim+1))
