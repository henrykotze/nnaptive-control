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
parser.add_argument('-loc', default='./train_data/', help='location to store responses, default: ./train_data')
parser.add_argument('-filename', default="response-0.npz", help='filename, default: response-0.npz')
parser.add_argument('-t', default=1000, help='time lenght of responses, default: 1000ms')
parser.add_argument('-numSim', default=2, help='number of responses to generate, default: 1')
parser.add_argument('-inputMag', default=1, help='magnitude of input given to system, default: +-1')
parser.add_argument('-system', default='pendulum', help='type of system to generate data, default: pendulum')
parser.add_argument('-init', default=0, help='Initial Condition, default: 0')
parser.add_argument('-rand', default=0, help='pick from normal distribution the input, default: 0')
parser.add_argument('-inputTime', default=50, help='time at which inputs starts, default: 50ms')
parser.add_argument('-startNumSim', default=0, help='to start the response-* different then 0, default: 0')
parser.add_argument('-timeSteps', default=0.01, help='timestep increments of responses, default: 0.01')
parser.add_argument('-maxInput', default=0.5, help='maximum input given to system')
parser.add_argument('-minInput', default=-0.5, help='minimum input given to system')


args = parser.parse_args()

zeta=float(vars(args)['zeta'])
wn=float(vars(args)['wn'])
time=int(vars(args)['t'])
numberSims = int(vars(args)['numSim'])
startSimNum = int(vars(args)['startNumSim'])
dir = vars(args)['loc']
filename = vars(args)['loc']+'/'+vars(args)['filename']
system = vars(args)['system']
initial = float(vars(args)['init'])
randomMag = int(vars(args)['rand'])
inputTime = int(vars(args)['inputTime'])
timeStep = float(vars(args)['timeSteps'])
inputMag = float(vars(args)['inputMag'])
maxInput = float(vars(args)['maxInput'])
minInput = float(vars(args)['minInput'])

# Add a Readme file in directory to show selected variables that describe the
# responses

filename = filename.replace(str(0),str(startSimNum))

# Write information to the readme file in the directory of the response
with open(str(dir + '/readme'),'wb+') as filen:
    print('Saving training info to')
    pickle.dump([system,time,numberSims,initial,zeta,wn,(numberSims+startSimNum),randomMag,inputMag,inputTime],filen)
filen.close()

def determine_system(system,wn,zeta,initial_condition):
    if(system == 'pendulum'):
        response = pendulum(wn,zeta,y=initial_condition*np.pi/180,time_step=timeStep)
    elif(system =='second'):
        response = second_order(wn,zeta,y=initial_condition)

    return response




if __name__ == '__main__':
    print('Creating the response of ', str(system))
    print('Writing responses to:', filename )


    for numSim in range(startSimNum,startSimNum+numberSims):
        print('Number of simulation: ', numSim)
        response = determine_system(system,wn,zeta,initial)

        # if(randomMag == 0):
        #     response.update_input(inputMag)
        # else:
        #     response.update_input(np.random.uniform(-inputMag,inputMag))

        input = np.zeros( (time,1) )
        y = np.zeros( (time,1) )
        ydot = np.zeros( (time,1) )
        ydotdot = np.zeros( (time,1) )

        for t in range(0,time):
            # time at which input starts
            if(t == inputTime):
                if(randomMag == 0):
                    response.update_input(inputMag)
                else:
                    response.update_input(np.random.uniform(minInput,maxInput))

            # temporary variables
            t1,t2,t3,t4 = response.getAllStates()
            input[t] = t1
            y[t] = t4
            ydot[t] = t3
            ydotdot[t] = t2

            # next time step
            response.step()


        # Saves response in *.npz file
        np.savez(filename,input=input,y_=y,y_dot=ydot,y_dotdot=ydotdot)

        # Change number on filename to correspond to simulation number
        filename = filename.replace(str(numSim),str(numSim+1))
