#!/usr/bin/env python3


from second_order import second_order
import numpy as np
import scipy.integrate as spi
import matplotlib.pyplot as plt
import random as rand
import argparse
from single_pendulum_v2 import pendulum, noisy_pendulum
import os
import pickle
import shelve


parser = argparse.ArgumentParser(\
        prog='create data for mutiple 2nd order functions',\
        description='Creates .npz files of step responses with given damping ratio\
                    frequency'
        )


parser.add_argument('-min_zeta', default = 0.3, help='the minimum damping ratio the responses default: 0.1')
parser.add_argument('-max_zeta', default = 1, help='the maximum damping ratio the responses, default: 1.5')
parser.add_argument('-min_wn', default= 0.5, help='the minimum natural frequency of the responses, default: 0.5')
parser.add_argument('-max_wn', default= 5, help='the maximum natural frequency of the responses, default: 5')
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
parser.add_argument('-noise', default=0, help='use a noise pendulum system')



args = parser.parse_args()

min_zeta=float(vars(args)['min_zeta'])
max_zeta=float(vars(args)['max_zeta'])
min_wn=float(vars(args)['min_wn'])
max_wn=float(vars(args)['max_wn'])
time=int(vars(args)['t'])
startSimNum = int(vars(args)['startNumSim'])
numberSims = int(vars(args)['numSim']) + startSimNum
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
noise = int(vars(args)['noise'])



if(noise == 1):
    system = 'noisy_pendulum'

# Add a Readme file in directory to show selected variables that describe the
# responses

filename = filename.replace(str(0),str(startSimNum))

# Store all arguments in a pickle format
with shelve.open( str(dir+'/readme') ) as db:
    for arg in vars(args):
        db[arg] = getattr(args,arg)
db.close()


def determine_system(system,timestep):
    if(system == 'pendulum'):
        response = pendulum(time_step=timestep)

    elif(system =='second'):
        response = second_order(time_step=timestep)

    elif(system =='noisy_pendulum'):
        response = noisy_pendulum(time_step=timestep)

    return response


def generateInput(responseDuration,startInput):

    input = np.zeros( (responseDuration,1) )
    timestep = startInput

    while timestep < responseDuration:
        magInput = np.random.random() # Magnitude Size of Input
        inputDur = int(responseDuration/4*(np.random.random() ) ) # Duration of input
        zeroInputDur = int(responseDuration/4*(np.random.random()) ) # Duration of zero input


        input[timestep:timestep+inputDur] = magInput
        timestep += inputDur
        input[timestep:timestep+zeroInputDur] = 0
        timestep += zeroInputDur

    return input


if __name__ == '__main__':
    print('Creating the response of ', str(system))
    print('Writing responses to:', filename )


    for numSim in range(startSimNum,startSimNum+numberSims):
        print('Number of response: ', numSim)
        response = determine_system(system,timeStep)

        # choosing randomly the natural frequency and damping ratio of response
        response_wn = np.random.uniform(min_wn,max_wn)
        response_zeta = np.random.uniform(min_zeta,max_zeta)

        response.set_wn(response_wn)
        response.set_zeta(response_zeta)


        time = int(4/(response_zeta*response_wn)*(1/timeStep))
        print(time)


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
        np.savez(filename,input=input,y_=y,y_dot=ydot,y_dotdot=ydotdot,zeta=response_zeta,wn=response_wn,system=str(system))

        # Change number on filename to correspond to simulation number
        filename = filename.replace(str(numSim),str(numSim+1))
