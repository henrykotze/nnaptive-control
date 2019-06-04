#!/usr/bin/env python3
import numpy as np
import sys
import argparse
import tensorflow as tf
from second_order import second_order
from single_pendulum_v2 import pendulum
from linear_control import PIDcontroller
import os
import pickle
import shelve
from tensorflow import keras
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

parser = argparse.ArgumentParser(\
        prog='Test the performance of neural network',\
        description='Environment where the trained neural network is tested'
        )


parser.add_argument('-loc', default='./biased_train_data/', help='location to store responses, default: ./train_data')
parser.add_argument('-init', default=0, help='offset from working point')
parser.add_argument('-wp', default=0, help='working point')
parser.add_argument('-model_path', default='./nn_mdl', help='path to neural network model')


args = parser.parse_args()

model_path = vars(args)['model_path']

dir = vars(args)['loc']

# Working point
wp = float(vars(args)['wp'])
# initial conditions of pendulum
theta = wp + float(vars(args)['init'])




print('----------------------------------------------------------------')
print('Fetching training info from: ', str(dir+'/readme'))
print('----------------------------------------------------------------')
with shelve.open( str(dir+'/readme')) as db:
    zeta=float((db)['zeta'])
    wn=float((db)['wn'])
    sim_time = int((db)['t'])
    dt = float((db)['dt'])
    maxInput = float((db)['maxInput'])

db.close()


print('----------------------------------------------------------------')
print('Training Information: ')
print('----------------------------------------------------------------')
with shelve.open(str(dir+'/readme')) as db:
    for key,value in db.items():
        print("{}: {}".format(key, value))
db.close()
print('wp: ',wp)
print('theta: ', theta)
print('----------------------------------------------------------------')



print('------------------------------------------------------------------------------------------------------')
print('Fetching neural network model from: ', str(model_path ))
print('------------------------------------------------------------------------------------------------------')

nn_model = keras.models.load_model(str(model_path))

inputsize = nn_model.get_input_shape_at(0)

# input to neural network
N=500
nn_input_matrix = np.zeros((1,N))

# Conversion to radians
deg2rad = np.pi/180

wp = wp*deg2rad
theta = 0*deg2rad


data = np.load('./biased_val_data/response-1.npz')
bias = data['bias']
input = data['input']


if __name__ == '__main__':


    # Size for arrays
    total_steps = int(np.ceil(sim_time/dt))
    #
    t = 0
    # counter
    step = 0

    # y generated by nonlinear model
    y_hat = np.zeros(total_steps)
    biased_y_hat = np.zeros(total_steps)
    y_hat_dotdot = np.zeros(total_steps)
    biased_y_hat_dotdot = np.zeros(total_steps)
    bias_pred = np.zeros(total_steps)
    # control received by nonlinear model


    # Nonlinear model of the pendulum
    pendulums = pendulum(wn=wn,zeta=zeta,time_step=dt,y=theta)
    biased_pendulums = pendulum(wn=wn,zeta=zeta,time_step=dt,y=theta)
    # pendulums = second_order(wn=wn,zeta=zeta,time_step=dt,y=theta,y_wp=wp)

    # Linear PID Controller
    P=5
    I=2
    D=0
    linear_controller = PIDcontroller(P,I,D,dt=dt)

    while t < sim_time-dt/2:

        # output of nonlinear model
        y_hat[step] = pendulums.getAllStates()[3]
        biased_y_hat[step] = biased_pendulums.getAllStates()[3]

        y_hat_dotdot[step] = pendulums.getAllStates()[1]
        biased_y_hat_dotdot[step] = biased_pendulums.getAllStates()[1]



        nn_input_matrix = np.roll(nn_input_matrix,1) # move elements one timestep back

        # insert new timestep
        nn_input_matrix[0,0] = input[step]/8
        nn_input_matrix[0,100] = np.sin(y_hat[step])
        nn_input_matrix[0,200] = np.sin(biased_y_hat[step])
        nn_input_matrix[0,300] =  y_hat_dotdot[step]/6.017502740394413
        nn_input_matrix[0,400] = biased_y_hat_dotdot[step]/6.156157104064169

        # reshape
        # print(nn_input_matrix)
        nn_input = nn_input_matrix[0].reshape((1,500))

        nn_output = nn_model.predict(nn_input)

        bias_pred[step] = nn_output

        pendulums.update_input(input[step])
        biased_pendulums.update_input(input[step] + bias[step])
        pendulums.step()
        biased_pendulums.step()

        # Increment time
        t += dt
        # Increment counter
        step += 1



plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)

plt.figure(1)
plt.plot(y_hat,'-', mew=1, ms=8,mec='w')
plt.plot(biased_y_hat,'-', mew=1, ms=8,mec='w')
plt.plot(input,'-', mew=1, ms=8,mec='w')
plt.legend(['$\hat y$','biased $\hat y$', 'input'])
plt.grid()

plt.figure(2)
plt.plot(bias_pred,'-', mew=1, ms=8,mec='w')
plt.plot(bias,'-', mew=1, ms=8,mec='w')
plt.grid()
plt.legend(['bias pred','bias true'])

plt.show()