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


parser.add_argument('-init', default=0, help='offset from working point')
parser.add_argument('-wp', default=0, help='working point')
parser.add_argument('-bias_model_path', default='./nn_mdl', help='path to neural network model')
parser.add_argument('-inv_model_path', default='./nn_mdl', help='path to neural network model')


args = parser.parse_args()

bias_model_path = vars(args)['bias_model_path']
inv_model_path = vars(args)['inv_model_path']

def getReadmePath(path):
    readme = ''
    if 'checkpoints' in path:
        dirs = path.split('/')
        pos = dirs.index("checkpoints")
        for i in range(0,pos):
            readme += dirs[i] + '/'

    else:
        dirs = path.split('/')
        pos = dirs.index("nn_mdl")
        for i in range(0,pos):
            readme += dirs[i] + '/'

    readme += 'readme'
    return readme


def compareBiasAndInvTraining(inv_readme, bias_readme):
    zeta_inv = 0
    wn_inv = 0
    dt_inv = 0
    maxInput_inv = 0

    zeta_bias = 0
    wn_bias = 0
    dt_bias = 0
    maxInput_bias = 0

    with shelve.open(inv_readme) as db:

        zeta_inv=float((db)['zeta'])
        wn_inv=float((db)['wn'])
        dt_inv = float((db)['dt'])
        maxInput_inv = float((db)['maxInput'])

    db.close()

    with shelve.open(bias_readme) as db:

        zeta_bias=float((db)['zeta'])
        wn_bias=float((db)['wn'])
        dt_bias = float((db)['dt'])
        maxInput_bias = float((db)['maxInput'])

    db.close()

    if(zeta_inv == zeta_bias and wn_inv == wn_bias and dt_inv == dt_bias and maxInput_inv == maxInput_bias):
        return True
    else:
        return False




bias_model_readme = getReadmePath(bias_model_path)
inv_model_readme = getReadmePath(inv_model_path)


# Working point
wp = float(vars(args)['wp'])
# initial conditions of pendulum
theta = wp + float(vars(args)['init'])


if(compareBiasAndInvTraining(inv_model_readme,bias_model_readme)):
    pass
else:
    raise Exception("Training Data of bias model and inverse model differ")




print('----------------------------------------------------------------')
print('Fetching training info from: ', inv_model_readme)
print('----------------------------------------------------------------')
with shelve.open( inv_model_readme) as db:
    zeta=float((db)['zeta'])
    wn=float((db)['wn'])
    sim_time = int((db)['t'])
    dt = float((db)['dt'])
    maxInput = float((db)['maxInput'])

db.close()


print('----------------------------------------------------------------')
print('Inverse Training Information: ')
print('----------------------------------------------------------------')
with shelve.open(inv_model_readme) as db:
    for key,value in db.items():
        print("{}: {}".format(key, value))
db.close()


print('----------------------------------------------------------------')
print('Bias Training Information: ')
print('----------------------------------------------------------------')
with shelve.open(bias_model_readme) as db:
    for key,value in db.items():
        print("{}: {}".format(key, value))
db.close()


print('------------------------------------------------------------------------------------------------------')
print('Fetching neural network model from: ')
print('------------------------------------------------------------------------------------------------------')

inv_nn_model = keras.models.load_model(str(inv_model_path))
bias_nn_model = keras.models.load_model(str(bias_model_path))

inv_inputsize = inv_nn_model.get_input_shape_at(0)
bias_inputsize = bias_nn_model.get_input_shape_at(0)

# input to neural network
bias_N=500
inv_N=400
bias_nn_input_matrix = np.zeros((1,bias_N))
inv_nn_input_matrix = np.zeros((1,inv_N))

# Conversion to radians
deg2rad = np.pi/180

wp = wp*deg2rad
theta = 0*deg2rad


bias_data = np.load('./biased_train_data/response-5.npz')
bias = bias_data['bias']/2



def invModel(y_ref,y_dotdot_ref,inv_model,nn_input_matrix):

    nn_input_matrix = np.roll(nn_input_matrix,1) # move elements one timestep back
    # insert new timestep
    nn_input_matrix[0,0] = np.sin(y_ref)
    nn_input_matrix[0,200] = y_dotdot_ref/7.2857607535455475
    # reshape
    # print(nn_input_matrix)
    inv_mdl_input = nn_input_matrix[0].reshape((1,400))
    control_output = inv_model.predict(inv_mdl_input)

    return [control_output,nn_input_matrix]


def biasIdentification(control_input,y,y_ddt,y_biased,y_ddt_biased,nn_input_matrix,nn_model):

    nn_input_matrix = np.roll(nn_input_matrix,1) # move elements one timestep back

    # insert new timestep
    nn_input_matrix[0,0] = control_input/8
    nn_input_matrix[0,100] = np.sin(y)
    nn_input_matrix[0,200] = np.sin(y_biased)
    nn_input_matrix[0,300] =  y_ddt/6.017502740394413
    nn_input_matrix[0,400] = y_ddt_biased/6.156157104064169

    nn_input = nn_input_matrix[0].reshape((1,500))

    bias = nn_model.predict(nn_input)

    return [bias,nn_input_matrix]



ref =   40.10704 * deg2rad


if __name__ == '__main__':
    sim_time = 15

    # Size for arrays
    total_steps = int(np.ceil(sim_time/dt))
    #
    t = 0
    # counter
    step = 0

    # y generated by nonlinear model
    y_hat = np.zeros(total_steps)
    y_star = np.zeros(total_steps)
    y_uncompensated = np.zeros(total_steps)
    biased_y_hat = np.zeros(total_steps)
    y_hat_dotdot = np.zeros(total_steps)
    biased_y_hat_dotdot = np.zeros(total_steps)
    bias_pred = np.zeros(total_steps)
    y_lin = np.zeros(total_steps)

    # Error between y_hat and y_star
    e = np.zeros(total_steps)

    # Control generated by linear controller
    u_star = np.zeros(total_steps)

    u_ideal = np.zeros(total_steps)

    # control received by nonlinear model
    u_hat = np.zeros(total_steps)

    # Generated neural network control output
    u_nn = np.zeros(total_steps)

    # Error between reference and y_hat
    error = np.zeros(total_steps)
    # control received by nonlinear model

    bias = np.ones(total_steps)*0.8


    # Nonlinear model of the pendulum
    pendulums = pendulum(wn=wn,zeta=zeta,time_step=dt,y=theta)
    biased_pendulums = pendulum(wn=wn,zeta=zeta,time_step=dt,y=theta)
    linearised_model = second_order(wn=wn,zeta=zeta,time_step=dt,y=theta,y_wp=wp)

    uncompensated_pendulums = pendulum(wn=wn,zeta=zeta,time_step=dt,y=theta)

    linearised_model2 = second_order(wn=wn,zeta=zeta,time_step=dt,y=theta,y_wp=wp)


    # Linear PID Controller
    P=2
    I=2
    D=0
    linear_controller = PIDcontroller(P,I,D,dt=dt)
    linear_controller2 = PIDcontroller(P,I,D,dt=dt)
    linear_controller3 = PIDcontroller(P,I,D,dt=dt)

    while t < sim_time-dt/2:

        # output of nonlinear model
        y_hat[step] = pendulums.getAllStates()[3]
        y_star[step] = linearised_model.getAllStates()[3]
        biased_y_hat[step] = biased_pendulums.getAllStates()[3]
        y_uncompensated[step] = uncompensated_pendulums.getAllStates()[3]
        y_lin[step] = linearised_model2.getAllStates()[3]

        y_hat_dotdot[step] = pendulums.getAllStates()[1]
        biased_y_hat_dotdot[step] = biased_pendulums.getAllStates()[1]

        e[step] = y_hat[step] - y_star[step]

        # error[step] = ref - y_hat[step]
        error[step] = ref - biased_y_hat[step]
        linear_controller.step(error[step])
        control_output = linear_controller.outputControl()


        linear_controller2.step(ref-y_uncompensated[step])
        control_output2 = linear_controller2.outputControl()
        linear_controller3.step(ref-y_lin[step])
        control_output3 = linear_controller3.outputControl()

        u_star[step] = control_output

        # Determine where the linear controller will take the linearised model
        linearised_model.update_input(u_star[step])
        linearised_model.step()
        y_ref = linearised_model.getAllStates()[3]
        y_dotdot_ref = linearised_model.getAllStates()[1]


        [control_input,inv_nn_input_matrix] = invModel(y_ref,y_dotdot_ref,inv_nn_model,inv_nn_input_matrix)
        u_nn[step] = control_input

        [bias_nn_output, bias_nn_input_matrix] = biasIdentification(control_input,y_hat[step],y_hat_dotdot[step],biased_y_hat[step],biased_y_hat_dotdot[step],bias_nn_input_matrix,bias_nn_model)
        bias_pred[step] = bias_nn_output

        u_hat[step] = u_nn[step] + bias[step] -bias_pred[step]

        pendulums.update_input(control_input)
        biased_pendulums.update_input(u_hat[step])
        uncompensated_pendulums.update_input(control_output2+bias[step])


        linearised_model2.update_input(control_output3+bias[step])

        pendulums.step()
        biased_pendulums.step()
        uncompensated_pendulums.step()
        linearised_model2.step()

        # Increment time
        t += dt
        # Increment counter
        step += 1



plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rc('font', size=12)

plt.figure(1)
# plt.plot(y_hat,'-', mew=1, ms=8,mec='w')
# plt.plot(y_star,'-', mew=1, ms=8,mec='w')
plt.plot(biased_y_hat,'-', mew=1, ms=8,mec='w')
plt.plot(y_uncompensated,'-', mew=1, ms=8,mec='w')
plt.plot(y_lin,'-', mew=1, ms=8,mec='w')
plt.legend(['compensated $\hat y$','uncompensated $y$','linearised + bias $y$'])
# plt.legend(['$\hat y$','$y^{*}$','biased $\hat y$','$y uncompensated $','$y_lin$'])
plt.grid()


plt.figure(2)
plt.plot(bias_pred,'-', mew=1, ms=8,mec='w')
plt.plot(bias,'-', mew=1, ms=8,mec='w')
plt.grid()
plt.legend(['bias pred','bias true'])

plt.show()
