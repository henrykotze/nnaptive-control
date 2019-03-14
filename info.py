#!/usr/bin/env python3



import argparse
import pickle


parser = argparse.ArgumentParser(\
        prog='Displays information regarding training or data',\
        description=''
        )


parser.add_argument('-loc', default='./train_data/readme', help='location of file, default: ./learning_data/training_info')
parser.add_argument('-type', default='train_data', help='information about specific type, ie: model or data')


args = parser.parse_args()
dir = vars(args)['loc']
type = str(vars(args)['type'])




if( type == 'train_data'):


    with open(str(dir),'rb') as filen:
        system,time,numberSims,initial,zeta,wn,numberSims,randomMag,inputMag,inputTime = pickle.load(filen)

    info =  '--------------------------------'+'\n'+'TRAINING DATA INFORMATION'+'\n' \
            + '--------------------------------'+'\n' \
            + 'system: ' + str(system) + '\n' \
            + 'wn: '+str(wn)+'\n' \
            + 'zeta: '+str(zeta)+'\n'+'time: ' + str(time) + ' ms'+'\n' \
            + 'initial condition: ' + str(initial) + '\n' \
            + 'number of responses: ' + str(numberSims) + '\n' \
            + 'input Magnitude: ' + str(inputMag) + '\n' \
            + 'random inputs: ' + str(randomMag) + '\n' \
            + 'start time of input: ' + str(inputTime) + ' ms' '\n' \
            + '--------------------------------'+'\n'

    print(info)


if(type == 'trained_nn'):
    print(type)
    with open(str(dir),'rb') as filen:
        system,epochs,mdl_name,mdl_loc,weight_reg,learning_rate= pickle.load(filen)

    info =  '--------------------------------'+'\n'+'NN MODEL INFORMATION'+'\n' \
            + '--------------------------------'+'\n' \
            + 'system: ' + str(system) + '\n' \
            + 'epochs: '+str(epochs)+'\n' \
            + 'learning_rate: '+str(learning_rate)+'\n' \
            + 'weight regularization: ' + str(weight_reg) + '\n' \
            + 'model name: ' + str(mdl_name) + '\n' \
            + 'model location: ' + str(mdl_loc) + '\n' \
            + '--------------------------------'+'\n'

    print(info)
