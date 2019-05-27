#!/usr/bin/env python3

import tensorflow as tf
from second_order import second_order
import numpy as np
import random as rand
import argparse
from single_pendulum_v2 import pendulum
import os
import pickle
import shelve
import sys



parser = argparse.ArgumentParser(\
        prog='inspect the features and labels of a dataset',\
        description=''
        )


parser.add_argument('-n', default=3, help='number of pairs to show')
args = parser.parse_args()
n = int(vars(args)['n'])

np.set_printoptions(threshold=sys.maxsize)

def loadData(dir):
    # in the directory, dir, determine how many *.npz files it contains
    with open(str(dir),'rb') as filen:
        print('==============================================================')
        print('Loading dataset from: ' ,str(dir))
        print('==============================================================\n')
        features,labels = pickle.load(filen)



    return [features,labels]




dir = './datasets/test'


features,labels = loadData(dir)


for item in range(0,n):
    print('===============================================================================')
    print('features-labels pair: ', item)
    print('===============================================================================')
    print('features: ',features[item][:])
    print('label: ',labels[item])
    print('===============================================================================')
