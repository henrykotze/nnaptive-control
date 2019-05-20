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
from scipy.optimize import curve_fit


def func(x,a,b):
    return a*x+b

x = np.array([0,50]);
y = np.array([0.01,0.5]);

y_ = np.log(y);


popt, pcov = curve_fit(func,x,y_)

print(popt)

A = np.exp(popt[1])
b = popt[0]

xx = np.arange(0,50);


yy = A*np.exp(b*xx)

plt.figure(1)
plt.plot(yy,'.-', mew=1, ms=8,mec='w')
plt.show()
