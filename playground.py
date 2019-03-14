#!/usr/bin/env python3
import argparse




dir = './learning_data/'
readme = 'readme.txt'
filename = dir+readme



f = open(filename, 'r').read()
print(f)
zeta=f.split('zeta: ')

print(zeta)
