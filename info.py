#!/usr/bin/env python3



import argparse
import pickle
import shelve


parser = argparse.ArgumentParser(\
        prog='Displays information regarding training or data',\
        description=''
        )


parser.add_argument('-path', default='./train_data/readme', help='location of file, default: ./learning_data/readme')


args = parser.parse_args()
path = str(vars(args)['path'])

print('--------------------------------'+'\n'+'README INFORMATION')
print('--------------------------------')
with shelve.open( path ) as db:
    for key in db:
        print(str(key)+': ' + str( db[key] ) )
db.close()
print('--------------------------------')
