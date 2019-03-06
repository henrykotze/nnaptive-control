import argparse




parser = argparse.ArgumentParser(\
        prog='my programs name without the extension',\
        description='Description of programs'




        )


parser.add_argument('-zeta', default = 1, help='the dmping ratio the response, default: 1')
parser.add_argument('-wn', default= 2, help='the natural frequency of the response, default: 2')
parser.add_argument('-loc', default='./learning_data', help='location to store responses, default: ./')
parser.add_argument('-filename', default="response-0.npz", help=', default: response-*')

args = parser.parse_args()

print(args)
print(vars(args)['zeta'])


wn = vars(args['wn'])
zeta = vars(args['zeta'])
dir = vars(args['loc'])
filename = vars(args['filename'])
