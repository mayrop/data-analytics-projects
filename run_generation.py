"""HELLO CLI
Usage:
    hello.py
    hello.py <name>
    hello.py -h|--help
    hello.py -v|--version
Options:
    <type> grids
"""
import csv
import os
import sys
from helpers import *
from uofgsocsai import LochLomondEnv

if len(sys.argv) < 1:
    print("usage: run_generation.py <type>")
    exit()

def generate_grids(base):
    grids = []
    for i in range(base):
        map_name_base = '{}x{}-base'.format(base, base)
        env = LochLomondEnv(problem_id=i, is_stochastic=True, 
                            reward_hole=-0.02, map_name_base=map_name_base)
        print("I: ", i, " - base: ", base)
        env.render()
        grid = EnvMDP.to_decoded(env).reshape(env.nrow * env.ncol)
        grids.append(np.hstack(([i], grid)))
    
    return grids


def main():
    """Main Program."""
    for i in [4, 8]:
        grids = generate_grids(i)
        with open('out/grids-{}.csv'.format(i), 'w') as file:
            writer = csv.writer(file, quoting=csv.QUOTE_NONNUMERIC)
            writer.writerows(grids)

    file.close()    
    #np.savetxt('out/grids.csv', grids, delimiter=",", fmt='%s') 


if __name__ == '__main__':
    main()

