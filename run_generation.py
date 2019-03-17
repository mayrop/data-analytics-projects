import csv
import os
import sys
from helpers import *
from uofgsocsai import LochLomondEnv

if len(sys.argv) < 1:
    print("usage: run_generation.py <type>")
    exit()

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

