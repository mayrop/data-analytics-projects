import sys
from helpers import *
from agents import *
from utils import print_table

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

def main(argv):
    """Main Program."""
    if len(argv) < 1:
        print("usage: run_rl.py <problem_id> <episodes=10000> <grid=8>")
        exit()

    problem_id = int(argv[0])
    episodes = 10000
    grid = '8x8-base'

    if len(argv) > 1 and str.isdigit(argv[1]):
        episodes = int(argv[1])

    if len(argv) > 2:
        grid = '{}_{}-base'.format(argv[2], argv[2])        

    agent = SimpleAgent(problem_id=problem_id, map_name_base=grid) 
    agent.solve(episodes=episodes)
    agent.evaluate(episodes)

if __name__ == '__main__':
    main(sys.argv[1:])

