"""
  University of Glasgow 
  Artificial Intelligence 2018-2019
  Assessed Exercise

  Random Agent
  Solution for an agent without sensory input which takes random actions. 

  Purpose: This agent should be used as a naive baseline.
  Requirements:
  - Sensors: None (/random/full; it doesn’t matter...)
  - Action: Discrete
  - State-space: No prior knowledge (i.e. it has not got a map)
  - Rewards/goal: No prior knowledge (does not know where the goal is located)  
"""
import sys
from helpers import *
from agents import *
from utils import print_table
import numpy as np

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
    seed = True

    if len(argv) > 1 and str.isdigit(argv[1]):
        episodes = int(argv[1])

    if len(argv) > 2:
        grid = '{}_{}-base'.format(argv[2], argv[2])        

    print('Solving with Random Agent')
    print('Problem: ', problem_id)
    print('Grid: ', grid)
    print('Episodes that will run...: ', episodes)
    print("\n\n")

    print("It was found out that setting the seed for random was slow.. you can turn it off with seed=False")
    print("More info in documentation...")
    agent = RandomAgent(problem_id=problem_id, map_name_base=grid) 
    agent.solve(episodes=episodes, seed=True)
    
    agent.env.reset()
    print("This is the environment: ")
    print(agent.env.render())
    agent.write_eval_files()

if __name__ == '__main__':
    main(sys.argv[1:])

