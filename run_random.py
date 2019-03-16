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
import matplotlib.pyplot as plt
import pandas as pd

def main(problem_id):
    """Write Doc Here"""
    episodes = 10000
    seed = True

    print('Running Random Agent')
    print('Problem: ', problem_id)

    agent = RandomAgent(problem_id=problem_id) 
    agent.solve(episodes=episodes, seed=True)
    
    agent.env.reset()
    print("This is the environment: ")
    print(agent.env.render())
    agent.write_eval_files()

    evaluation = np.array(agent.eval)

    # Plotting mean rewards    
    x = pd.to_numeric(evaluation[:,1])
    y = pd.to_numeric(evaluation[:,6])
    
    plt.plot(x, y)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.savefig('out_random_{}_mean_reward.png'.format(problem_id))
    plt.close()

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: run_rl.py <problem_id>")
        exit()

    main(int(sys.argv[1]))

