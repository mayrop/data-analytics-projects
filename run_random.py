"""
    University of Glasgow 
    Artificial Intelligence 2018-2019
    Assessed Exercise

    Random Agent
    Solution for an agent without sensory input which takes random actions. 

    Purpose: This agent should be used as a naive baseline.
    Requirements:
    - Sensors: None
    - Action: Discrete
    - State-space: No prior knowledge (i.e. it has not got a map)
    - Rewards/goal: No prior knowledge (does not know where the goal is located)  
"""
import sys
from helpers import parse_args
from agents import RandomAgent
import numpy as np

def main(args):
    """Main Program."""     

    problem_ids, episodes, grid = parse_args(args)
    print('It was found out that setting the seed for random was slow.. you can turn it on with seed=True')
    print('More info in documentation...')
    
    # Reset the random generator to a known state (for reproducability)
    np.random.seed(12)

    for problem_id in problem_ids:
        # this seed doesn't work... if needed, change seed to True below
        agent = RandomAgent(problem_id=problem_id, map_name_base=grid) 
        agent.solve(episodes=episodes, seed=None)
        agent.evaluate(episodes)

if __name__ == '__main__':
    main(sys.argv[1:])

