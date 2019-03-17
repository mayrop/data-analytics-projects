"""
    University of Glasgow 
    Artificial Intelligence 2018-2019
    Assessed Exercise

    Simple Agent
    Purpose: This agent is used as an ideal baseline to find the optimal path under ideal circumstances. 

    Requirements:
    - Sensors: Oracle (i.e. you’re allowed to read the location of all object in the environment e.g. using env.desc)
    - Actions: Discrete and noise-free.
    - State-space: Fully observable a priori
    - Rewards/goal: Fully known a priori (you are allowed to inform the problem with the rewards and location of terminal states)
"""
import sys
from helpers import parse_args
from agents import SimpleAgent

def main(args):
    problem_ids, episodes, grid = parse_args(args)        

    for problem_id in problem_ids:
        agent = SimpleAgent(problem_id=problem_id, map_name_base=grid) 
        agent.solve(episodes=episodes)
        agent.evaluate(episodes)

if __name__ == '__main__':
    main(sys.argv[1:])

