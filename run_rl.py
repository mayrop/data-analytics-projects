"""
    University of Glasgow 
    Artificial Intelligence 2018-2019
    Assessed Exercise

    Reinforcement Learning Agent
    Solve the problem with minimal assumptions about the environment.

    Requirements:
    - Sensors: Perfect information about the current state and thus available actions in that state; no prior knowledge about the state-space in general
    - Action: Discrete and noisy. The requested action is only carried out correctly with a certain probability
    - State-space: No prior knowledge, but partially observable/learn-able via the sensors/actions.
    - Rewards: No prior knowledge, but partially observable via sensors/actions.  
"""
import sys
from helpers import compare_utils, parse_args
from agents import ReinforcementLearningAgent, PassiveAgent

def main(args):
    problem_ids, episodes, grid = parse_args(args)

    for problem_id in problem_ids:
        agent = ReinforcementLearningAgent(problem_id=problem_id, map_name_base=grid) 
        agent.solve(episodes=episodes)
        agent.evaluate(episodes)

        passive_agent = PassiveAgent(problem_id=problem_id, map_name_base=grid) 
        passive_agent.solve()
        passive_agent.evaluate(episodes)

        compare_utils(passive_agent.U, agent.U)

if __name__ == '__main__':
    main(sys.argv[1:])

