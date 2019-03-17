"""
    University of Glasgow 
    Artificial Intelligence 2018-2019
    Assessed Exercise 
"""
import sys
from helpers import compare_utils, parse_args
from agents import RandomAgent, ReinforcementLearningAgent, PassiveAgent, SimpleAgent
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_eval(agents, rows, labels, filename, training=True):
    random_agent_eval = np.array(agents['random'].eval)
    random_x = pd.to_numeric(random_agent_eval[:,1])
    random_y = pd.to_numeric(random_agent_eval[:,6])

    simple_agent_eval = np.array(agents['simple'].eval)
    simple_x = pd.to_numeric(simple_agent_eval[:,1])
    simple_y = pd.to_numeric(simple_agent_eval[:,6])

    if training:
        rl_agent_eval = np.array(agents['rl']._train)
        rl_x = pd.to_numeric(rl_agent_eval[:,1])
        rl_y = pd.to_numeric(rl_agent_eval[:,5])
    else:
        rl_agent_eval = np.array(agents['rl'].eval)
        rl_x = pd.to_numeric(rl_agent_eval[:,1])
        rl_y = pd.to_numeric(rl_agent_eval[:,6])

    plt.plot(random_x[rows], random_y[rows], '-b', label='Random')
    plt.plot(simple_x[rows], simple_y[rows], '-g', label='Simple')
    plt.plot(rl_x[rows], rl_y[rows], '-r', label='RL')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.title('Plot of Episode number vs Reward.')
    plt.legend(loc='upper right')    
    plt.savefig('out/out-all-{}.png'.format(filename))
    plt.close()

def main(args):
    problem_ids, episodes, grid = parse_args(args)

    for problem_id in problem_ids:
        # this seed doesn't work... if needed, change seed to True below
        random_agent = RandomAgent(problem_id=problem_id, map_name_base=grid) 
        random_agent.solve(episodes=episodes, seed=None)
        random_agent.evaluate(episodes)

        simple_agent = SimpleAgent(problem_id=problem_id, map_name_base=grid) 
        simple_agent.solve(episodes=episodes)
        simple_agent.evaluate(episodes)

        rl_agent = ReinforcementLearningAgent(problem_id=problem_id, map_name_base=grid) 
        rl_agent.solve(episodes=episodes)
        rl_agent.evaluate(episodes)

        passive_agent = PassiveAgent(problem_id=problem_id, map_name_base=grid) 
        passive_agent.solve()
        passive_agent.evaluate(episodes)

        labels = ['Episodes', 'Mean Reward']
        agents = {
            'random': random_agent,
            'simple': simple_agent,
            'rl': rl_agent
        }

        plot_eval(agents, range(999), labels, 'first_1000_training')
        plot_eval(agents, range(999), labels, 'first_1000_evaluation', training=False)
        plot_eval(agents, range(episodes), labels, 'training')
        plot_eval(agents, range(episodes), labels, 'evaluation', training=False)

        # compare_utils(passive_agent.U, agent.U)

if __name__ == '__main__':
    main(sys.argv[1:])

