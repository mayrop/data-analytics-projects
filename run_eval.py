from run_random import random_agent
from run_simple import simple_agent
from run_rl import rl_agent
import sys
import matplotlib.pyplot as plt
from helpers import *

def main(problem_id):

    random_dataframe, random_stats = random_agent_data = random_agent(problem_id)
    simple_dataframe, simple_stats = simple_agent_data = simple_agent(problem_id)
    rl_dataframe, rl_stats = rl_agent_data = rl_agent(problem_id)

    labels = ['Episodes', 'Mean Reward']
    filename = 'out_{}'.format(problem_id)
    title = 'My title {}'.format(problem_id)
    subtitle = 'My subtitle {}'.format(problem_id)

    plt.plot(random_dataframe['mean_rewards'], '#ee5a24', label='Random Agent')
    plt.plot(simple_dataframe['mean_rewards'], '#8e44ad', label='Simple Agent')
    plt.plot(rl_dataframe['mean_rewards'], '#2ed573', label='Reinforcement Learning Agent')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.suptitle(title)
    plt.title(subtitle)
    plt.legend(loc='best')    
    plt.savefig(filename)
    plt.close()


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: run_rl.py <problem_id>")
        exit()
    
    if not (0 <= int(sys.argv[1]) <= 7):
        raise ValueError("Problem ID must be 0 <= problem_id <= 7")        

    main(int(sys.argv[1]))

