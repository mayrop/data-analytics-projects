import sys
from helpers import *
from agents import *
from utils import print_table
import pandas as pd

def main(problem_id):
    """Write Doc Here"""
    episodes = 10000

    print('Running Simple Agent')
    print('Problem: ', problem_id)

    agent = SimpleAgent(problem_id=problem_id) 
    agent.solve(episodes=episodes)
    policy = agent.policy        
    arrows = policy_to_arrows(policy, 8, 8)
    
    agent.env.reset()
    print("This is the environment: ")
    print(agent.env.render())
    print("This is the final policy: ")    
    print_table(arrows)
    agent.write_eval_files()

    evaluation = np.array(agent.eval)

    # Plotting mean rewards    
    x = pd.to_numeric(evaluation[:,1])
    y = pd.to_numeric(evaluation[:,6])
    
    plt.plot(x, y)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.savefig('out_simple_{}_mean_reward.png'.format(problem_id))
    plt.close()    

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: run_simple.py <problem_id>")
        exit()

    main(int(sys.argv[1]))