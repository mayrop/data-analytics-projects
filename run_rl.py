import sys
from helpers import *
from agents import *
from utils import print_table
import pandas as pd

def main(problem_id):
    """Write Doc Here"""
    episodes = 10000

    print('Running Reinforment Learning Agent')
    print('Problem: ', problem_id)

    agent = ReinforcementLearningAgent(problem_id=problem_id) 
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
    train = np.array(agent._train)

    # Plotting mean rewards    
    x = pd.to_numeric(train[:,1])
    y = pd.to_numeric(train[:,5])
    
    plt.plot(x, y)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.savefig('out_rl_{}_mean_train_reward.png'.format(problem_id))
    plt.close()    

    # Plotting mean rewards    
    x = pd.to_numeric(evaluation[:,1])
    y = pd.to_numeric(evaluation[:,6])
    
    plt.plot(x, y)
    plt.xlabel('Episodes')
    plt.ylabel('Mean Reward')
    plt.savefig('out_rl_{}_mean_evaluation_reward.png'.format(problem_id))
    plt.close()    


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: run_simple.py <problem_id>")
        exit()

    main(int(sys.argv[1]))
