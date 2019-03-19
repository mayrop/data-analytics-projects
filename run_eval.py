from run_random import random_agent
import sys
import matplotlib.pyplot as plt

def main(problem_id):

    random_agent_data = random_agent(problem_id)

    x = range(1, len(random_agent_data) + 1)
    y = random_agent_data['mean_rewards']
    
    add_plot('out_random_{}_mean_reward.png'.format(problem_id))

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: run_rl.py <problem_id>")
        exit()
    
    if not (0 <= int(sys.argv[1]) <= 7):
        raise ValueError("Problem ID must be 0 <= problem_id <= 7")        

    main(int(sys.argv[1]))

