from helpers import *
from uofgsocsai import LochLomondEnv
import numpy as np
import pandas as pd
import sys
import itertools  
import operator  

def random_agent(problem_id):

    # should be less than or equal to 0.0 (you can fine tune this  depending on you RL agent choice)
    reward_hole = 0.0

    # should be False for A-star (deterministic search) and True for the RL agent
    is_stochastic = True 

    # you can decide you rerun the problem many times thus generating many episodes... you can learn from them all!
    max_episodes = 10000   

    # you decide how many iterations/actions can be executed per episode
    max_iter_per_episode = 1000 

    # TODO - add doc on this
    lost_episodes = 0
    cumulative_rewards = 0

    # Generate the specific problem 
    env = LochLomondEnv(problem_id=problem_id, is_stochastic=True, reward_hole=reward_hole)
    results = []

    for e in range(max_episodes): # iterate over episodes
        # Reset the random generator to a known state (for reproducability)
        np.random.seed(e)

        observation = env.reset() # reset the state of the env to the starting state     
        
        for iter in range(max_iter_per_episode):
            # your agent goes here (the current agent takes random actions)
            action = env.action_space.sample() 

            # observe what happends when you take the action
            observation, reward, done, info = env.step(action)
          
            # Check if we are done and monitor rewards etc...
            if (done and reward==reward_hole): 
                lost_episodes += 1
                break

            if (done and reward == +1.0):
                break

        results.append([e, iter, int(reward), lost_episodes])

    columns = ['episode', 'iteration', 'reward', 'lost_episodes']
    
    dataframe = pd.DataFrame(data=np.array(results), index=np.array(results)[0:,0], columns=columns)
    dataframe['cumulative_rewards'] = list(itertools.accumulate(dataframe['reward'], operator.add))
    dataframe['mean_rewards'] = dataframe.apply(lambda x: mean_rewards(x), axis=1)

    x = range(1, len(dataframe) + 1)
    y = dataframe['mean_rewards']
    
    add_plot('out_random_{}_mean_reward.png'.format(problem_id))

    return dataframe

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: run_rl.py <problem_id>")
        exit()
    
    if not (0 <= int(sys.argv[1]) <= 7):
        raise ValueError("Problem ID must be 0 <= problem_id <= 7")        

    random_agent(int(sys.argv[1]))

