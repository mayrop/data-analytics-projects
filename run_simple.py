from helpers import *
from uofgsocsai import LochLomondEnv
import numpy as np
import pandas as pd
import sys
import itertools  
import operator  
from search import *

def get_action_from_location(previous, current):
    # todo - double check

    # previous[0] = y coordinate of prev value
    # previous[1] = x coordinate of prev value

    # current[0] = y coordinate of current value
    # current[1] = x coordinate of current value

    # down
    if current[0] > previous[0]:
        return 1
    # up
    if current[0] < previous[0]:
        return 3
    # right
    if current[1] > previous[1]:
        return 2
    # left
    if current[1] < previous[1]:
        return 0

def simple_agent(problem_id):

    # should be less than or equal to 0.0 (you can fine tune this  depending on you RL agent choice)
    reward_hole = 0.0

    # you can decide you rerun the problem many times thus generating many episodes... you can learn from them all!
    max_episodes = 10000   

    # you decide how many iterations/actions can be executed per episode
    max_iter_per_episode = 100 

    # TODO - add doc on this
    lost_episodes = 0

    actions = []
    results = []

    env = LochLomondEnv(problem_id=problem_id, is_stochastic=False, reward_hole=reward_hole)
    state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)

    undirected_graph = UndirectedGraph(state_space_actions)
    undirected_graph.locations = state_space_locations
    graph_problem = GraphProblem(state_initial_id, state_goal_id, undirected_graph)

    node = astar_search(problem=graph_problem, h=None)
    best_path = node.solution()

    for i in range(len(best_path)):
        if i == 0:
            previous = undirected_graph.locations[state_initial_id]
        else:
            previous = undirected_graph.locations[best_path[i - 1]]

        current = undirected_graph.locations[best_path[i]]

        action = get_action_from_location(previous, current)
        actions.append(action)

    for e in range(max_episodes): # iterate over episodes

        observation = env.reset() # reset the state of the env to the starting state     
        
        for iter in range(max_iter_per_episode):
            # pick an action from the solution
            action = actions[iter]

            # observe what happends when you take the action
            observation, reward, done, info = env.step(action)
          
            # Check if we are done and monitor rewards etc...
            if (done and reward==reward_hole): 
                lost_episodes += 1
                break

            if (done and reward == +1.0):
                break

        results.append([e, iter, int(reward), lost_episodes])

    columns = ['episode', 'iterations', 'reward', 'lost_episodes']
    
    dataframe = pd.DataFrame(data=np.array(results), index=np.array(results)[0:,0], columns=columns)
    dataframe['cumulative_rewards'] = list(itertools.accumulate(dataframe['reward'], operator.add))
    dataframe['mean_rewards'] = dataframe.apply(lambda x: mean_rewards(x), axis=1)

    x = range(1, len(dataframe) + 1)
    y = dataframe['mean_rewards']
    
    title = 'My title: problem id {}'.format(problem_id)
    subtitle = 'My subtitle {}'.format(problem_id)
    labels = ['Episodes', 'Mean Reward']

    add_plot(x, y, 'out_simple_{}_mean_reward.png'.format(problem_id), title, subtitle, labels)

    return dataframe

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: run_simple.py <problem_id>")
        exit()
    
    if not (0 <= int(sys.argv[1]) <= 7):
        raise ValueError("Problem ID must be 0 <= problem_id <= 7")        

    simple_agent(int(sys.argv[1]))

