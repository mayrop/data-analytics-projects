"""
    University of Glasgow 
    Artificial Intelligence 2018-2019
    Assessed Exercise

    Basic demo code for the CUSTOM Open AI Gym problem used in AI (H) '18-'19
    https://arxiv.org/pdf/1802.05313.pdf
https://arxiv.org/pdf/1802.05313.pdf
https://arxiv.org/pdf/1806.04242.pdf
https://towardsdatascience.com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10
https://www.google.co.uk/search?biw=1608&bih=937&tbm=isch&sa=1&ei=x5mAXOGvNYC71fAP1YingA0&q=performance+measure+in+artificial+intelligence+plots+frozenlake&oq=performance+measure+in+artificial+intelligence+plots+frozenlake&gs_l=img.3...11470.12925..13059...0.0..0.101.559.10j1......1....1..gws-wiz-img.PU8F5GU2FRU#imgrc=_

"""
import os, sys
import pandas as pd
import gym
import numpy as np
import time
from helpers import *
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
import matplotlib.pyplot as plt
from pprint import pprint

AIMA_TOOLBOX_ROOT="aima-python"
sys.path.append(AIMA_TOOLBOX_ROOT)
from search import *

#from agents import RandomAgent

# agent = RandomAgent(problem_id=1)
# print(agent.solve())

# Setup the parameters for the specific problem (you can change all of these if you want to) 
problem_id = 0        # problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent 
reward_hole = 0.0     # should be less than or equal to 0.0 (you can fine tune this  depending on you RL agent choice)
is_stochastic = True  # should be False for A-star (deterministic search) and True for the RL agent

max_episodes = 1   # you can decide you rerun the problem many times thus generating many episodes... you can learn from them all!
max_iter_per_episode = 1 # you decide how many iterations/actions can be executed per episode

# # Generate the specific problem 
env = LochLomondEnv(problem_id=problem_id, is_stochastic=True, reward_hole=reward_hole)

# # Let's visualize the problem/env
print(env.desc)

# # # Create a representation of the state space for use with AIMA A-star
# # state_space_locations, state_space_actions, state_initial_id, state_goal_id = env2statespace(env)

# Reset the random generator to a known state (for reproducability)
np.random.seed()

# ####
total_rewards = 0
my_list = []
my_list.append(["Run", "Episode", "Iteration", "PrevLocationX", "PrevLocationY", "NewLocationX", "NewLocationY", "Action", "Done", "Reward", "CumulativeReward"])
coordinates = env2statespace(env)[4]

def get_action(action):
  if action == 0:
    return "left"
  elif action == 1:
    return "down"
  elif action == 2:
    return "right"
  elif action == 3:
    return "up"
  return "unkown"

for r in range(5):
  for e in range(max_episodes): # iterate over episodes
    observation = env.reset() # reset the state of the env to the starting state     
    # print("Episode", e)

    for iter in range(max_iter_per_episode):
      #print("Iteration: ", iter)
      #env.render() # for debugging/develeopment you may want to visualize the individual steps by uncommenting this line      
      action = env.action_space.sample() # your agent goes here (the current agent takes random actions)
      prev_location = coordinates[observation]
      observation, reward, done, info = env.step(action) # observe what happends when you take the action

      #print(info)

      my_list.append([r+1, e+1, iter+1, prev_location[0], prev_location[1], coordinates[observation][0], coordinates[observation][1], get_action(action), done, reward, total_rewards])

      if (done and reward == reward_hole): 
        #env.render()     
        #print("We have reached a hole :-( [we can't move so stop trying; just give up]")

        break
      if (done and reward == +1.0):
        #env.render()     
        total_rewards += 1 
        #print("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achived the goal]")
        
        break

print("total rewards: ", total_rewards, " in ", max_episodes)
np.savetxt("foo.csv", my_list, delimiter=",", fmt='%s')
problem_id = 0        # problem_id \in [0:7] generates 8 diffrent problems on which you can train/fine-tune your agent 
reward_hole = 0.0     # should be less than or equal to 0.0 (you can fine tune this  depending on you RL agent choice)
is_stochastic = True  # should be False for A-star (deterministic search) and True for the RL agent

max_episodes = 50   # you can decide you rerun the problem many times thus generating many episodes... you can learn from them all!
max_iter_per_episode = 1000 # you decide how many iterations/actions can be executed per episode

# # Generate the specific problem 
env = LochLomondEnv(problem_id=problem_id, is_stochastic=False, reward_hole=reward_hole)
# Reset the random generator to a known state (for reproducability)
np.random.seed()

# ####
total_rewards = 0
my_list = []
my_list.append(["Run", "Episode", "Iteration", "PrevLocationX", "PrevLocationY", "NewLocationX", "NewLocationY", "Action", "Done", "Reward", "CumulativeReward"])

locations, actions, state_initial_id, state_goal_id, my_map = env2statespace(env)

maze_map = UndirectedGraph(actions)
maze_map.locations = locations

maze_problem = GraphProblem(state_initial_id, state_goal_id, maze_map)

print(type(maze_problem))
#print("state_initial_id", state_initial_id)
#print("state_goal_id", state_goal_id)
print("Initial state: " + maze_problem.initial)
print("Goal state: "    + maze_problem.goal)

print(pprint(maze_map.__dict__))
print(pprint(maze_problem.__dict__))

def my_best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""

    iterations = 0
   
    f = memoize(f, 'f')
    node = Node(problem.initial)

    iterations += 1

    if problem.goal_test(node.state):
      iterations += 1
      return(iterations, node)
    
    frontier = PriorityQueue('min', f)
    frontier.append(node)
   
    iterations += 1    

    explored = set()

    while frontier:
      node = frontier.pop() 
      iterations += 1
      
      if problem.goal_test(node.state):
        iterations += 1
        return(iterations, node)

      explored.add(node.state)
      
      for child in node.expand(problem):
        if child.state not in explored and child not in frontier:
          frontier.append(child)
          iterations += 1

        elif child in frontier:
          incumbent = frontier[child]
          if f(child) < f(incumbent):
            del frontier[incumbent]
            frontier.append(child)
            iterations += 1

      iterations += 1
    #return None

def my_astar_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h') # define the heuristic function
    return my_best_first_graph_search(problem, lambda n: n.path_cost + h(n))
        
all_node_colors=[]
my_astar_search(problem=maze_problem, h=None)
# for r in range(5):
#   for e in range(max_episodes): # iterate over episodes
#     observation = env.reset() # reset the state of the env to the starting state     
#     print("Episode", e)

#     for iter in range(max_iter_per_episode):
#       #print("Iteration: ", iter)
#       #env.render() # for debugging/develeopment you may want to visualize the individual steps by uncommenting this line      
#       action = env.action_space.sample() # your agent goes here (the current agent takes random actions)
#       prev_location = coordinates[observation]
#       observation, reward, done, info = env.step(action) # observe what happends when you take the action

#       my_list.append([r+1, e+1, iter+1, prev_location[0], prev_location[1], coordinates[observation][0], coordinates[observation][1], get_action(action), done, reward, total_rewards])

#       if (done and reward == reward_hole): 
#         #env.render()     
#         print("We have reached a hole :-( [we can't move so stop trying; just give up]")

#         break
#       if (done and reward == +1.0):
#         #env.render()     
#         total_rewards += 1 
#         print("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achived the goal]")
        
#         break

# #np.savetxt('values.csv',np.asarray(lines), delimiter=",", )