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
from agents import RandomAgent, SimpleAgent

# lines = []
# for i in range(8):
#   for base_map in [4, 8]:
#     if i >= base_map:
#       continue

#     map_name_base = str(base_map) + "x" + str(base_map) + "-base"
#     agent = RandomAgent(problem_id=i, max_episodes=1000, max_iter_per_episode=100, map_name_base=map_name_base)
#     agent.solve()
#     for line in agent.lines:
#       lines.append(line)

#     print("total rewards: ", agent.total_rewards, " in ", agent.max_episodes)
#     np.savetxt("data/random_agent_problem_id_" + str(i) + "_map_" + str(base_map) + ".csv", agent.lines, delimiter=",", fmt='%s')  

# np.savetxt("data/random_all.csv", lines, delimiter=",", fmt='%s')  

lines = []
for i in range(8):
  for base_map in [4, 8]:
    if i >= base_map:
      continue

    map_name_base = str(base_map) + "x" + str(base_map) + "-base"
    agent = SimpleAgent(problem_id=i, max_episodes=1000, max_iter_per_episode=100, map_name_base=map_name_base)
    agent.solve()
    for line in agent.lines:
      lines.append(line)

    print("total rewards: ", agent.total_rewards, " in ", agent.max_episodes)
    np.savetxt("data/simple_agent_problem_id_" + str(i) + "_map_" + str(base_map) + ".csv", agent.lines, delimiter=",", fmt='%s')  

np.savetxt("data/simple_all.csv", lines, delimiter=",", fmt='%s')  

#locations, actions, state_initial_id, state_goal_id, my_map = env2statespace(env)

#maze_map = UndirectedGraph(actions)
#maze_map.locations = locations

#maze_problem = GraphProblem(state_initial_id, state_goal_id, maze_map)

#print(type(maze_problem))
#print("state_initial_id", state_initial_id)
#print("state_goal_id", state_goal_id)
#print("Initial state: " + maze_problem.initial)
#print("Goal state: "    + maze_problem.goal)

#print(pprint(maze_map.__dict__))
#print(pprint(maze_problem.__dict__))

# def my_best_first_graph_search(problem, f):
#     """Search the nodes with the lowest f scores first.
#     You specify the function f(node) that you want to minimize; for example,
#     if f is a heuristic estimate to the goal, then we have greedy best
#     first search; if f is node.depth then we have breadth-first search.
#     There is a subtlety: the line "f = memoize(f, 'f')" means that the f
#     values will be cached on the nodes as they are computed. So after doing
#     a best first search you can examine the f values of the path returned."""

#     iterations = 0
   
#     f = memoize(f, 'f')
#     node = Node(problem.initial)

#     iterations += 1

#     if problem.goal_test(node.state):
#       iterations += 1
#       return(iterations, node)
    
#     frontier = PriorityQueue('min', f)
#     frontier.append(node)
   
#     iterations += 1    

#     explored = set()

#     while frontier:
#       node = frontier.pop() 
#       iterations += 1
      
#       if problem.goal_test(node.state):
#         iterations += 1
#         return(iterations, node)

#       explored.add(node.state)
      
#       for child in node.expand(problem):
#         if child.state not in explored and child not in frontier:
#           frontier.append(child)
#           iterations += 1

#         elif child in frontier:
#           incumbent = frontier[child]
#           if f(child) < f(incumbent):
#             del frontier[incumbent]
#             frontier.append(child)
#             iterations += 1

#       iterations += 1
#     #return None

# def my_astar_search(problem, h=None):
#     """A* search is best-first graph search with f(n) = g(n)+h(n).
#     You need to specify the h function when you call astar_search, or
#     else in your Problem subclass."""
#     h = memoize(h or problem.h, 'h') # define the heuristic function
#     return my_best_first_graph_search(problem, lambda n: n.path_cost + h(n))
        
# my_astar_search(problem=maze_problem, h=None)
