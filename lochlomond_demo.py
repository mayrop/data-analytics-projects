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

lines = []
for i in range(8):
  for base_map in [4, 8]:
    if i >= base_map:
      continue

    map_name_base = str(base_map) + "x" + str(base_map) + "-base"
    agent = RandomAgent(problem_id=i, map_name_base=map_name_base)
    agent.solve(max_episodes=1000, max_iter_per_episode=100)
    for line in agent.lines:
      lines.append(line)

    print("total rewards: ", agent.total_rewards)
    # np.savetxt("data/random_agent_problem_id_" + str(i) + "_map_" + str(base_map) + ".csv", agent.lines, delimiter=",", fmt='%s')  
