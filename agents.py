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
import os
import sys
import numpy as np
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
from helpers import *
sys.path.append("aimda")
from search import *
import matplotlib.pyplot as plt
import random

class MyAbstractAIAgent():
    """
    TODO


    """   
    def __init__(self, problem_id, map_name_base="8x8-base"):
        # map_name_base="4x4-base"
        if not (0 <= problem_id <= 7):
            raise ValueError("Problem ID must be 0 <= problem_id <= 7")

        self.env = LochLomondEnv(problem_id=problem_id, 
                                 is_stochastic=self.is_stochastic(), 
                                 reward_hole=self.reward_hole(), 
                                 map_name_base=map_name_base)
        self.map_name_base = map_name_base
        self.problem_id = problem_id
        # state_space_locations, state_space_actions, state_initial_id, state_goal_id, states_indexes
        self.env_mapping = env2statespace(self.env)
        self.coordinates = self.env_mapping[4]
        self.reset_lines()
        self.reset_rewards()

    def header(self):
        return [
            "ProblemId",
            "Map",
            "Episode",
            "Iteration",
            "Action",
            "Done",
            "Reward",
            "CumulativeReward",
            "PrevLocationX",
            "PrevLocationY",
            "NewLocationX",
            "NewLocationY"
        ]

    def is_stochastic(self):
        raise NotImplementedError

    def reward_hole(self):
        raise NotImplementedError

    def solve(self, max_episodes=200, max_iter_per_episode=10):
        self.reset_lines()
        self.reset_rewards()

        for e in range(max_episodes): # iterate over episodes
            observation = self.env.reset()
            done = False
            i = 0
            self.set_episode_seed(e)

            while not done and i < max_iter_per_episode:
                i += 1

                action = self.action(i) # your agent goes here (the current agent takes random actions)
                prev_location = self.coordinates[observation]

                observation, reward, done, info = self.env.step(action) 
                # observe what happends when you take the action
                self.total_rewards += int(reward)

                self.lines.append([self.problem_id, self.map_name_base,
                    e+1, i, to_human(action), done, int(reward), 
                    self.total_rewards, prev_location[0], prev_location[1], 
                    self.coordinates[observation][0], self.coordinates[observation][1]
                ])

        return self.total_rewards

    def action(self, i):
        raise NotImplementedError

    def env(self):
        return self.env

    def reset_lines(self):
        self.lines = [] # reset
        self.lines.append(self.header())

    def reset_rewards(self):
        self.total_rewards = 0

    def set_episode_seed(self, seed):
        # by default no seed for abstract agent
        return None

################################
################################

class RandomAgent(MyAbstractAIAgent):

    """
    TODO


    """  

    def is_stochastic(self):
        return True

    def reward_hole(self):
        return 0.0

    def set_episode_seed(self, seed):
        return
        self.env.seed(seed)
        self.env.action_space.seed(seed)

    def action(self, i):
        return self.env.action_space.sample()

    def train(self):
        return

################################
################################

class SimpleAgent(MyAbstractAIAgent):

    """
    TODO


    """   

    def is_stochastic(self):
        return False

    def reward_hole(self):
        return 0.0

    def action(self, i):
        try:
            self._actions
        except AttributeError:
            self.train()

        return self._actions[i]

    def train(self):
        # locations, actions, state_initial_id, state_goal_id, my_map
        graph = UndirectedGraph(self.env_mapping[1])
        graph.locations = self.env_mapping[0]
        problem = GraphProblem(self.env_mapping[2], 
                               self.env_mapping[3], graph)

        node = astar_search(problem=problem, h=None)
        solution = [self.env_mapping[2]] + node.solution()
        
        def map_from_states(x1, x2, y1, y2):
            if x2 > x1:
                return 2 #"right"
            if x2 < x1:
                return 0 # "left"
            if y2 > y1:
                return 1 # "down"
            if y2 < y1:
                return 3 # "up"

        self._actions = []

        for i in range(1, len(solution)):
            prev = graph.locations[solution[i - 1]]
            curr = graph.locations[solution[i]]
            action = map_from_states(x1=prev[0], x2=curr[0], 
                                     y1=prev[1], y2=curr[1])
            self._actions.append(action)

################################
################################

class MyQLearningAgent(MyAbstractAIAgent):
    """ An exploratory Q-learning agent. It avoids having to learn the transition
        model because the Q-value of a state can be related directly to those of
        its neighbors. [Figure 21.8]
    """
    def is_stochastic(self):
        return True

    def reward_hole(self):
        return -0.04
