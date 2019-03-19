"""
    University of Glasgow 
    Artificial Intelligence 2018-2019
    Assessed Exercise

    Basic demo code for the CUSTOM Open AI Gym problem used in AI (H) '18-'19
"""
import copy
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import random
sys.path.append("aima")
from helpers import *
from mdp import policy_iteration
from mdp import value_iteration
from search import *
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
import json

class MyAbstractAIAgent():

class ReinforcementLearningAgent(MyAbstractAIAgent):
    """ An exploratory Q-learning agent. It avoids having to learn the transition
        model because the Q-value of a state can be related directly to those of
        its neighbors. [Figure 21.8]
    """
    def is_stochastic(self):
        return True

    def reward_hole(self):
        if self.map_name_base == '4x4-base':
            return -0.7

        return -0.08

    def files(self):
        return ['evaluation', 'u', 'policy_arrows', 'q', 'train']        

    def action(self, position):
        return self.policy[pos_to_coord(position, self.env.ncol)]

    def name(self):
        return 'rl'

    def train(self, episodes=10000, iterations=1000, gamma=0.95):
        mdp = EnvMDP(self.env, gamma=gamma)
        states = mdp.states
        rewards = 0
        failures = 0
        timeouts = 0
        cols = self.env.ncol
        step = self.env.step

        self.init_trianing(mdp, Ne=5, Rplus=2, alpha=lambda n: 60./(59+n))
        self.graphs = {coord_to_pos(state[0], state[1], mdp.cols):[] for state in states}
        self._train = []

        positions = {coord_to_pos(state[0], state[1], mdp.cols):state for state in states}
        coordinates = {state:coord_to_pos(state[0], state[1], mdp.cols) for state in states}

        for e in range(1, episodes + 1):
            state = self.env.reset()
            reward = 0

            for i in range(iterations):
                action = self.best_action(positions[state], reward, e)
                
                if action is not None:
                    state, reward, done, info = step(action) 
                
                if done:
                    self.best_action(positions[state], reward, e)

                    if reward == 1.0:
                        rewards += int(reward)
                    else:
                        failures += 1                    
                    
                    break

            if not done:
                timeouts += 1
            
            self._train.append([self.problem_id, e, i, int(reward), 
                               rewards, rewards / e, failures, timeouts])

            if e % 100 == 0:
                print("Train Episode", e)
                for state in states:
                    self.update_u()
                    index = coordinates[state]
                    self.graphs[index].append([e, self.U[state]])
    
        # graph_utility_estimates(self.graphs)

        self.update_u()

        self.Q = []
        self.policy = {}
        actions = self.training_actions

        for state_action, value in list(self.training_Q.items()):
            state, action = state_action

            self.Q.append([coordinates[state], state[0], state[1], action, to_human(action), value])
            self.policy[state] = argmax(actions(state), key=lambda a1: self.training_Q[state, a1])

    def update_u(self):
        self.U = q_to_u(self.training_Q)  

 