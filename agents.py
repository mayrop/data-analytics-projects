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
import matplotlib.pyplot as plt
import random
sys.path.append("aima")
from search import *
from mdp import policy_iteration
from mdp import value_iteration
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
from helpers import *

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
        self.reset()

    def is_stochastic(self):
        raise NotImplementedError

    def reward_hole(self):
        raise NotImplementedError

    def reset(self):
        self.rewards = 0
        self.failures = 0
        self.eval = []

    def solve(self, episodes=200, iterations=200, reset=True, seed=False):

        for e in range(1, episodes + 1): # iterate over episodes
            state = self.env.reset()
            self.set_episode_seed(e, seed)

            for i in range(1, iterations+1):
                action = self.action(i-1) 
                new_state, reward, done, info = self.env.step(action) 

                if done:
                    if reward == self.reward_hole():
                        self.failures += 1
                    else:
                        self.rewards += int(reward)

                self.rewards += int(reward)
                self.eval.append([self.problem_id, e, i, to_human(action), 
                    done, int(reward), self.failures, self.rewards,
                    state, new_state])

                state = new_state

                if done:
                    # break the cycle
                    break;

    def action(self, i):
        raise NotImplementedError

    def env(self):
        return self.env

    def set_episode_seed(self, seed, force=False):
        # by default no seed for abstract agent
        return None

    def policy(self):
        raise NotImplementedError

    def alias(self):
        return 'out_{}_{}_{}_'.format(self.name(), self.problem_id, 
                                     self.map_name_base)

    def write_eval_files(self):
        def data_for_file(name):
            if name == 'policy':
                return policy_to_list(self.policy())
            if name == 'u':
                return u_to_list(self.u())
            if name == 'eval':
                return self.eval

            return []

        for file in self.files():
            filename = '{}_{}.csv'.format(self.alias(), file)
            data = [self.header(file)] + data_for_file(file)
            np.savetxt(filename, data, delimiter=",", fmt='%s') 

    def header(self, index):
        if index == 'eval':
            return ['id', 'episode', 'iteration', 'action', 'done', 
                'reward', 'failures', 'rewards', 'old_state', 'new_state']

        if index == 'policy':
            return ['x', 'y', 'action']

        if index == 'u':
            return ['x', 'y', 'u'] 

        return []           

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

    def set_episode_seed(self, seed, force=False):
        if force:
            self.env.seed(seed)
            self.env.action_space.seed(seed)

    def action(self, i):
        return self.env.action_space.sample()

    def train(self):
        return

    def files(self):
        return ['eval']

    def name(self):
        return 'random'

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
        self._policy = {}

        for i in range(1, len(solution)):
            prev = graph.locations[solution[i - 1]]
            curr = graph.locations[solution[i]]
            action = map_from_states(x1=prev[0], x2=curr[0], 
                                     y1=prev[1], y2=curr[1])

            self._actions.append(action)
            self._policy[(prev[0], prev[1])] = action

    def policy(self):
        try:
            return(self._policy)
        except AttributeError:
            self.train()

        return(self._policy)

    def files(self):
        return ['eval', 'policy']

    def name(self):
        return 'simple'


################################
################################

class UofGPassiveAgent(MyAbstractAIAgent):
    """ An exploratory Q-learning agent. It avoids having to learn the transition
        model because the Q-value of a state can be related directly to those of
        its neighbors. [Figure 21.8]
    """
    def is_stochastic(self):
        return True

    def reward_hole(self):
        return -0.04

    def solve(self, episodes=200, iterations=200, reset=True, seed=False):
        mdp = EnvMDP(self.env)
        self._policy = policy_iteration(mdp)
        self._u = value_iteration(mdp, epsilon=0.000000000001)

    def files(self):
        return ['policy', 'u']

    def policy(self):
        return self._policy

    def u(self):
        return self._u

    def name(self):
        return 'passive'

        
################################
################################

class UofGQLearningAgent(MyAbstractAIAgent):
    """ An exploratory Q-learning agent. It avoids having to learn the transition
        model because the Q-value of a state can be related directly to those of
        its neighbors. [Figure 21.8]
    """
    def is_stochastic(self):
        return True

    def reward_hole(self):
        return -0.04

    #def train(self):
        