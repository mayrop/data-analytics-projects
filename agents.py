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
from rl import run_single_trial
from search import *
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
import json

class MyAbstractAIAgent():
    """
    TODO


    """   
    def __init__(self, problem_id, map_name_base="8x8-base"):
        # map_name_base="4x4-base"
        if not (0 <= problem_id <= 7):
            raise ValueError("Problem ID must be 0 <= problem_id <= 7")

        self.map_name_base = map_name_base
        self.env = LochLomondEnv(problem_id=problem_id, 
                                 is_stochastic=self.is_stochastic(), 
                                 reward_hole=self.reward_hole(), 
                                 map_name_base=map_name_base)

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

    def solve(self, episodes=10000, iterations=200, reset=True, seed=False):
        self.train()
        for e in range(1, episodes + 1): # iterate over episodes
            state = self.env.reset()
            self.set_episode_seed(e, seed)

            for i in range(1, iterations+1):
                action = self.action(state) 
                state, reward, done, info = self.env.step(action) 

                if done:
                    if reward == 1.0:
                        self.rewards += int(reward)
                    else:
                        self.failures += 1

                    self.eval.append([self.problem_id, e, i, to_human(action), 
                        int(reward), self.failures, self.rewards])

                    # break the cycle
                    break;                   
                    

    def action(self, i):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError        

    def env(self):
        return self.env

    def set_episode_seed(self, seed, force=False):
        # by default no seed for abstract agent
        return None

    def policy(self):
        try:
            return(self._policy)
        except AttributeError:
            self.train()

        return self._policy

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
            if name == 'q':
                return self._Q  
            if name == 'train':
                return self._train                   
            if name == 'graphs':
                return self.graphs             

            return []

        for file in self.files():
            if file == 'graphs':
                filename = '{}_{}.json'.format(self.alias(), file)
                with open(filename, 'w') as outfile:
                    json.dump(data_for_file(file), outfile)
            else:
                filename = '{}_{}.csv'.format(self.alias(), file)
                data = [self.header(file)] + data_for_file(file)
                np.savetxt(filename, data, delimiter=",", fmt='%s') 

    def header(self, index):
        if index == 'eval':
            return ['id', 'episode', 'iteration', 'action',
                'reward', 'failures', 'rewards']

        if index == 'policy':
            return ['x', 'y', 'action']

        if index == 'u':
            return ['x', 'y', 'u'] 

        if index == 'train':
            return ['id', 'episode', 'iteration', 'reward', 'failures', 'rewards']

        if index == 'graphs':
            return ['x', 'y', 'value']             

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

    def action(self, position):
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

    def action(self, position):
        return self.policy()[pos_to_coord(position, self.env.ncol)]

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

        self._policy = {}

        for i in range(1, len(solution)):
            prev = graph.locations[solution[i - 1]]
            curr = graph.locations[solution[i]]
            action = map_from_states(x1=prev[0], x2=curr[0], 
                                     y1=prev[1], y2=curr[1])

            self._policy[(prev[0], prev[1])] = action

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
        self._U = value_iteration(mdp, epsilon=0.000000000001)

    def files(self):
        return ['policy', 'u']

    def policy(self):
        return self._policy

    def u(self):
        return self._U

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
        if self.map_name_base == '4x4-base':
            return -0.7

        # this should be modified for diff env
        return -0.05

    def train(self):
        mdp = EnvMDP(self.env)
        q_agent = QLearningAgentUofG(mdp, Ne=5, Rplus=2, alpha=lambda n: 60./(59+n))

        states = mdp.states
        self.graphs = {coord_to_pos(state[0], state[1], mdp.cols):[] for state in states}

        episodes = 100000
        rewards = 0
        failures = 0
        self._train = []

        for e in range(1, episodes + 1):
            print("doing e: ", e)
            reward, i = run_single_trial(q_agent, mdp)   
            print("here...")     
    
            if reward == 1.0:
                rewards += int(reward)
            else:
                failures += 1

            self._train.append([self.problem_id, e, i, int(reward), failures, rewards])

            if i % 100 == 0:
            #    if self.map_name_base == '4x4-base':
                for state in states:
                    q_agent.update_u()
                    index = coord_to_pos(state[0], state[1], mdp.cols)
                    self.graphs[index].append([e, q_agent.U[state]])
    
        graph_utility_estimates(self.graphs)

        q_agent.update_u()

        self._Q = []
        self._policy = {}
        self._U = q_agent.U

        for state_action, value in list(q_agent.Q.items()):
            state, action = state_action
            index = coord_to_pos(state[0], state[1], mdp.cols)
            self._Q.append([index, action, value])
            self._policy[state] = argmax(mdp.actlist, key=lambda a1: q_agent.Q[state, a1])

    def u(self):
        return self._U

    def q(self):
        return self._Q

    def files(self):
        return ['eval', 'u', 'policy', 'q', 'graphs', 'train']        

    def action(self, position):
        return self.policy()[pos_to_coord(position, self.env.ncol)]

    def name(self):
        return 'rl'