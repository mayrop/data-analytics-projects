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
        self.reset()

    def is_stochastic(self):
        raise NotImplementedError

    def reward_hole(self):
        raise NotImplementedError

    def reset(self):
        self.rewards = 0
        self.failures = 0
        self.eval = []
        self.timeouts = 0

    def solve(self, episodes=10000, iterations=1000, seed=None, gamma=0.95):
        self.train(episodes=episodes, iterations=iterations)
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

                    # break the cycle
                    break;

            if not done:
                self.timeouts += 1

            self.eval.append([self.problem_id, e, i, to_human(action), 
                        int(reward), self.rewards, self.failures, self.timeouts])
                    

    def action(self, i):
        raise NotImplementedError

    def train(self):
        raise NotImplementedError        

    def env(self):
        return self.env

    def set_episode_seed(self, episode, seed=None):
        # by default no seed for abstract agent
        return None

    def alias(self):
        return 'out_{}_{}_{}_'.format(self.name(), self.problem_id, 
                                     self.map_name_base)

    def write_eval_files(self):
        def data_for_file(name):
            if name == 'policy':
                return policy_to_list(self.policy)
            if name == 'policy_arrows':
                return policy_to_arrows(self.policy, self.env.ncol, self.env.ncol)
            if name == 'u':
                return u_to_list(self.U)
            if name == 'eval':
                return self.eval
            if name == 'q':
                return self.Q  
            if name == 'train':
                return self._train                   
            if name == 'graphs':
                return self.graphs             

            return []

        for file in self.files():
            print("Writing file...: ", file)
            if file == 'graphs':
                filename = 'out/{}_{}.json'.format(self.alias(), file)
                with open(filename, 'w') as outfile:
                    json.dump(data_for_file(file), outfile)
            elif file == 'policy_arrows':
                filename = 'out/{}_{}.txt'.format(self.alias(), file)
                data = data_for_file(file)
                np.savetxt(filename, data, delimiter="\t", fmt='%s') 
            else:
                filename = 'out/{}_{}.csv'.format(self.alias(), file)
                data = [self.header(file)] + data_for_file(file)
                np.savetxt(filename, data, delimiter=",", fmt='%s') 

    def header(self, key):
        headers = {
            'eval': [
                'id', 'episode', 'iteration', 'action',
                'reward', 'rewards', 'failures', 'timeouts'
            ],
            'policy': [
                'x', 'y', 'action'
            ],
            'u': [
                'x', 'y', 'u'
            ],
            'train': [
                'id', 'episode', 'iteration', 'reward', 
                'rewards', 'failures', 'timeouts'
            ],
            'graphs': [
                'x', 'y', 'value'
            ],
            'q': [
                'position', 'x', 'y', 'action', 'action_friendly', 'value'
            ]
        }

        if key in headers:
            return headers[key]

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

    def set_episode_seed(self, episode, seed=None):
        if seed is not None:
            self.env.seed(episode)
            self.env.action_space.seed(episode)

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
        return self.policy[pos_to_coord(position, self.env.ncol)]

    def train(self):
        # locations, actions, state_initial_id, state_goal_id, my_map
        # state_space_locations, state_space_actions, state_initial_id, state_goal_id, states_indexes
        self.env_mapping = env2statespace(self.env)
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

        self.policy = {}

        for i in range(1, len(solution)):
            prev = graph.locations[solution[i - 1]]
            curr = graph.locations[solution[i]]
            action = map_from_states(x1=prev[0], x2=curr[0], 
                                     y1=prev[1], y2=curr[1])

            self.policy[(prev[0], prev[1])] = action

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
        self.policy = policy_iteration(mdp)
        self.U = value_iteration(mdp, epsilon=0.000000000001)

    def files(self):
        return ['policy', 'u']

    def name(self):
        return 'passive'

        
################################
################################

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

        return -0.05

    def files(self):
        return ['eval', 'u', 'policy', 'policy_arrows', 'q', 'graphs', 'train']        

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
                               rewards, failures, timeouts])

            if e % 100 == 0:
                print("Episode", e)
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

    def init_trianing(self, mdp, alpha, Ne, Rplus):
        self.gamma = mdp.gamma
        self.terminals = mdp.terminals
        self.all_act = mdp.actlist
        self.Ne = Ne  # iteration limit in exploration function
        self.Rplus = Rplus  # large value to assign before iteration limit
        self.training_Q = defaultdict(float)
        self.Nsa = defaultdict(float)
        self.s = None
        self.a = None
        self.r = None

        if alpha:
            self.alpha = alpha
        else:
            self.alpha = lambda n: 1./(1+n)  # udacity video

    def training_actions(self, state):
        """ Return actions possible in given state.
            Useful for max and argmax. """
        if state in self.terminals:
            return [None]
        else:
            return self.all_act

    def f(self, u, n, a, noise):       
        """ Exploration function."""

        if self.map_name_base == '4x4-base':
            if n < self.Ne:
                return self.Rplus

            return u

        # for 8x8 grid
        return u + noise

    def best_action(self, new_state, new_reward, episode):
        noise = np.random.random((1, 4)) / (episode)
        alpha, gamma, terminals = self.alpha, self.gamma, self.terminals
        Q, Nsa = self.training_Q, self.Nsa
        actions = self.training_actions
        s, a, r = self.s, self.a, self.r

        if a is not None:
            Nsa[s, a] += 1
            Q[s, a] += alpha(Nsa[s, a]) * (r + gamma * max(Q[new_state, a1] for a1 in actions(new_state)) - Q[s, a])

        if new_state in terminals:
            self.training_Q[new_state, None] = new_reward
            self.s = self.a = self.r = None
        else:  
            self.s, self.r = new_state, new_reward            
            self.a = argmax(self.all_act, key=lambda a1: self.f(Q[new_state, a1], 
                                                                Nsa[s, a1], a1, noise[0][a1]))
            if random.uniform(0, 1) < 0.05:
                self.a = random.randint(0, len(self.all_act)-1)
                #epsilon -= 10**-3

        return self.a