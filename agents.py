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
                return QLearningAgentUofG.u_to_list(self.u())
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

class ReinforcementLearningAgent(MyAbstractAIAgent):
    """ An exploratory Q-learning agent. It avoids having to learn the transition
        model because the Q-value of a state can be related directly to those of
        its neighbors. [Figure 21.8]
    """
    def is_stochastic(self):
        return True

    def reward_hole(self):
        if self.map_name_base == '4x4-base':
            #print("REWARD HOLE!!!")
            return -0.7

        # this should be modified for diff env
        return -0.15

    def train(self):
        mdp = EnvMDP(self.env)
        q_agent = QLearningAgentUofG(mdp, Ne=5, Rplus=2, alpha=lambda n: 60./(59+n))

        states = mdp.states
        self.graphs = {coord_to_pos(state[0], state[1], mdp.cols):[] for state in states}

        episodes = 100000
        iterations = 1000
        rewards = 0
        failures = 0
        self._train = []
        cols = self.env.ncol
        step = self.env.step

        positions = {coord_to_pos(state[0], state[1], mdp.cols):state for state in states}
        coordinates = {state:coord_to_pos(state[0], state[1], mdp.cols) for state in states}

        for e in range(1, episodes + 1):
            state = self.env.reset()
            reward = 0
            print("EPISODEEEE", e)
            for i in range(iterations):
                action = q_agent.best_action(positions[state], reward, e)
                
                if action is not None:
                    state, reward, done, info = step(action) 
                
                if done:
                    q_agent.best_action(positions[state], reward, e)

                    if reward == 1.0:
                        rewards += int(reward)
                    else:
                        failures += 1                    
                    
                    break
            
            self._train.append([self.problem_id, e, i, int(reward), failures, rewards])

            if e % 100 == 0:
            #    if self.map_name_base == '4x4-base':
                for state in states:
                    q_agent.update_u()
                    index = coordinates[state]
                    #print(state, index)
                    self.graphs[index].append([e, q_agent.U[state]])
    

        print(q_agent.U)
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


class QLearningAgentUofG(QLearningAgent):
    """ An exploratory Q-learning agent. It avoids having to learn the transition
        model because the Q-value of a state can be related directly to those of
        its neighbors. [Figure 21.8]

        Source of this class: 
        - Labs from Artifial Intelligence (H), University of Glasgow class 2019
        - Modified for convenience
    """

    def f(self, u, n, a, noise):       
        """ Exploration function."""
        # print(n)
        #if n < self.Ne:
        #    return self.Rplus + noise
        #if n < self.Ne:
        #    return self.Rplus

        #print("retturning ", u, a)
        return u + noise

    def best_action(self, new_state, new_reward, episode):
        #print("best action: ", new_reward, new_state)
        noise = np.random.random((1, 4)) / (episode)
        # / (episode**2.)
        #print(noise[0])
        alpha, gamma, terminals = self.alpha, self.gamma, self.terminals
        Q, Nsa = self.Q, self.Nsa
        actions = self.actions_in_state
        s, a, r = self.s, self.a, self.r

        if a is not None: # corrected from the book, we check if the last action was none i.e. no prev state or a terminal state, the book says to check for s
            Nsa[s, a] += 1
            #print("updating ", s, a, " ---- ", r)
            #print("Nsa[s, a]", Nsa[s, a])
            Q[s, a] += alpha(Nsa[s, a]) * (r + 0.96 * max(Q[new_state, a1] for a1 in actions(new_state)) - Q[s, a])

        if new_state in terminals:
            self.Q[new_state, None] = new_reward
            self.s = self.a = self.r = None
        else:  
            self.s, self.r = new_state, new_reward            
            self.a = argmax(self.all_act, key=lambda a1: self.f(Q[new_state, a1], Nsa[s, a1], a1, noise[0][a1]))
            if random.uniform(0, 1) < 0.075:
                self.a = random.randint(0, len(self.all_act)-1)
                #epsilon -= 10**-3

        return self.a

    def update_u(self):
        self.U = QLearningAgentUofG.q_to_u(self.Q)

    @staticmethod
    def u_to_list(U):
        return [[int(x), int(y), U[(x, y)]] for x, y in U]

    @staticmethod
    def q_to_u(Q):
        """ Source of this function: 
            - Labs from Artifial Intelligence (H), University of Glasgow class 2019
        """
        U = defaultdict(lambda: -1000.) 
        
        for state_action, value in Q.items():
            state, action = state_action
            if U[state] < value:
                U[state] = value     

        return U