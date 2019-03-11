import os, sys
import numpy as np
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
from helpers import *
AIMA_TOOLBOX_ROOT="aima-python"
sys.path.append(AIMA_TOOLBOX_ROOT)
from search import *
from IPython.display import clear_output
import matplotlib.pyplot as plt
import random

class MyAbstractAIAgent():
    """
    TODO


    """   
    def __init__(self, problem_id, reward_hole, is_stochastic, 
                 map_name_base="8x8-base"):
        # map_name_base="4x4-base"
        if not (0 <= problem_id <= 7):
            raise ValueError("Problem ID must be 0 <= problem_id <= 7")

        self.env = LochLomondEnv(problem_id=problem_id, 
                                 is_stochastic=is_stochastic, 
                                 reward_hole=reward_hole, 
                                 map_name_base=map_name_base)
        self.map_name_base = map_name_base
        self.problem_id = problem_id
        self.is_stochastic = is_stochastic
        self.reward_hole = reward_hole    
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

    def set_episode_seed(self, seed):
        return None

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
    def __init__(self, problem_id, map_name_base="8x8-base"):
        super(RandomAgent, self).__init__(problem_id=problem_id,  
                                          reward_hole=0.0, 
                                          is_stochastic=True,
                                          map_name_base=map_name_base)

    def set_episode_seed(self, seed):
        return
        self.env.seed(seed)
        self.env.action_space.seed(seed)

    def action(self, i):
        return self.env.action_space.sample()

################################
################################

class SimpleAgent(MyAbstractAIAgent):

    """
    TODO


    """   
    def __init__(self, problem_id, map_name_base="8x8-base"):
        super(SimpleAgent, self).__init__(problem_id=problem_id, 
                                          reward_hole=0.0, 
                                          is_stochastic=False,
                                          map_name_base=map_name_base)
        self._init_actions(self)

    def action(self, i):
        return self._actions[i]

    def _init_actions(self):
        # locations, actions, state_initial_id, state_goal_id, my_map
        graph = UndirectedGraph(self.env_mapping[1])
        graph.locations = self.env_mapping[0]
        problem = GraphProblem(self.env_mapping[2], 
                               self.env_mapping[3], graph)

        node = perform_a_star_search(problem=problem, h=None)
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


class QLearningAgent(MyAbstractAIAgent):
    """ An exploratory Q-learning agent. It avoids having to learn the transition
        model because the Q-value of a state can be related directly to those of
        its neighbors. [Figure 21.8]
    """
    def __init__(self, problem_id, map_name_base="8x8-base"):
        super(QLearningAgent, self).__init__(problem_id=problem_id, 
                                             reward_hole=-0.15, 
                                             is_stochastic=True,
                                             map_name_base=map_name_base)

        self.terminals = self.env.terminals

    def expected_utility(self, a, s, U):
        return sum([p * U[s1] for (p, s1, r, done) in self.env.P[s][a]])


    def random_action_for_s(self, state):
        if state in self.terminals:
            return None

        return self.env.action_space.sample()

    def policy_iteration(self):
        U = {s: 0 for s in self.env.P}
        #pi = {s: self.env.P[a] for s in self.env.P}
        pi = {s: self.random_action_for_s(s) for s in self.env.P}
  
        while True:
            U = self.policy_evaluation(pi, U)
            unchanged = True
            for s in self.env.P:
                a = argmax(self.env.P[s], key=lambda a: self.expected_utility(a, s, U))
                if a != pi[s]:
                    pi[s] = a
                    unchanged = False
            
            if unchanged:
                return pi

    def T(self, state, action):
        if action is None:
            return [(0.0, state, self.env_mapping[5][state], True)]

        return self.env.P[state][action]

    def policy_evaluation(self, pi, U, k=20):
        """Return an updated utility mapping U from each state in the MDP to its
        utility, using an approximation (modified policy iteration)."""
        R, gamma = self.env_mapping[5], 0.999

        for i in range(k):
            for s in self.env.P:
                U[s] = R[s] + gamma * sum([p * U[s1] for (p, s1, r, done) in self.T(s, pi[s])])

        return U


    def solve(self, max_episodes=3, max_iter_per_episode=100):
        self.reset_lines()
        self.reset_rewards()

        # Hyperparameters
        alpha = 0.1
        gamma = 0.6
        epsilon = 0.1
        Q = np.zeros([self.env.observation_space.n, self.env.action_space.n])

        for i in range(1, max_episodes):
            state = self.env.reset()

            epochs, penalties, reward, = 0, 0, 0
            done = False
            
            while not done:
                if random.uniform(0, 1) < epsilon:
                    action = self.env.action_space.sample() # Explore action space
                else:
                    action = np.argmax(Q[state]) # Exploit learned values

                next_state, reward, done, info = self.env.step(action) 
                
                old_value = Q[state, action]
                next_max = np.max(Q[next_state])
                
                new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
                Q[state, action] = new_value

                if reward == -1:
                    penalties += 1

                state = next_state
                epochs += 1
                
            if i % 100 == 0:
                clear_output(wait=True)
                print(f"Episode: {i}")

        print("Training finished.\n")
        print(Q)

        for row_number, value in enumerate(Q):
            print(row_number, value)

    def get_u(self):
        U = defaultdict(lambda: -1000.) 

        for state_action, value in self._agent.Q.items():
            state, action = state_action
            if U[state] < value:
                U[state] = value            

        return U

    def graph_utility_estimates_q(self, no_of_iterations=5000):
        states_to_graph = [54, 62, 46, 10]
        graphs = {state:[] for state in states_to_graph}

        plt.figure(0)

        for iteration in range(1, no_of_iterations+1):
            self.solve(max_episodes=1)
            
            U = defaultdict(lambda: -1000.)
            for state_action, value in self._agent.Q.items():
                state, action = state_action
                if U[state] < value:
                    U[state] = value            

            for state in states_to_graph:            
                graphs[state].append((iteration, U[state]))
        
        for state, value in graphs.items():
            state_x, state_y = zip(*value)

            plt.plot(state_x, state_y, label=str(state))
        
        print(self._agent.Q.items())

        plt.ylim([-1.2,1.2])
        plt.legend(loc='lower right')
        plt.xlabel('Iterations')
        plt.ylabel('U')
        plt.show(block=True)