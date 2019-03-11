import os, sys
import numpy as np
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
from helpers import *
AIMA_TOOLBOX_ROOT="aima-python"
sys.path.append(AIMA_TOOLBOX_ROOT)
from search import *
import matplotlib.pyplot as plt
import random

def to_human(action):
    if action == 0:
        return "left"
    elif action == 1:
        return "down"
    elif action == 2:
        return "right"
    elif action == 3:
        return "up"
    return "unkown"


def perform_best_first_graph_search(problem, f):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""

    f = memoize(f, 'f')
    node = Node(problem.initial)

    if problem.goal_test(node.state):
        return(node)
    
    frontier = PriorityQueue('min', f)
    frontier.append(node)
   
    explored = set()
    
    while frontier:
        node = frontier.pop()
      
        if problem.goal_test(node.state):
            return(node)

        explored.add(node.state)
      
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
            elif child in frontier:
                incumbent = frontier[child]
                if f(child) < f(incumbent):
                    del frontier[incumbent]
                    frontier.append(child)
    return None

def perform_a_star_search(problem, h=None):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h') # define the heuristic function
    return perform_best_first_graph_search(problem, lambda n: n.path_cost + h(n))    



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

class QLearningAgentUofG:

    def __init__(self, terminals, gamma, actlist, Ne, Rplus, alpha=None):

        self.gamma = gamma
        self.terminals = terminals
        self.all_act = actlist
        self.Ne = Ne  # iteration limit in exploration function
        self.Rplus = Rplus  # large value to assign before iteration limit
        self.Q = defaultdict(float)
        self.Nsa = defaultdict(float)
        self.s = None
        self.a = None
        self.r = None

        if alpha:
            self.alpha = alpha
        else:
            self.alpha = lambda n: 1./(1+n)  # udacity video

    def f(self, u, n):
        """ Exploration function. Returns fixed Rplus until
        agent has visited state, action a Ne number of times.
        Same as ADP agent in book."""
        if n < self.Ne:
            return self.Rplus
        else:
            return u

    def actions_in_state(self, state):
        """ Return actions possible in given state.
            Useful for max and argmax. """
        if state in self.terminals:
            return [None]
        else:
            return self.all_act

    def __call__(self, percept):
        alpha, gamma, terminals = self.alpha, self.gamma, self.terminals
        Q, Nsa = self.Q, self.Nsa
        actions_in_state = self.actions_in_state

        s, a, r = self.s, self.a, self.r
        s1, r1 = self.update_state(percept) # current state and reward;  s' and r'
        
        if s in terminals: # if prev state was a terminal state it should be updated to the reward
            Q[s, None] = r  
        
        if a is not None: # corrected from the book, we check if the last action was none i.e. no prev state or a terminal state, the book says to check for s
            Nsa[s, a] += 1
            Q[s, a] += alpha(Nsa[s, a]) * (r + gamma * max(Q[s1, a1] for a1 in actions_in_state(s1)) - Q[s, a])
        
        # Update for next iteration
        if s in terminals:
            self.s = self.a = self.r = None
        else:
            self.s, self.r = s1, r1
            self.a = argmax(actions_in_state(s1), key=lambda a1: self.f(Q[s1, a1], Nsa[s1, a1]))
        
        return self.a

    def update_state(self, percept):
        """To be overridden in most cases. The default case
        assumes the percept to be of type (state, reward)."""
        return percept


class QLearningAgent(MyAbstractAIAgent):
    """ An exploratory Q-learning agent. It avoids having to learn the transition
        model because the Q-value of a state can be related directly to those of
        its neighbors. [Figure 21.8]
    """
    def __init__(self, problem_id, Rplus=10, Ne=30, gamma=0.99, alpha=None, map_name_base="8x8-base"):
        super(QLearningAgent, self).__init__(problem_id=problem_id, 
                                          reward_hole=-1, 
                                          is_stochastic=True,
                                          map_name_base=map_name_base)

        self._agent = QLearningAgentUofG(terminals=self.env.terminals, gamma=gamma,
                                    Ne=Ne, Rplus=Rplus, alpha=alpha,
                                    actlist=list(range(self.env.action_space.n)))


    def solve(self, max_episodes=200, max_iter_per_episode=100):
        self.reset_lines()
        self.reset_rewards()

        random.seed(123)
        for e in range(max_episodes): # iterate over episodes
            current_state = self.env.reset() # reset the state of the env to the starting state     
            current_reward = 0.0
            i = 0

            while True and i < max_iter_per_episode: # TODO - add max iterations
                percept = (current_state, current_reward)        
                next_action = self._agent(percept)

                if next_action is None:
                    break

                current_state, current_reward, done, info = self.env.step(next_action)
                i += 1

    def get_u(self):
        U = defaultdict(lambda: -1000.) 
        

        for state_action, value in self._agent.Q.items():
            state, action = state_action
            if U[state] < value:
                U[state] = value            

        return U

    def graph_utility_estimates_q(self, no_of_iterations=50000):
        states_to_graph = [0, 4, 5, 19, 29, 35, 41, 42, 46, 52, 49, 59, 60, 59, 61, 54, 55, 62, 63]
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

        plt.ylim([-10,1.2])
        plt.legend(loc='lower right')
        plt.xlabel('Iterations')
        plt.ylabel('U')
        plt.show(block=True)