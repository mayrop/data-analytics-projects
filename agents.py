import os, sys
import numpy as np
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
from helpers import *
AIMA_TOOLBOX_ROOT="aima-python"
sys.path.append(AIMA_TOOLBOX_ROOT)
from search import *


def map_action(action):
    if action == 0:
        return "left"
    elif action == 1:
        return "down"
    elif action == 2:
        return "right"
    elif action == 3:
        return "up"
    return "unkown"


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

    def action(self):
        return self.env.action_space.sample()

    def solve(self, max_episodes=200, max_iter_per_episode=500):
        self.reset_lines()
        self.reset_rewards()

        for e in range(max_episodes): # iterate over episodes
            observation = self.env.reset() # reset the state of the env to the starting state     
            self.set_episode_seed(e)

            for iter in range(max_iter_per_episode):
                action = self.env.action_space.sample() # your agent goes here (the current agent takes random actions)
                prev_location = self.coordinates[observation]

                observation, reward, done, info = self.env.step(action) # observe what happends when you take the action
                self.lines.append([self.problem_id, self.map_name_base,
                    e+1, iter+1, map_action(action), done, reward, 
                    self.total_rewards, prev_locationv_location[0], prev_location[1], 
                    self.coordinates[observation][0], self.coordinates[observation][1]
                ])

                if (done and reward == self.reward_hole): 
                    break

                if (done and reward == +1.0):
                    self.total_rewards += 1 
                    break

        return self.total_rewards

    def set_episode_seed(self, seed):
        np.random.seed(seed)
        return None

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
        
        locations, actions, state_initial_id, state_goal_id, my_map = self.env_mapping

        self.maze_map = UndirectedGraph(actions)
        self.maze_map.locations = locations
        self.maze_problem = GraphProblem(state_initial_id, state_goal_id, self.maze_map)
        self.locations = locations

    def solve(self, max_episodes=200, max_iter_per_episode=500):
        self.reset_lines()
        self.reset_rewards()

        for e in range(max_episodes): # iterate over episodes
            node = self.perform_a_star_search(problem=self.maze_problem, h=None)
            # cnode = node.parent
            # solution_path.append(cnode)
            # while cnode.state != "S_00_00":    
            #     cnode = cnode.parent  
            #     solution_path.append(cnode)

            #solution = [self.maze_problem.initial] + node.solution()
            #solution  node.solution()
            len_solution = len(solution)
            # print("len_solution", len_solution)
            # print("INITIAL", self.maze_problem.initial)
            # print("solution", solution)

            # ProblemId,Map,Episode,Iteration,Action,Done,Reward,CumulativeReward,PrevLocationX,PrevLocationY,NewLocationX,NewLocationY
            for i in range(1, min(len_solution, max_iter_per_episode)):
                prev = self.locations[solution[i - 1]]
                curr = self.locations[solution[i]]

                action = self.map_action_from_states(x1=prev[0], x2=curr[0], y1=prev[1], y2=curr[1])

                done = solution[i] == self.maze_problem.goal
                reward = 0

                if done:
                    self.total_rewards += 1
                    reward = 1

                self.lines.append([self.problem_id, self.map_name_base, e+1, i, 
                    action, done, reward, self.total_rewards, prev[0], prev[1], 
                    curr[0], curr[1]
                ])

        return self.total_rewards

    def map_action_from_states(self, x1, x2, y1, y2):
        if x2 > x1:
            return "right"
        if x2 < x1:
            return "left"
        if y2 > y1:
            return "down"
        if y2 < y1:
            return "up"
        #TODO ADD EXCEPTOIN HERE
  
    def perform_a_star_search(self, problem, h=None):
        """A* search is best-first graph search with f(n) = g(n)+h(n).
        You need to specify the h function when you call astar_search, or
        else in your Problem subclass."""
        h = memoize(h or problem.h, 'h') # define the heuristic function
        return self.perform_best_first_graph_search(problem, lambda n: n.path_cost + h(n))    

    def perform_best_first_graph_search(self, problem, f):
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
    def __init__(self, problem_id, Rplus=2, Ne=5, gamma=0.99, alpha=None, map_name_base="8x8-base"):
        super(QLearningAgent, self).__init__(problem_id=problem_id, 
                                          reward_hole=-10.0, 
                                          is_stochastic=True,
                                          map_name_base=map_name_base)

        self._agent = QLearningAgentUofG(terminals=self.env.terminals, gamma=gamma,
                                    Ne=Ne, Rplus=Rplus, alpha=alpha,
                                    actlist=list(range(self.env.action_space.n)))


    def solve(self, max_episodes=200, max_iter_per_episode=500):
        self.reset_lines()
        self.reset_rewards()

        for e in range(max_episodes): # iterate over episodes
            observation = self.env.reset() # reset the state of the env to the starting state     

            while True: # TODO - add max iterations
                print("---------------")        
                current_reward = mdp.R(current_state)
                percept = (current_state, current_reward)        
                next_action = agent_program(percept)
                if next_action is None:
                    break
                current_state = take_single_action(mdp, current_state, next_action)
                
                print("percept:", percept)
                print("next_action", next_action)
                print("current_state", current_state)
                print("---------------")
            # means it already performed an action
            for i in range(max_iter_per_episode):
                print(self.env.render())
                percept = (state, reward)
                action = self.learn(percept, done)
                print("Running iteration: ", i)
                print("\tPercept: ", percept)
                print("\tAction: ", map_action(action))

                if action is not None:
                    state, reward, done, info = self.env.step(action)
                    print("\tQ: ", Q)
                    print("\tNsa: ", Nsa)

                    print("\tObservation: ", state)
                    print("\tCoordinates: ", coordinates[state])
                    print("\tReward: ", reward)
                    print("\tDone: ", done)
                    print("\tInfo: ", info)
                else:
                    break

                #print("self.env.action_space", )
                # print(self.actions(self.s))
                # print(

                #     argmax(self.actions(self.s), key=lambda a1: self.f(self.Q[self.s, a1], self.Nsa[self.s, a1]))
                # )

                #observation, reward, done, info = self.env.step(action)
                # print(learning_rate)

            # if not self.lastaction:

            # print(observation)
            # # print(self.env.goal)
            # print(self.env.isd)

        #     for i in range(max_iter_per_episode):
        #         action = self.env.action_space.sample() # your agent goes here (the current agent takes random actions)
        #         prev_location = self.coordinates[observation]

        #         observation, reward, done, info = self.env.step(action) # observe what happends when you take the action
        #         self.lines.append([self.problem_id, self.map_name_base, e+1, iter+1, 
        #             map_action(action), done, reward, self.total_rewards, 
        #             prev_locationv_location[0], prev_location[1], 
        #             self.coordinates[observation][0], self.coordinates[observation][1]
        #         ])

        #         if (done and reward == self.reward_hole): 
        #             break

        #         if (done and reward == +1.0):
        #             self.total_rewards += 1 
        #             break

        # return self.total_rewards
    def learn(self, percept, done):
        Nsa = self.Nsa
        f, gamma, lr = self.f, self.gamma, self.get_learning_rate()
        
        s, action, reward = self.s, self.action, self.reward
        s1, r1 = self.update_percept(percept)
        self.update_nsa()

        if done:
            print("\nI AM DONE!!!...")
            self.Q[s, None] = reward

        if action is not None:
            print("Updating Q...")
            print("\tPrevious S: ", s)
            print("\tPrevious Action: ", action)
            self.Q[s, action] += lr * (reward + gamma * max(self.Q[s1, a1] for a1 in self.actions(s1)) - self.Q[s, action])
            print(self.Q)

        if done:
            print("resetting state reward action")
            self.s = self.reward = self.action = None
        
        if not done:
            self.s, self.reward = s1, r1
            self.action = argmax(self.actions(s1), key=lambda a1: f(self.Q[s1, a1], Nsa[s1, a1]))

        return self.action


    def update_percept(self, percept):
        # todo... change here?
        return percept

    def get_learning_rate(self):
        if self.action is not None and self.Nsa[self.s, self.action] is not None:
            return self.alpha(self.Nsa[self.s, self.action])

        return self.alpha(0)

    def update_nsa(self):
        if self.action is not None:
            self.Nsa[self.s, self.action] += 1

    # def __call__(self, percept):
    #     alpha, gamma, terminals = self.alpha, self.gamma, self.terminals
    #     Q, Nsa = self.Q, self.Nsa
    #     actions_in_state = self.actions_in_state

    #     s, a, r = self.s, self.a, self.r
    #     s1, r1 = self.update_state(percept) # current state and reward;  s' and r'
        
        #print(s)
        #print(a)
        #print(r)
        #print(s1)
        #print(r1)

        ## s1, r1 = self.update_state(percept)
        ## Q, Nsa, s, a, r = self.Q, self.Nsa, self.s, self.a, self.r
        ## alpha, gamma, terminals = self.alpha, self.gamma, self.terminals,
        ## actions_in_state = self.actions_in_state
        ## 
        ## if s in terminals:
        ##     Q[s, None] = r1
        ## if s is not None:
        ##     Nsa[s, a] += 1
        ##     Q[s, a] += alpha(Nsa[s, a]) * (r + gamma * max(Q[s1, a1]
        ##                                    for a1 in actions_in_state(s1)) - Q[s, a])
        ## if s in terminals:
        ##     self.s = self.a = self.r = None
        ## else:
        ##     self.s, self.r = s1, r1
        ##     self.a = argmax(actions_in_state(s1), key=lambda a1: self.f(Q[s1, a1], Nsa[s1, a1]))
        ## return self.a

        # if s in terminals: # if prev state was a terminal state it should be updated to the reward
        #     Q[s, None] = r  

        # if a is not None: # corrected from the book, we check if the last action was none 
        #                   # i.e. no prev state or a terminal state, the book says to check for s
        #     Nsa[s, a] += 1
        #     Q[s, a] += alpha(Nsa[s, a]) * (r + gamma * max(Q[s1, a1] for a1 in actions_in_state(s1)) - Q[s, a])
        
        # # Update for next iteration
        # if s in terminals:
        #     self.s = self.a = self.r = None
        # else:
        #     self.s, self.r = s1, r1
        #     self.a = argmax(actions_in_state(s1), key=lambda a1: self.f(Q[s1, a1], Nsa[s1, a1]))
        
        # return self.a


           # """Execute trial for given agent_program
    #and mdp. mdp should be an instance of subclass
    #of mdp.MDP """

    # def take_single_action(mdp, s, a):
    #     """
    #     Select outcome of taking action a
    #     in state s. Weighted Sampling.
    #     """
    #     x = random.uniform(0, 1)
    #     cumulative_probability = 0.0
    #     for probability_state in mdp.T(s, a):
    #         probability, state = probability_state
    #         cumulative_probability += probability
    #         if x < cumulative_probability:
    #             break
    #     return state

        # current_state = mdp.init
        # while True:
        #     print("---------------")        
        #     current_reward = mdp.R(current_state)
        #     percept = (current_state, current_reward)        
        #     next_action = agent_program(percept)
        #     if next_action is None:
        #         break
        #     current_state = take_single_action(mdp, current_state, next_action)
            
        #     print("percept:", percept)
        #     print("next_action", next_action)
        #     print("current_state", current_state)
        #     print("---------------")