from helpers import *
from uofgsocsai import LochLomondEnv
import numpy as np
import pandas as pd
import sys
import itertools  
import operator  
from search import *

class QLearningAgentUofG:
    """ An exploratory Q-learning agent. It avoids having to learn the transition
        model because the Q-value of a state can be related directly to those of
        its neighbors. [Figure 21.8]

    # TODO _CHECK!!!
    import sys
    from mdp import sequential_decision_environment
    north = (0, 1)
    south = (0,-1)
    west = (-1, 0)
    east = (1, 0)
    policy = {(0, 2): east, (1, 2): east, (2, 2): east, (3, 2): None, (0, 1): north, (2, 1): north, (3, 1): None, (0, 0): north, (1, 0): west, (2, 0): west, (3, 0): west,}
    q_agent = QLearningAgent(sequential_decision_environment, Ne=5, Rplus=2, alpha=lambda n: 60./(59+n))
    for i in range(200):
        run_single_trial(q_agent,sequential_decision_environment)
    
    """
    def __init__(self, terminals, all_act, alpha, gamma, Ne, Rplus):

        self.gamma = gamma
        self.alpha = alpha
        self.terminals = terminals
        self.all_act = all_act
        self.Q = defaultdict(float)
        self.Nsa = defaultdict(float)        
        self.s = None
        self.a = None
        self.r = None
        self.Ne = Ne
        self.Rplus = Rplus

    def f(self, u, n):       
        """ Exploration function. Returns fixed Rplus until
        agent has visited state, action a Ne number of times.
        Same as ADP agent in book."""
        # TODO ---- change comment
        # if n < self.Ne:
        #    return self.Rplus

        return u

    def actions_in_state(self, state):
        """ Return actions possible in given state.
            Useful for max and argmax. """
        if state in self.terminals:
            return [None]
        else:
            return self.all_act

    def best_action(self, new_state, new_reward, episode):
        alpha, gamma, terminals = self.alpha, self.gamma, self.terminals
        Q, Nsa = self.Q, self.Nsa
        s, a, r = self.s, self.a, self.r

        if a is not None:
            Nsa[s, a] += 1
            Q[s, a] += alpha * (r + gamma * max(Q[new_state, a1] for a1 in self.actions_in_state(new_state)) - Q[s, a])

        if new_state in terminals:
            self.Q[new_state, None] = new_reward
            self.s = self.a = self.r = None
        else:
            self.s, self.r = new_state, new_reward            
            self.a = argmax(self.actions_in_state(new_state), key=lambda a1: self.f(Q[new_state, a1], Nsa[s, a1]))
            
            if random.uniform(0, 1) < 1/(episode+1):
                self.a = random.randint(0, len(self.all_act)-1)


        # print(Q)

        return self.a

    def update_state(self, percept):
        """To be overridden in most cases. The default case
        assumes the percept to be of type (state, reward)."""
        return percept


def rl_agent(problem_id):

    # should be less than or equal to 0.0 (you can fine tune this  depending on you RL agent choice)
    reward_hole = -0.01

    # you can decide you rerun the problem many times thus generating many episodes... you can learn from them all!
    max_episodes = 10000   

    # you decide how many iterations/actions can be executed per episode
    max_iter_per_episode = 100 

    # TODO - add doc on this
    lost_episodes = 0

    actions = []
    results = []
    failures = 0
    rewards = 0

    env = LochLomondEnv(problem_id=problem_id, is_stochastic=True, reward_hole=reward_hole)
    all_act = list(range(env.action_space.n))

    q_agent = QLearningAgentUofG(terminals=get_terminals(env), all_act=all_act, alpha=0.8, gamma=0.8, Rplus=2, Ne=5)
    #q_agent = QLearningAgentUofG(terminals=get_terminals(env), alpha=lambda n: 60./(59+n), gamma=0.95, Rplus=2, Ne=5)
    
    for e in range(max_episodes): # iterate over episodes

        state = env.reset() # reset the state of the env to the starting state     
        reward = 0
    
        for iter in range(max_iter_per_episode):
            action = q_agent.best_action(state, reward, e+1)
            
            if action is not None:
                state, reward, done, info = env.step(action) 
            
            if done:
                q_agent.best_action(state, reward, e+1)

                if reward == 1.0:
                    rewards += int(reward)
                else:
                    failures += 1                    
                
                break

        results.append([e, iter, int(reward), lost_episodes])

    columns = ['episode', 'iteration', 'reward', 'lost_episodes']
    
    dataframe = pd.DataFrame(data=np.array(results), index=np.array(results)[0:,0], columns=columns)
    dataframe['cumulative_rewards'] = list(itertools.accumulate(dataframe['reward'], operator.add))
    dataframe['mean_rewards'] = dataframe.apply(lambda x: mean_rewards(x), axis=1)

    x = range(1, len(dataframe) + 1)
    y = dataframe['mean_rewards']
    
    add_plot(x, y, 'out_rl_{}_mean_reward.png'.format(problem_id))

    return dataframe

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: run_simple.py <problem_id>")
        exit()
    
    if not (0 <= int(sys.argv[1]) <= 7):
        raise ValueError("Problem ID must be 0 <= problem_id <= 7")        

    rl_agent(int(sys.argv[1]))

