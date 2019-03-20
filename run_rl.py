from helpers import *
from uofgsocsai import LochLomondEnv
import numpy as np
import pandas as pd
import sys
import itertools  
import operator  
from search import *
from utils import print_table

class QLearningAgentUofG:
    """ TODO
    
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

    def f(self, u, n, noise):       
        """ Exploration function. """
        # TODO ---- change comment
        # if n < self.Ne:
        #    return self.Rplus

        return u + noise

    def actions_in_state(self, state):
        """ Return actions possible in given state.
            Useful for max and argmax. """
        if state in self.terminals:
            return [None]
        else:
            return self.all_act

    def __call__(self, new_state, new_reward, episode):
        alpha, gamma, terminals = self.alpha, self.gamma, self.terminals
        Q, Nsa = self.Q, self.Nsa
        s, a, r = self.s, self.a, self.r

        if a is not None:
            Nsa[s, a] += 1
            # Q[s, a] += alpha * (r + gamma * max(Q[new_state, a1] for a1 in self.actions_in_state(new_state)) - Q[s, a])
            Q[s, a] += alpha(Nsa[s, a]) * (r + gamma * max(Q[new_state, a1] for a1 in self.actions_in_state(new_state)) - Q[s, a])

        if new_state in terminals:
            self.Q[new_state, None] = new_reward
            self.s = self.a = self.r = None
        else:
            noise = np.random.random((1, 4)) / (episode)

            self.s, self.r = new_state, new_reward            
            #self.a = argmax(self.actions_in_state(new_state), key=lambda a1: self.f(Q[new_state, a1], Nsa[s, a1]))
            self.a = argmax(self.actions_in_state(new_state), key=lambda a1: self.f(Q[new_state, a1], Nsa[s, a1], noise[0][a1]))

            # if random.uniform(0, 1) < 1/(episode+1):
            #     self.a = random.randint(0, len(self.all_act)-1)
            if random.uniform(0, 1) < 0.075:
                self.a = random.randint(0, len(self.all_act)-1)            

        return self.a

    def update_state(self, percept):
        """To be overridden in most cases. The default case
        assumes the percept to be of type (state, reward)."""
        return percept


def rl_agent(problem_id):

    # should be less than or equal to 0.0 (you can fine tune this  depending on you RL agent choice)
    reward_hole = -0.05

    # you can decide you rerun the problem many times thus generating many episodes... you can learn from them all!
    max_episodes = 10000   

    # you decide how many iterations/actions can be executed per episode
    max_iter_per_episode = 1000 

    # TODO - add doc on this
    lost_episodes = 0

    actions = []
    results = []

    env = LochLomondEnv(problem_id=problem_id, is_stochastic=True, reward_hole=reward_hole)
    all_act = list(range(env.action_space.n))

    #q_agent = QLearningAgentUofG(terminals=get_terminals(env), all_act=all_act, alpha=0.8, gamma=0.8, Rplus=2, Ne=5)
    q_agent = QLearningAgentUofG(terminals=get_terminals(env), all_act=all_act, alpha=lambda n: 60./(59+n), gamma=0.95, Rplus=2, Ne=5)

    print('Running Q Learning Agent')
    
    for e in range(max_episodes): # iterate over episodes

        state = env.reset() # reset the state of the env to the starting state     
        reward = 0
    
        for iter in range(max_iter_per_episode):
            action = q_agent(state, reward, e+1)
            
            if action is not None:
                state, reward, done, info = env.step(action) 
            
            if done:
                q_agent(state, reward, e+1)

                # Check if we are done and monitor rewards etc...
                if (reward == reward_hole): 
                    lost_episodes += 1
            
                break

        results.append([e, iter, int(reward), lost_episodes])

    # Computing policy
    policy = {}

    for state_action, value in list(q_agent.Q.items()):
        state, action = state_action
        policy[state] = argmax(q_agent.actions_in_state(state), key=lambda a: q_agent.Q[state, a])

    print('Policy: ')
    print_table(to_arrows(policy, 8, 8))

    # Save the results to a file
    np.save('out_rl_{}.npy'.format(problem_id), np.array(results))
    # Save the results to a CSV file
    np.savetxt('out_rl_{}.csv'.format(problem_id), np.array(results), 
               header="episode,iterations,reward,lost_episodes", delimiter=",", fmt='%s')

    np.savetxt('out_rl_{}_policy.txt'.format(problem_id), to_arrows(policy, 8, 8), delimiter="\t", fmt='%s')

    # Adding plot for all episodes
    columns = ['episode', 'iterations', 'reward', 'lost_episodes']

    dataframe = pd.DataFrame(data=np.array(results), index=np.array(results)[0:,0], columns=columns)
    dataframe['cumulative_rewards'] = list(itertools.accumulate(dataframe['reward'], operator.add))
    dataframe['mean_rewards'] = dataframe.apply(lambda x: mean_rewards(x), axis=1)

    x = range(1, len(dataframe) + 1)
    y = dataframe['mean_rewards']
    
    title = 'My title: problem id {}'.format(problem_id)
    subtitle = 'My subtitle {}'.format(problem_id)
    labels = ['Episodes', 'Mean Reward']
    
    add_plot(x, y, 'out_rl_{}.png'.format(problem_id), title, subtitle, labels)

    # Adding plot for the last 1000 episodes
    dataframe_ac = pd.DataFrame(data=np.array(results)[range(max_episodes-1000, max_episodes),:], columns=columns)
    dataframe_ac['episode'] = range(1000)
    dataframe_ac['cumulative_rewards'] = list(itertools.accumulate(dataframe_ac['reward'], operator.add))
    dataframe_ac['mean_rewards'] = dataframe_ac.apply(lambda x: mean_rewards(x), axis=1)

    x = range(1, len(dataframe_ac) + 1)
    y = dataframe_ac['mean_rewards']

    title = 'My title: problem id {}'.format(problem_id)
    subtitle = 'My subtitle {}'.format(problem_id)
    labels = ['Episodes', 'Mean Reward']
    
    add_plot(x, y, 'out_rl_{}_after_convergence.png'.format(problem_id), title, subtitle, labels)

    # Getting numerical summaries
    stats = {
        'all_stats': pd.DataFrame(dataframe.describe(include = 'all')),
        'successes_stats': dataframe[(dataframe.reward == 1)].describe(include = 'all'),
        'failures_stats': dataframe[(dataframe.reward != 1)].describe(include = 'all'),
        'all_stats_ac': pd.DataFrame(dataframe_ac.describe(include = 'all')),
        'successes_stats_ac': dataframe_ac[(dataframe_ac.reward == 1)].describe(include = 'all'),
        'failures_stats_ac': dataframe_ac[(dataframe_ac.reward != 1)].describe(include = 'all')                
    }

    print('Printing stats for rl agent...')
    for stat, values in stats.items():
        print(stat)
        print(values)
        print("\n\n")

    return dataframe, stats

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("usage: run_rl.py <problem_id>")
        exit()
    
    if not (0 <= int(sys.argv[1]) <= 7):
        raise ValueError("Problem ID must be 0 <= problem_id <= 7")        

    rl_agent(int(sys.argv[1]))

