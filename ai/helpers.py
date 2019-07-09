import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from uofgsocsai import LochLomondEnv

sys.path.append("aima")
from search import *
from mdp import MDP
from rl import PassiveTDAgent

# ______________________________________________________________________________
# Random

def pos_to_coord(pos, ncol):
    return (pos % ncol, pos // ncol)

def coord_to_pos(x, y, ncol):
    return x + y * ncol    

def to_human(action):
    if action == 0:
        return "left"
    elif action == 1:
        return "down"
    elif action == 2:
        return "right"
    elif action == 3:
        return "up"
    return "."

def u_to_list(U):
    return [[int(x), int(y), U[(x, y)]] for x, y in U]

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

def generate_grids(cols):
    grids = []
    for i in range(cols):
        map_name_base = '{}x{}-base'.format(cols, cols)
        env = LochLomondEnv(problem_id=i, is_stochastic=True, 
                            reward_hole=-0.02, map_name_base=map_name_base)

        env.render()
        grid = EnvMDP.to_decoded(env).reshape(env.nrow * env.ncol)
        grids.append(np.hstack(([i], grid)))
    
    return grids


def parse_args(argv):
    if len(argv) < 1:
        print('usage: run_rl.py <problem_ids> <episodes=10000> <grid=8>')
        exit()

    problem_ids = [abs(int(problem_id.strip())) for problem_id in argv[0].split(',')]
    problem_ids = [problem_id for problem_id in problem_ids if problem_id < 8]
    
    if len(problem_ids) == 0:
        print('please provide problem ids between 0 and 7')
        print('usage: run_rl.py <problem_ids> <episodes=10000> <grid=8>')
        exit()

    episodes = 10000
    grid = '8x8-base'

    if len(argv) > 1 and str.isdigit(argv[1]):
        if int(argv[1]) < 2000:
            print('please a number of episodes > 2000')
            print('usage: run_rl.py <problem_ids> <episodes=10000> <grid=8>')
            exit()
        episodes = int(argv[1])

    if len(argv) > 2 and str.isdigit(argv[2]):
        if int(argv[2]) not in [4, 8]:
            print('please provide a grid, either 4 or 8')
            print('usage: run_rl.py <problem_ids> <episodes=10000> <grid=8>')
            exit()
        grid = '{}x{}-base'.format(argv[2], argv[2])
        problem_ids = [problem_id for problem_id in problem_ids if problem_id < int(argv[2])]

    return problem_ids, episodes, grid


def plot_eval(agents, rows, labels, filename, title, subtitle, training=True):
    random_agent_eval = np.array(agents['random'].eval)
    random_x = pd.to_numeric(random_agent_eval[:,1])
    random_y = pd.to_numeric(random_agent_eval[:,6])

    simple_agent_eval = np.array(agents['simple'].eval)
    simple_x = pd.to_numeric(simple_agent_eval[:,1])
    simple_y = pd.to_numeric(simple_agent_eval[:,6])

    if training:
        rl_agent_eval = np.array(agents['rl']._train)
        rl_x = pd.to_numeric(rl_agent_eval[:,1])
        rl_y = pd.to_numeric(rl_agent_eval[:,5])
    else:
        rl_agent_eval = np.array(agents['rl'].eval)
        rl_x = pd.to_numeric(rl_agent_eval[:,1])
        rl_y = pd.to_numeric(rl_agent_eval[:,6])

    plt.plot(random_x[rows], random_y[rows], '-b', label='Random')
    plt.plot(simple_x[rows], simple_y[rows], '-g', label='Simple')
    plt.plot(rl_x[rows], rl_y[rows], '-r', label='RL')
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.suptitle(title, fontsize=12)
    plt.title(subtitle, fontsize=10)
    plt.legend(loc='upper right')    
    plt.savefig('out/out_all_{}.png'.format(filename))
    plt.close()

# ______________________________________________________________________________

# Markov Decision Process

class EnvMDP(MDP):

    """A two-dimensional grid MDP, as in [Figure 17.1]. All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state). Also, you should specify the terminal states."""

    def __init__(self, env, gamma=.99):
        grid = EnvMDP.to_grid_matrix(env)
        reward = {}
        states = set()
        self.rows = len(grid)
        self.cols = len(grid[0])
        self.grid = grid

        for x in range(self.cols):
            for y in range(self.rows):
                if grid[y][x] is not None:
                    states.add((x, y))
                    reward[(x, y)] = grid[y][x]

        self.states = states
        
        terminals = EnvMDP.to_position(env, letter=b'GH')
        actlist = list(range(env.action_space.n))
        transitions = EnvMDP.to_transitions(env)
        init = EnvMDP.to_position(env, letter=b'S')[0]

        MDP.__init__(self, init, actlist=actlist,
                     terminals=terminals, transitions=transitions, 
                     reward=reward, states=states, gamma=gamma)

    def T(self, state, action):
        return self.transitions[state][action] if action is not None else [(0.0, state)]

    def to_arrows(self, policy):
        return policy_to_arrows(policy, self.rows, self.cols)

    @staticmethod
    def to_decoded(env):
        matrix = env.desc.reshape(env.nrow * env.ncol)
        
        grid = [str(state, "utf-8") for state in matrix]
        return np.array(grid).reshape((env.nrow, env.ncol))        

    @staticmethod
    def to_grid_matrix(env):
        """ 
        Maps the state space from an Open AI env to a simple grid
        """    
        matrix = env.desc.reshape(env.nrow * env.ncol)

        def state_value(state):
            if b'H' in state:
                return env.reward_hole
            if b'G' in state:
                return env.reward
            return env.path_cost

        grid = [state_value(state) for state in matrix]
        return np.array(grid).reshape((env.nrow, env.ncol))

    @staticmethod
    def to_grid_env(env):
        """ 
        Maps the state space from an Open AI env to a simple grid
        """    
        matrix = env.desc.reshape(env.nrow * env.ncol)

        def state_value(state):
            if b'H' in state:
                return b'H'
            if b'G' in state:
                return b'G'
            return '.'

        grid = [state_value(state) for state in matrix]
        return np.array(grid).reshape((env.nrow, env.ncol))        

    @staticmethod
    def to_position(env, letter=b'S'):
        """ 
        Maps the state space from the Open AI env to the positions
        """    
        grid = list(env.desc.reshape(env.nrow * env.ncol))
        
        indexes = [i for i, val in enumerate(grid) if bytes(val) in letter]

        if len(indexes):
            return [pos_to_coord(pos, env.ncol) for pos in indexes]

        raise ValueError('Env does not contain position: ' + letter)

    @staticmethod
    def to_transitions(env):
        """ 
        Maps the state space from the Open AI env to transitions
        transitions[current_pos][action] = [(prob, newstate)]

        """            
        
        ncol = env.ncol

        transitions = {}

        for state in env.P:
            pos = pos_to_coord(state, ncol)
            transitions[pos] = {}
            for action in env.P[state]:
                _transitions = env.P[state][action]
                transition = [(p, pos_to_coord(s, ncol)) for p, s, __, __ in _transitions]
                transitions[pos][action] = transition
            
        return transitions

# ______________________________________________________________________________

# Reinforcement Learning. Policies

def policy_to_grid(mapping, rows, cols):
    """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""

    states = []
    for pos in list(range(rows * cols)):
        coord = pos_to_coord(pos, cols)
        if coord in mapping:
            states.append(mapping[coord])
        else: 
            states.append('')        
    
    return np.array(states).reshape(rows, cols)

def policy_to_arrows(policy, rows, cols):
    chars = {2: '>', 3: '^', 0: '<', 1: 'v', None: '.'}
    return policy_to_grid({s: chars[a] for (s, a) in policy.items()}, rows, cols)

def policy_to_list(policy):
    return [[x, y, to_human(policy[(x, y)])] for x, y in policy]

# ______________________________________________________________________________

# Reinforcement Learning. QLearning

def compare_utils(U1, U2, H1="U1", H2="U2"):
    """ Source of this function: 
        - Labs from Artifial Intelligence (H), University of Glasgow class 2019
    """
    U_diff = dict()
    
    print("%s \t %s \t %s \t %s" % ("State",H1,H2,"Diff"))
    U_2norm = 0.0
    U_maxnorm = -10000

    for state in U1.keys():
        if state not in U2:
            print("State skipped: ", state)
            continue

        U_diff[state] = U1[state] - U2[state]        
        U_2norm = U_2norm + U_diff[state] ** 2

        if np.abs(U_diff[state]) > U_maxnorm:
            U_maxnorm = np.abs(U_diff[state])
        
        print("%s: \t %+.3f \t %+.3f \t %+.5f" % (state,U1[state],U2[state],U_diff[state]))
    
    print("")    
    print("Max norm: %.5f" % (U_maxnorm))     
    print("2-norm : %.5f" % (U_2norm))     


# ______________________________________________________________________________

def env2statespace(env):
    """ 
    This simple parser demonstrates how you can extract the state space from the Open AI env

    We *assume* full observability, i.e., we can directly ignore Hole states. Alternatively, 
    we could place a very high step cost to a Hole state or use a directed representation 
    (i.e., you can go to a Hole state but never return). Feel free to experiment with both if time permits.

    Input:
        env: an Open AI Env follwing the std in the FrozenLake-v0 env

    Output:
        state_space_locations : a dict with the available states
        state_space_actions   : a dict of dict with available actions in each state
        state_start_id        : the start state
        state_goal_id         : the goal state  
        states_indexes        : a dict with observation locations

        These objects are enough to define a Graph problem using the AIMA toolbox, e.g., using  
        UndirectedGraph, GraphProblem and astar_search (as in AI (H) Lab 3)

    Notice: the implementation is very explicit to demonstarte all the steps (it could be made more elegant!)

    """
    state_space_locations = {} # create a dict
    states_indexes = {} # create a dict
    cont = 0
    rewards = {}
    for i in range(env.desc.shape[0]):
        for j in range(env.desc.shape[1]):  
            #temp = i
            #i = j
            #j = temp
            #print(env.desc[0,1])
            #print(env.desc)
            states_indexes[cont] = (int(j), int(i))
            
            cont += 1 

            rewards[cont-1] = 0
            if (b'H' in env.desc[i,j]):
                rewards[cont-1] = -1
            else:
                state_id = "S_" + str(int(j)) + "_" + str(int(i))
                state_space_locations[state_id] = (int(j), int(i))
                if env.desc[i,j] == b'S':
                    state_initial_id = state_id                                                 
                elif env.desc[i,j] == b'G':
                    state_goal_id = state_id                      
                    rewards[cont-1] = 1

                #-- Generate state / action list --#
                # First define the set of actions in the defined coordinate system             
                actions = {"west": [-1,0],"east": [+1,0],"north": [0,+1], "south": [0,-1]}
                state_space_actions = {}
                for state_id in state_space_locations:                                       
                    possible_states = {}
                    for action in actions:
                        #-- Check if a specific action is possible --#
                        delta = actions.get(action)
                        state_loc = state_space_locations.get(state_id)
                        state_loc_post_action = [state_loc[0]+delta[0],state_loc[1]+delta[1]]

                        #-- Check if the new possible state is in the state_space, i.e., is accessible --#
                        state_id_post_action = "S_" + str(state_loc_post_action[0]) + "_" + str(state_loc_post_action[1])                        
                        if state_space_locations.get(state_id_post_action) != None:
                            possible_states[state_id_post_action] = 1 
                        
                    #-- Add the possible actions for this state to the global dict --#                              
                    state_space_actions[state_id] = possible_states

    return state_space_locations, state_space_actions, state_initial_id, state_goal_id, states_indexes, rewards
