import sys
sys.path.append("aima")
from search import *
from mdp import MDP
import numpy as np
import matplotlib.pyplot as plt

def get_action_from_location(previous, current):
    # todo - double check

    # previous[0] = y coordinate of prev value
    # previous[1] = x coordinate of prev value

    # current[0] = y coordinate of current value
    # current[1] = x coordinate of current value

    # down
    if current[0] > previous[0]:
        return 1
    # up
    if current[0] < previous[0]:
        return 3
    # right
    if current[1] > previous[1]:
        return 2
    # left
    if current[1] < previous[1]:
        return 0


def mean_rewards(row):
    return row.cumulative_rewards / (int(row.episode) + 1)


def add_plot(x, y, name, title, subtitle, labels):
    plt.plot(x, y)
    plt.xlabel(labels[0])
    plt.ylabel(labels[1])
    plt.suptitle(title)
    plt.title(subtitle)    
    plt.savefig(name)
    plt.close()


def position_to_coordinates(pos, ncol):
    """ 
    TODO
    """       
    return (pos % ncol, pos // ncol)


def get_terminals(env):
    """ 
    TODO
    """    
    grid = list(env.desc.reshape(env.nrow * env.ncol))
    
    return [i for i, val in enumerate(grid) if bytes(val) in b'GH']


def to_grid(mapping, rows, cols):
    """ Inspired in GridMDP(MDP) from AIMA Toolbox """

    states = []
    for pos in list(range(rows * cols)):
        if pos in mapping:
            states.append(mapping[pos])
        else: 
            states.append('')        
    
    return np.array(states).reshape(rows, cols)


def to_arrows(policy, rows, cols):
    """ Inspired in GridMDP(MDP) from AIMA Toolbox """
    chars = {2: '>', 3: '^', 0: '<', 1: 'v', None: '.'}

    return to_grid({s: chars[a] for (s, a) in policy.items()}, rows, cols)


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

        These objects are enough to define a Graph problem using the AIMA toolbox, e.g., using  
        UndirectedGraph, GraphProblem and astar_search (as in AI (H) Lab 3)

    Notice: the implementation is very explicit to demonstarte all the steps (it could be made more elegant!)

    """
    state_space_locations = {} # create a dict
    for i in range(env.desc.shape[0]):
        for j in range(env.desc.shape[1]):   
            if not (b'H' in env.desc[i,j]):
                state_id = "S_"+str(int(i))+"_"+str(int(j) )
                state_space_locations[state_id] = (int(i),int(j))
                if env.desc[i,j] == b'S':
                    state_initial_id = state_id                                                 
                elif env.desc[i,j] == b'G':
                    state_goal_id = state_id                      

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
                        state_id_post_action = "S_"+str(state_loc_post_action[0])+"_"+str(state_loc_post_action[1])                        
                        if state_space_locations.get(state_id_post_action) != None:
                            possible_states[state_id_post_action] = 1 
                        
                    #-- Add the possible actions for this state to the global dict --#                              
                    state_space_actions[state_id] = possible_states

    return state_space_locations, state_space_actions, state_initial_id, state_goal_id