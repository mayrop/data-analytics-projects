import numpy as np
import sys
AIMA_TOOLBOX_ROOT="aima-python"
sys.path.append(AIMA_TOOLBOX_ROOT)
from search import *
from mdp import MDP
from rl import PassiveTDAgent
from rl import run_single_trial
import matplotlib.pyplot as plt

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

def to_human_arrow(action):
    if action == 0:
        return "<"
    elif action == 1:
        return "v"
    elif action == 2:
        return ">"
    elif action == 3:
        return "^"
    elif action == -1:
        return "H"
    elif action == 100:
        return "G"
    elif action == None:
        return "X"
    return "."    


# ______________________________________________________________________________


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


def pos_to_coord(pos, ncol):
    return (pos % ncol, pos // ncol)


def graph_utility_estimates(agent, mdp, iterations, states):
    graphs = {state:[] for state in states}

    for iteration in range(1, iterations+1):
        run_single_trial(agent, mdp)
        for state in states:
            graphs[state].append((iteration, agent.U[state]))

    for state, value in graphs.items():
        state_x, state_y = zip(*value)
        plt.plot(state_x, state_y, label=str(state))

    plt.ylim([-1.2, 1.2])
    plt.legend(loc='lower right')
    plt.xlabel('Iterations')
    plt.ylabel('U')
    plt.show(block=True)

# ______________________________________________________________________________


class EnvMDP(MDP):

    """A two-dimensional grid MDP, as in [Figure 17.1]. All you have to do is
    specify the grid as a list of lists of rewards; use None for an obstacle
    (unreachable state). Also, you should specify the terminal states.
    An action is an range[0, action] unit vector; e.g. (1, 0) means move east."""

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
        actlist = list(range(env.action_space.n))
        transitions = EnvMDP.to_transitions(env)
        terminals = EnvMDP.to_position(env, letter=b'GH')
        init = EnvMDP.to_position(env, letter=b'S')[0]

        MDP.__init__(self, init, actlist=actlist,
                     terminals=terminals, transitions=transitions, 
                     reward=reward, states=states, gamma=gamma)

    def T(self, state, action):
        return self.transitions[state][action] if action is not None else [(0.0, state)]

    def to_grid(self, mapping):
        """Convert a mapping from (x, y) to v into a [[..., v, ...]] grid."""
        
        rows, cols = self.rows, self.cols
        states = []
        for pos in list(range(rows * cols)):
            coord = pos_to_coord(pos, cols)
            states.append(mapping[coord])
        
        return np.array(states).reshape(rows, cols)

    def to_arrows(self, policy):
        chars = {2: '>', 3: '^', 0: '<', 1: 'v', None: '.'}
        return self.to_grid({s: chars[a] for (s, a) in policy.items()})


    @staticmethod
    def to_grid_matrix(env):
        """ 
        This simple parser maps the state space from the Open AI env to a simple grid

        Input:
            env: an Open AI Env follwing the std in the FrozenLake-v0 env

        Output:
            a grid with the reward values for each state with shape (env.nrow * env.ncol)

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
    def to_position(env, letter=b'S'):
        """ 
        This simple parser maps the state space from the Open AI env to the positions

        Input:
            env: an Open AI Env follwing the std in the FrozenLake-v0 env

        Output:
            array with the positions that match letter

        """    
        grid = list(env.desc.reshape(env.nrow * env.ncol))
        
        indexes = [i for i, val in enumerate(grid) if bytes(val) in letter]

        if len(indexes):
            return [pos_to_coord(pos, env.ncol) for pos in indexes]

        raise ValueError('Env does not contain position: ' + letter)

    @staticmethod
    def to_transitions(env):
        """ 
        This simple parser maps the state space from the Open AI env to a simple grid

        Input:
            env: an Open AI Env follwing the std in the FrozenLake-v0 env

        Output:
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


