import numpy as np
import sys
#AIMA_TOOLBOX_ROOT="aima-python"
sys.path.append("aima")
from search import *
from mdp import MDP
from rl import PassiveTDAgent
from rl import QLearningAgent
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

def compare_utils(U1, U2, H1="U1", H2="U2"):
    U_diff = dict()
    
    print("%s \t %s \t %s \t %s" % ("State",H1,H2,"Diff"))
    U_2norm = 0.0
    U_maxnorm = -10000

    for state in U1.keys():
        U_diff[state] = U1[state] - U2[state]        
        U_2norm = U_2norm + U_diff[state]**2

        if np.abs(U_diff[state]) > U_maxnorm:
            U_maxnorm = np.abs(U_diff[state])
        
        print("%s: \t %+.3f \t %+.3f \t %+.5f" % (state,U1[state],U2[state],U_diff[state]))
    
    print("")    
    print("Max norm: %.5f" % (U_maxnorm))     
    print("2-norm : %.5f" % (U_2norm))     
    #return U_diff,U_2norm,U_maxnorm

def my_graph_utility_estimates(agent, mdp, iterations, states=None):
    if states is None:
        states = mdp.states

    graphs = {state:[] for state in states}

    for i in range(1, iterations+1):
        if callable(getattr(agent, 'set_episode', None)):
            agent.set_episode(i+1)

        run_single_trial(agent, mdp)
        for state in states:
            if callable(getattr(agent, 'update_u', None)):
                agent.update_u()

            graphs[state].append((i, agent.U[state]))

    return agent, graphs


def graph_utility_estimates(agent, mdp, iterations, states=None):
    agent, graphs = my_graph_utility_estimates(agent, mdp, iterations, states)

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

def q_to_u(Q):
    U = defaultdict(lambda: -1000.) 
    
    for state_action, value in Q.items():
        state, action = state_action
        if U[state] < value:
            U[state] = value     

    return U

class QLearningAgentUofG(QLearningAgent):
    """ An exploratory Q-learning agent. It avoids having to learn the transition
        model because the Q-value of a state can be related directly to those of
        its neighbors. [Figure 21.8]

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

    def f(self, u, n, noise):
        if n < self.Ne:
            return self.Rplus        
        """ Exploration function. Returns fixed Rplus until
        agent has visited state, action a Ne number of times.
        Same as ADP agent in book."""
        #print((u + noise)[0])
        
        return u + noise

    def set_episode(self, e):
        self.e = e

    def __call__(self, percept):
        n_actions = len(self.all_act)
        noise = np.random.random((1, n_actions)) / (self.e**2.)

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
            self.a = argmax(actions_in_state(s1), key=lambda a1: self.f(Q[s1, a1], Nsa[s1, a1], noise[0][a1]))
            if random.uniform(0, 1) < 0.1:
                self.a = random.randint(0, n_actions-1)

        return self.a

    def update_u(self):
        self.U = q_to_u(self.Q)