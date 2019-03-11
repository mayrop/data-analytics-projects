import numpy as np

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


def env2grid(env):
    """ 
    This simple parser maps the state space from the Open AI env to a simple grid

    We *assume* full observability, i.e., we can directly ignore Hole states. Alternatively, 
    we could place a very high step cost to a Hole state or use a directed representation 
    (i.e., you can go to a Hole state but never return). Feel free to experiment with both if time permits.

    Input:
        env: an Open AI Env follwing the std in the FrozenLake-v0 env

    Output:
        a grid with the reward values for each state with shape (env.nrow * env.ncol)

    """    
    matrix = env.desc.reshape(env.nrow * env.ncol)

    def state_default_value(x):
        if b'H' in x:
            return env.reward_hole
        if b'G' in x:
            return env.reward
        return env.path_cost

    grid = [state_default_value(x) for x in matrix]
    return np.array(grid).reshape((env.nrow, env.ncol))


def position_to_coordinates(pos, ncol):
    return (pos // ncol, pos % ncol)


def env_to_terminals(env):
    """ 
    This simple parser maps the state space from the Open AI env to a simple grid

    We *assume* full observability, i.e., we can directly ignore Hole states. Alternatively, 
    we could place a very high step cost to a Hole state or use a directed representation 
    (i.e., you can go to a Hole state but never return). Feel free to experiment with both if time permits.

    Input:
        env: an Open AI Env follwing the std in the FrozenLake-v0 env

    Output:
        array with the positions of terminals

    """
#    def coordinates(state):
    grid = env.desc.reshape(env.nrow * env.ncol)
    terminals = []

    for key, val in enumerate(grid):
        if b'H' in val or b'G' in val:
            terminals.append(position_to_coordinates(key, env.ncol))

    return terminals


# ______________________________________________________________________________


