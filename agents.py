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
  def __init__(self, 
               problem_id=0, 
               max_episodes=200, 
               max_iter_per_episode=500, reward_hole=0.0, is_stochastic=True, map_name_base="8x8-base", skip_header=False):
    # map_name_base="4x4-base"        
    self.env = LochLomondEnv(
      problem_id=problem_id, 
      is_stochastic=is_stochastic, 
      reward_hole=reward_hole, 
      map_name_base=map_name_base
    )
    self.map_name_base = map_name_base
    self.problem_id = problem_id
    self.is_stochastic = is_stochastic
    self.reward_hole = reward_hole    
    self.max_episodes = max_episodes
    self.max_iter_per_episode = max_iter_per_episode
    # state_space_locations, state_space_actions, state_initial_id, state_goal_id, states_indexes
    self.env_mapping = env2statespace(self.env)
    self.coordinates = self.env_mapping[4]
    self.skip_header = skip_header
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
    if (self.skip_header is False):
      self.lines.append(self.header())

  def reset_rewards(self):
    self.total_rewards = 0

  def solve(self):
    self.reset_lines()
    self.reset_rewards()

    for e in range(self.max_episodes): # iterate over episodes
      observation = self.env.reset() # reset the state of the env to the starting state     
      self.set_episode_seed(e)

      for iter in range(self.max_iter_per_episode):
        action = self.env.action_space.sample() # your agent goes here (the current agent takes random actions)
        prev_location = self.coordinates[observation]

        observation, reward, done, info = self.env.step(action) # observe what happends when you take the action
        self.lines.append([
          self.problem_id,
          self.map_name_base,
          e+1, 
          iter+1, 
          map_action(action), 
          done, 
          reward, 
          self.total_rewards, 
          prev_location[0], 
          prev_location[1], 
          self.coordinates[observation][0], 
          self.coordinates[observation][1]
        ])

        if (done and reward == self.reward_hole): 
          break

        if (done and reward == +1.0):
          self.total_rewards += 1 
          break

    return self.total_rewards

  def set_episode_seed(self, seed):
    # by default no seed for abstract agent
    return None

################################
################################

class RandomAgent(MyAbstractAIAgent):

  """
  TODO


  """  
  def __init__(self, problem_id=0, max_episodes=200, max_iter_per_episode=500, map_name_base="8x8-base", skip_header=False):
    super(RandomAgent, self).__init__(
      problem_id=problem_id, 
      max_episodes=max_episodes, 
      max_iter_per_episode=max_iter_per_episode, 
      reward_hole=0.0, 
      is_stochastic=True,
      map_name_base=map_name_base,
      skip_header=skip_header
    )

  def action(self):
    return self.env.action_space.sample()

  def set_episode_seed(self, seed):
    np.random.seed(seed)
    return None

################################
################################

class SimpleAgent(MyAbstractAIAgent):

  """
  TODO


  """   
  def __init__(self, problem_id=0, max_episodes=200, max_iter_per_episode=500, map_name_base="8x8-base"):
    super(SimpleAgent, self).__init__(
      problem_id=problem_id, 
      max_episodes=max_episodes, 
      max_iter_per_episode=max_iter_per_episode, 
      reward_hole=0.0, 
      is_stochastic=False,
      map_name_base=map_name_base
    )

  def solve(self):
    locations, actions, state_initial_id, state_goal_id, my_map = self.env_mapping
    self.maze_map = UndirectedGraph(actions)
    self.maze_map.locations = locations

    self.node_colors = dict()

    print(locations)
    print(self.env.render())
    print(state_initial_id)
    print(actions)
    # initialise a graph
    for n, p in locations.items():   
      self.node_colors[n] = "white" # node_colors to color nodes while exploring the map    

    #print(self.maze_map.__dict__)
    self.maze_problem = GraphProblem(state_initial_id, state_goal_id, self.maze_map)
    #print(self.maze_problem.__dict__)
    # print(self.maze_problem.h)
    iterations, all_node_colors, node = self.my_astar_search(problem=self.maze_problem, h=None)
    print(iterations)
    print(all_node_colors)
    print(node.solution())

  def action(self):
    return "TODO"
    #return self.env.action_space.sample()
  
  def my_astar_search(self, problem, h=None):
      """A* search is best-first graph search with f(n) = g(n)+h(n).
      You need to specify the h function when you call astar_search, or
      else in your Problem subclass."""
      h = memoize(h or problem.h, 'h') # define the heuristic function
      return self.my_best_first_graph_search(problem, lambda n: n.path_cost + h(n))    

  def my_best_first_graph_search(self, problem, f):
      """Search the nodes with the lowest f scores first.
      You specify the function f(node) that you want to minimize; for example,
      if f is a heuristic estimate to the goal, then we have greedy best
      first search; if f is node.depth then we have breadth-first search.
      There is a subtlety: the line "f = memoize(f, 'f')" means that the f
      values will be cached on the nodes as they are computed. So after doing
      a best first search you can examine the f values of the path returned."""

      ## we use these two variables at the time of visualisations
      iterations = 0
      all_node_colors = []
      node_colors = dict(self.node_colors)
     
      #print(f)
      f = memoize(f, 'f')
      node = Node(problem.initial)
      #print(node)
      #print(node.state)

      node_colors[node.state] = "red"
      iterations += 1
      print(iterations)

      all_node_colors.append(dict(node_colors))

      if problem.goal_test(node.state):
          node_colors[node.state] = "green"
          iterations += 1
          all_node_colors.append(dict(node_colors))
          return(iterations, all_node_colors, node)
      
      frontier = PriorityQueue('min', f)
      frontier.append(node)
     
      node_colors[node.state] = "orange"
      iterations += 1    
      all_node_colors.append(dict(node_colors))

      explored = set()
      frontier_iteration = 0
      
      while frontier:
        frontier_iteration += 1
        print("Iteration: ", frontier_iteration, "- Length: ", len(frontier))

        node = frontier.pop()
        print("Exploring: ", node)
        print("Explored: ", explored)
                
        node_colors[node.state] = "red"
        iterations += 1
        all_node_colors.append(dict(node_colors))
        
        if problem.goal_test(node.state):
          node_colors[node.state] = "green"
          iterations += 1
          print("GOAL REACHED")
          all_node_colors.append(dict(node_colors))
          return(iterations, all_node_colors, node)

        explored.add(node.state)
        
        for child in node.expand(problem):
          if child.state not in explored and child not in frontier:
              print("child: ", child)
              frontier.append(child)
              node_colors[child.state] = "orange"
              iterations += 1
              all_node_colors.append(dict(node_colors))
          elif child in frontier:
              incumbent = frontier[child]
              if f(child) < f(incumbent):
                  del frontier[incumbent]
                  frontier.append(child)
                  node_colors[child.state] = "orange"
                  iterations += 1
                  all_node_colors.append(dict(node_colors))

        node_colors[node.state] = "gray"
        iterations += 1
        all_node_colors.append(dict(node_colors))
      return None