import numpy as np
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
from helpers import *

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

class RandomAgent():
  def __init__(self, problem_id=0, max_episodes=20, max_iter_per_episode=500, reward_hole=0.0, is_stochastic=True):
    self.problem_id = problem_id
    self.is_stochastic = is_stochastic
    self.reward_hole = reward_hole
    self.env = LochLomondEnv(problem_id=self.problem_id, is_stochastic=self.is_stochastic, reward_hole=reward_hole)
    self.max_episodes = max_episodes
    self.max_iter_per_episode = max_iter_per_episode
    self.total_rewards = 0
    self.lines = []

  def solve(self):
    coordinates = env2statespace(self.env)[4]
    self.lines = [] # reset
    self.lines.append(self.header())
    self.total_rewards = 0

    for e in range(self.max_episodes): # iterate over episodes
      observation = self.env.reset() # reset the state of the env to the starting state     
      #print("observation: ", observation)
      np.random.seed(e)

      for iter in range(self.max_iter_per_episode):
        # self.env.render() # for debugging/develeopment you may want to visualize the individual steps by uncommenting this line      
        action = self.env.action_space.sample() # your agent goes here (the current agent takes random actions)
        prev_location = coordinates[observation]

        observation, reward, done, info = self.env.step(action) # observe what happends when you take the action
        self.lines.append([e+1, iter+1, map_action(action), done, reward, self.total_rewards, prev_location[0], prev_location[1], coordinates[observation][0], coordinates[observation][1]])

        if (done and reward == self.reward_hole): 
          #env.render()     
          #print("We have reached a hole :-( [we can't move so stop trying; just give up]")

          break
        if (done and reward == +1.0):
          #env.render()
          self.total_rewards += 1 
          break

    return self.total_rewards

  def header(self):
    return ["Episode", "Iteration", "Action", "Done", "Reward", "CumulativeReward", "PrevLocationX", "PrevLocationY", "NewLocationX", "NewLocationY"]

  def action(self):
    return self.env.action_space.sample()

  def problem_id(self):
    return self.problem_id

  def env(self):
    return self.env
