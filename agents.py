import gym
import numpy as np
import time
from uofgsocsai import LochLomondEnv # load the class defining the custom Open AI Gym problem
import os, sys

class RandomAgent():
  def __init__(self, problem_id=0, max_episodes=20, max_iter_per_episode=500):
    self.problem_id = problem_id
    self.reward_hole = 0.0
    self.is_stochastic = True
    self.env = LochLomondEnv(problem_id=self.problem_id, is_stochastic=self.is_stochastic, reward_hole=self.reward_hole)
    self.max_episodes = max_episodes
    self.max_iter_per_episode = max_iter_per_episode

  def solve(self):
    performances = []

    for e in range(self.max_episodes): # iterate over episodes
      observation = self.env.reset() # reset the state of the env to the starting state     
      
      for iter in range(self.max_iter_per_episode):
        # your agent goes here (the current agent takes random actions)
        # action = self.action()
        # observation, reward, done, info = self.env.step(action) # observe what happends when you take the action

        # print("observation: ", observation)
        # print("reward: ", reward)
        # print("done: ", done)
        # print("info: ", info)
        # print("env.action_space: ", self.env.action_space.__dict__)
        # # TODO: You'll need to add code here to collect the rewards for plotting/reporting in a suitable manner

        # print("e, iter, reward, done =" + str(e) + " " + str(iter) + " " + str(reward) + " " + str(done))

        # # Check if we are done and monitor rewards etc...
        # if (done and reward == self.reward_hole): 
        #   self.env.render()     
        #   print("We have reached a hole :-( [we can't move so stop trying; just give up]")
        #   break

        # if (done and reward == +1.0):
        #   self.env.render()     
        #   print("We have reached the goal :-) [stop trying to move; we can't]. That's ok we have achived the goal]")
        #   break

  def action(self):
    return self.env.action_space.sample()

  def problem_id(self):
    return self.problem_id

  def env(self):
    return self.env
