import unittest
from agents import RandomAgent, SimpleAgent, UofGPassiveAgent, UofGQLearningAgent
import random
import numpy as np
from mdp import policy_iteration
from mdp import value_iteration
from helpers import *
from uofgsocsai import LochLomondEnv
from rl import PassiveTDAgent
from rl import run_single_trial
from utils import argmax, vector_add, print_table

# python test.py TestRandomAgent.test_4_by_4_q_learning_agent

class TestRandomAgent(unittest.TestCase):

    def test_pos_to_coord(self):
        self.assertEqual((1, 0), pos_to_coord(1, 3))
        self.assertEqual((2, 0), pos_to_coord(2, 3))
        self.assertEqual((2, 2), pos_to_coord(8, 3))
        self.assertEqual((0, 3), pos_to_coord(9, 3))
        self.assertEqual((1, 3), pos_to_coord(10, 3))
        self.assertEqual((2, 3), pos_to_coord(11, 3))

    def test_coord_to_pos(self):
        self.assertEqual(1, coord_to_pos(1, 0, 3))
        self.assertEqual(2, coord_to_pos(2, 0, 3))
        self.assertEqual(8, coord_to_pos(2, 2, 3))
        self.assertEqual(9, coord_to_pos(0, 3, 3))
        self.assertEqual(10, coord_to_pos(1, 3, 3))
        self.assertEqual(11, coord_to_pos(2, 3, 3))
        self.assertEqual(7, coord_to_pos(3, 1, 4))
        self.assertEqual(5, coord_to_pos(1, 2, 2))

    def test_env(self):
        env = LochLomondEnv(problem_id=0, is_stochastic=True, 
                            reward_hole=-0.02, map_name_base="4x4-base")

        self.assertEqual(b'S', env.desc[0,0])
        self.assertEqual(b'F', env.desc[0,1])
        self.assertEqual(b'H', env.desc[1,1])
        self.assertEqual(b'G', env.desc[3,0])


    def test_env_2_grid(self):
        env = LochLomondEnv(problem_id=0, is_stochastic=True, 
                            reward_hole=-0.2, map_name_base="4x4-base")

        mdp = EnvMDP(env)
        grid = EnvMDP.to_grid_matrix(env)
        self.assertEqual(0, grid[0,0])
        self.assertEqual(0, grid[0,1])
        self.assertEqual(-0.2, grid[1,1])
        self.assertEqual(env.reward, grid[3,0])


    def test_env_2_terminals(self):
        env = LochLomondEnv(problem_id=1, is_stochastic=True, 
                            reward_hole=-0.2, map_name_base="4x4-base")

        mdp = EnvMDP(env)
        terminals = EnvMDP.to_position(env, letter=b'GH')

        self.assertEqual((1, 1), terminals[0])
        self.assertEqual((3, 1), terminals[1])
        self.assertEqual((3, 2), terminals[2])
        self.assertEqual((0, 3), terminals[3])
        self.assertEqual((1, 3), terminals[4])


    def test_env_mdp(self):
        env = LochLomondEnv(problem_id=1, is_stochastic=True, 
                            reward_hole=-0.2, map_name_base="4x4-base")

        mdp = EnvMDP(env)

        self.assertEqual(4, mdp.rows)
        self.assertEqual(4, mdp.cols)
        self.assertAlmostEqual(-0.2, mdp.grid[3][0])
        self.assertTrue((0, 1) in mdp.states)
        self.assertEqual((1, 1), mdp.terminals[0])


    def test_env_2_transitions(self):
        env = LochLomondEnv(problem_id=1, is_stochastic=True, 
                            reward_hole=-0.2, map_name_base="4x4-base")

        mdp = EnvMDP(env)
        transitions = EnvMDP.to_transitions(env)
        # transitions[current_pos][action] = [(prob, newstate)]

        # moving to the left should...
        ## move to the bottom with 0.333 prob
        self.assertAlmostEqual(transitions[(0, 0)][0][2][0], 0.333, places=3)
        self.assertEqual(transitions[(0, 0)][0][2][1], (0, 1))

        ## stay 0.333 prob
        self.assertAlmostEqual(transitions[(0, 0)][0][1][0], 0.333, places=3)
        self.assertEqual(transitions[(0, 0)][0][1][1], (0, 0))

        ## stay 0.333 prob
        self.assertAlmostEqual(transitions[(0, 0)][0][0][0], 0.333, places=3)
        self.assertEqual(transitions[(0, 0)][0][0][1], (0, 0))

        # moving to the down should...
        ## stay 0.333 prob
        self.assertAlmostEqual(transitions[(0, 0)][1][0][0], 0.333, places=3)
        self.assertEqual(transitions[(0, 0)][1][0][1], (0, 0))

        ## move to the bottom with 0.333 prob
        self.assertAlmostEqual(transitions[(0, 0)][1][1][0], 0.333, places=3)
        self.assertEqual(transitions[(0, 0)][1][1][1], (0, 1))

        ## move to the right with 0.333 prob
        self.assertAlmostEqual(transitions[(0, 0)][1][2][0], 0.333, places=3)
        self.assertEqual(transitions[(0, 0)][1][2][1], (1, 0))


    def test_env_2_init(self):
        env = LochLomondEnv(problem_id=1, is_stochastic=True, 
                            reward_hole=-0.2, map_name_base="4x4-base")

        mdp = EnvMDP(env)
        initial = EnvMDP.to_position(env, letter=b'S')
        self.assertEqual((1, 0), initial[0])


    def test_env_2_init_02(self):
        env = LochLomondEnv(problem_id=2, is_stochastic=True, 
                            reward_hole=-0.2, map_name_base="4x4-base")
        initial = EnvMDP.to_position(env, letter=b'S')
        self.assertEqual((2, 0), initial[0])    


    def test_4_by_4_random_agent(self):
        agent = RandomAgent(problem_id=1, map_name_base="4x4-base")
        self.assertEqual(agent.problem_id, 1)
        self.assertEqual(agent.is_stochastic(), True)
        self.assertEqual(agent.env.ncol, 4)
        self.assertEqual(agent.env.nrow, 4)

        self.assertTrue('S_0_0' in agent.env_mapping[0])
        self.assertTrue('S_0_1' in agent.env_mapping[0])
        self.assertFalse('S_1_1' in agent.env_mapping[0])


    def test_8_by_8_random_agent(self):
        agent = RandomAgent(problem_id=0)
        self.assertEqual(agent.problem_id, 0)
        self.assertEqual(agent.is_stochastic(), True)
        self.assertEqual(agent.env.ncol, 8)
        self.assertEqual(agent.env.nrow, 8)
        # print(agent.env.render())
        # print(agent.env_mapping[0])

        # state_space_locations, state_space_actions, state_initial_id, state_goal_id, states_indexes
        self.assertTrue('S_0_0' in agent.env_mapping[0])
        self.assertTrue('S_0_1' in agent.env_mapping[0])
        self.assertEqual('S_0_0', agent.env_mapping[2])

        agent = RandomAgent(problem_id=1)
        self.assertEqual(agent.problem_id, 1)
        self.assertEqual(agent.is_stochastic(), True)
        self.assertEqual(agent.env.ncol, 8)
        self.assertEqual(agent.env.nrow, 8)

        # state_space_locations, state_space_actions, state_initial_id, state_goal_id, states_indexes
        self.assertFalse('S_0_0' in agent.env_mapping[0])
        self.assertTrue('S_0_1' in agent.env_mapping[0])
        # self.assertEqual('S_0_1', agent.env_mapping[2])

        # agent = RandomAgent(problem_id=2)
        # self.assertEqual(agent.problem_id, 2)
        # self.assertEqual(agent.is_stochastic(), True)
        # self.assertEqual(agent.env.ncol, 8)
        # self.assertEqual(agent.env.nrow, 8)

        # # state_space_locations, state_space_actions, state_initial_id, state_goal_id, states_indexes
        # self.assertFalse('S_0_0' in agent.env_mapping[0])
        # self.assertTrue('S_0_1' in agent.env_mapping[0])
        # self.assertEqual('S_0_2', agent.env_mapping[2])


    def test_simple_agent(self):
        agent = SimpleAgent(problem_id=1, map_name_base="8x8-base")
        agent.solve()
        
        self.assertEqual([1, 1, 11, 'down', 1, 0, 1], agent.eval[0])

        policy = agent.policy()        
        arrows = policy_to_arrows(policy, 8, 8)
        self.assertEqual(['', '>', '>', '>', 'v', '', '', ''], arrows[3].tolist())

        policy_list = policy_to_list(policy)
        self.assertEqual([1, 3, 'right'], policy_list[3])


    def test_passive(self):
        agent = UofGPassiveAgent(problem_id=1, map_name_base="8x8-base")
        agent.solve()
        
        policy = agent.policy()
        arrows = policy_to_arrows(policy, 8, 8)
        self.assertEqual(['v', '^', '^', '^', '>', 'v', '>', 'v'], arrows[1].tolist())
        
        policy_list = policy_to_list(policy)
        self.assertEqual([7, 3, 'down'], policy_list[0])

        agent.write_eval_files()


    def test_qlearning(self):
        agent = UofGQLearningAgent(problem_id=0, map_name_base="8x8-base")
        agent.solve()

        agent.write_eval_files()

if __name__ == '__main__':
    unittest.main()