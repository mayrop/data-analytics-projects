import unittest
from agents import RandomAgent, SimpleAgent, QLearningAgent
import random
import numpy as np
from helpers import *
from uofgsocsai import LochLomondEnv

# python test.py TestRandomAgent.test_4_by_4_q_learning_agent

class TestRandomAgent(unittest.TestCase):

    def test_position_to_coordinates(self):
        self.assertEqual((0, 1), position_to_coordinates(1, 3))
        self.assertEqual((0, 2), position_to_coordinates(2, 3))
        self.assertEqual((2, 2), position_to_coordinates(8, 3))
        self.assertEqual((3, 0), position_to_coordinates(9, 3))
        self.assertEqual((3, 1), position_to_coordinates(10, 3))
        self.assertEqual((3, 2), position_to_coordinates(11, 3))

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

        grid = env_to_grid(env)
        self.assertEqual(0, grid[0,0])
        self.assertEqual(0, grid[0,1])
        self.assertEqual(-0.2, grid[1,1])
        self.assertEqual(env.reward, grid[3,0])


    def test_env_2_terminals(self):
        env = LochLomondEnv(problem_id=1, is_stochastic=True, 
                            reward_hole=-0.2, map_name_base="4x4-base")

        terminals = env_letter_to_position(env, letter=b'GH')
        self.assertEqual((1, 1), terminals[0])
        self.assertEqual((1, 3), terminals[1])
        self.assertEqual((2, 3), terminals[2])
        self.assertEqual((3, 0), terminals[3])
        self.assertEqual((3, 1), terminals[4])


    def test_env_2_init(self):
        env = LochLomondEnv(problem_id=1, is_stochastic=True, 
                            reward_hole=-0.2, map_name_base="4x4-base")
        initial = env_letter_to_position(env, letter=b'S')
        self.assertEqual((0, 1), initial[0])


    def test_env_2_init_02(self):
        env = LochLomondEnv(problem_id=2, is_stochastic=True, 
                            reward_hole=-0.2, map_name_base="4x4-base")
        initial = env_letter_to_position(env, letter=b'S')
        self.assertEqual((0, 2), initial[0])    


    def test_4_by_4_q_learning_agent(self):
        agent = QLearningAgent(problem_id=0, map_name_base="4x4-base")
        # self.assertEqual(agent.get_learning_rate(), 1.0)
        # self.assertEqual(0, agent.s)
        # agent.last_action = 1
        # agent.update_nsa()

        # self.assertEqual(agent.get_learning_rate(), 0.5)
        # agent.update_nsa()

        # self.assertAlmostEqual(agent.get_learning_rate(), 0.333, places=3)
        # agent.update_nsa()

        # self.assertEqual(agent.get_learning_rate(), 0.25)
        #agent.solve(max_episodes=1,max_iter_per_episode=1)


    def test_4_by_4_q_learning_agent_solve(self):
        agent = QLearningAgent(problem_id=1, map_name_base="4x4-base")

#        print(agent.env.desc)

        #print(position_to_coordinates(agent.env))
#         print(agent.env.render())
#         print(agent.env.terminals)

#         policy = agent.policy_iteration()
#         policy_list = (list(policy.values()))

#         for k in agent.env.terminals:
#             policy_list[k] = -1

#         #policy_list[np.argmax(agent.env.isd)] = None
#         print(agent.env_mapping)

#         print(policy)
#         print(agent.env.terminals)

# #        agent.solve(max_episodes=1000, max_iter_per_episode=100)

#         human = [to_human_arrow(k) for k in policy_list]
        
#         print(np.array(human).reshape(4, 4))
#         #agent.graph_utility_estimates_q()
#         agent.solve()


        #print(agent.actions_in_state(0).keys().tolist())

    def test_4_by_4_simple_agent(self):
        agent = SimpleAgent(problem_id=1, map_name_base="8x8-base")
        agent.solve()

    def test_4_by_4_random_agent(self):
        agent = RandomAgent(problem_id=1, map_name_base="4x4-base")
        self.assertEqual(agent.problem_id, 1)
        self.assertEqual(agent.is_stochastic, True)
        self.assertEqual(agent.env.ncol, 4)
        self.assertEqual(agent.env.nrow, 4)

        self.assertTrue('S_0_0' in agent.env_mapping[0])
        self.assertTrue('S_0_1' in agent.env_mapping[0])
        self.assertFalse('S_1_1' in agent.env_mapping[0])

    def test_8_by_8_random_agent(self):
        agent = RandomAgent(problem_id=0)
        self.assertEqual(agent.problem_id, 0)
        self.assertEqual(agent.is_stochastic, True)
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
        self.assertEqual(agent.is_stochastic, True)
        self.assertEqual(agent.env.ncol, 8)
        self.assertEqual(agent.env.nrow, 8)

        # state_space_locations, state_space_actions, state_initial_id, state_goal_id, states_indexes
        self.assertFalse('S_0_0' in agent.env_mapping[0])
        self.assertTrue('S_0_1' in agent.env_mapping[0])
        # self.assertEqual('S_0_1', agent.env_mapping[2])

        # agent = RandomAgent(problem_id=2)
        # self.assertEqual(agent.problem_id, 2)
        # self.assertEqual(agent.is_stochastic, True)
        # self.assertEqual(agent.env.ncol, 8)
        # self.assertEqual(agent.env.nrow, 8)

        # # state_space_locations, state_space_actions, state_initial_id, state_goal_id, states_indexes
        # self.assertFalse('S_0_0' in agent.env_mapping[0])
        # self.assertTrue('S_0_1' in agent.env_mapping[0])
        # self.assertEqual('S_0_2', agent.env_mapping[2])   



if __name__ == '__main__':
    unittest.main()