import unittest
from agents import RandomAgent, SimpleAgent

class TestRandomAgent(unittest.TestCase):

    def test_4_by_4_simple_agent(self):
        agent = SimpleAgent(problem_id=1, max_episodes=500, max_iter_per_episode=500, map_name_base="4x4-base")

    def test_4_by_4_random_agent(self):
        agent = RandomAgent(problem_id=1, max_episodes=500, max_iter_per_episode=500, map_name_base="4x4-base")
        self.assertEqual(agent.problem_id, 1)
        self.assertEqual(agent.is_stochastic, True)
        self.assertEqual(agent.env.ncol, 4)
        self.assertEqual(agent.env.nrow, 4)

        self.assertTrue('S_0_0' in agent.env_mapping[0])
        self.assertTrue('S_0_1' in agent.env_mapping[0])
        self.assertFalse('S_1_1' in agent.env_mapping[0])

    def test_8_by_8_random_agent(self):
        agent = RandomAgent(problem_id=0, max_episodes=500, max_iter_per_episode=500, map_name_base="8x8-base")
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

        agent = RandomAgent(problem_id=1, max_episodes=500, max_iter_per_episode=500, map_name_base="8x8-base")
        self.assertEqual(agent.problem_id, 1)
        self.assertEqual(agent.is_stochastic, True)
        self.assertEqual(agent.env.ncol, 8)
        self.assertEqual(agent.env.nrow, 8)

        # state_space_locations, state_space_actions, state_initial_id, state_goal_id, states_indexes
        self.assertFalse('S_0_0' in agent.env_mapping[0])
        self.assertTrue('S_0_1' in agent.env_mapping[0])
        self.assertEqual('S_0_1', agent.env_mapping[2])

        agent = RandomAgent(problem_id=2, max_episodes=500, max_iter_per_episode=500, map_name_base="8x8-base")
        self.assertEqual(agent.problem_id, 2)
        self.assertEqual(agent.is_stochastic, True)
        self.assertEqual(agent.env.ncol, 8)
        self.assertEqual(agent.env.nrow, 8)

        # state_space_locations, state_space_actions, state_initial_id, state_goal_id, states_indexes
        self.assertFalse('S_0_0' in agent.env_mapping[0])
        self.assertTrue('S_0_1' in agent.env_mapping[0])
        self.assertEqual('S_0_2', agent.env_mapping[2])   

        # agent.env.render()
        # print(agent.env.s)
        # print(agent.env.ncol)
        # print(agent.env.nrow)
        # // Floor division - division that results into whole number adjusted to the left in the number line
        # print(agent.env.s // agent.env.ncol, agent.env.s % agent.env.ncol)
        # print(agent.env.desc.shape)
        # print(agent.env.desc.tolist())

if __name__ == '__main__':
    unittest.main()