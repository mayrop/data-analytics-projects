"""
    University of Glasgow 
    Artificial Intelligence 2018-2019
    Assessed Exercise 
"""
import sys
from helpers import compare_utils, parse_args, plot_eval
from agents import RandomAgent, ReinforcementLearningAgent, PassiveAgent, SimpleAgent

def main(args):

    problem_ids, episodes, grid = parse_args(args)

    for problem_id in problem_ids:
        # this seed doesn't work... if needed, change seed to True below
        random_agent = RandomAgent(problem_id=problem_id, map_name_base=grid) 
        random_agent.solve(episodes=episodes, seed=None)
        random_agent.evaluate(episodes)

        simple_agent = SimpleAgent(problem_id=problem_id, map_name_base=grid) 
        simple_agent.solve(episodes=episodes)
        simple_agent.evaluate(episodes)

        rl_agent = ReinforcementLearningAgent(problem_id=problem_id, map_name_base=grid) 
        rl_agent.solve(episodes=episodes)
        rl_agent.evaluate(episodes)

        passive_agent = PassiveAgent(problem_id=problem_id, map_name_base=grid) 
        passive_agent.solve()
        passive_agent.evaluate(episodes)

        # Adding the plots for evaluation
        labels = ['Episodes', 'Mean Reward']
        agents = {
            'random': random_agent,
            'simple': simple_agent,
            'rl': rl_agent
        }

        title = 'Problem {}. Episodes vs Mean Reward Plot'.format(problem_id)

        filename = '{}_{}_first_1000_training'.format(problem_id, random_agent.env.ncol)
        subtitle = 'First 1000 Episodes vs Mean Reward (Training Phase)'
        plot_eval(agents, range(999), labels, filename, title, subtitle)

        filename = '{}_{}_training'.format(problem_id, random_agent.env.ncol)
        subtitle = 'Episodes Number vs Mean Reward (Training Phase)'
        plot_eval(agents, range(episodes), labels, filename, title, subtitle)

        filename = '{}_{}_first_1000_evaluation'.format(problem_id, random_agent.env.ncol)
        subtitle = 'First 1000 episodes vs Mean Reward (Evaluation Phase)'        
        plot_eval(agents, range(999), labels, filename, title, subtitle, training=False)

        filename = '{}_{}_evaluation'.format(problem_id, random_agent.env.ncol)
        subtitle = 'Episodes Number vs Mean Reward (Evaluation Phase)'        
        plot_eval(agents, range(episodes), labels, filename, title, subtitle, training=False)

        compare_utils(passive_agent.U, rl_agent.U)

if __name__ == '__main__':
    main(sys.argv[1:])

