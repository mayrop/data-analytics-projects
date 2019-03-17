import sys
from helpers import compare_utils, parse_args
from agents import ReinforcementLearningAgent, PassiveAgent

def main(args):
    """Main Program."""
    problem_ids, episodes, grid = parse_args(args)

    for problem_id in problem_ids:
        agent = ReinforcementLearningAgent(problem_id=problem_id, map_name_base=grid) 
        agent.solve(episodes=episodes)
        agent.evaluate(episodes)

        passive_agent = PassiveAgent(problem_id=problem_id, map_name_base=grid) 
        passive_agent.solve()
        passive_agent.evaluate(episodes)

        compare_utils(passive_agent.U, agent.U)

if __name__ == '__main__':
    main(sys.argv[1:])

