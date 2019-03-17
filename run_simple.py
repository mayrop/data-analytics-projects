import sys
from helpers import parse_args
from agents import SimpleAgent

def main(args):
    problem_ids, episodes, grid = parse_args(args)        

    for problem_id in problem_ids:
        agent = SimpleAgent(problem_id=problem_id, map_name_base=grid) 
        agent.solve(episodes=episodes)
        agent.evaluate(episodes)

if __name__ == '__main__':
    main(sys.argv[1:])

