import sys
from helpers import *
from agents import *
from utils import print_table

print('Number of arguments:', len(sys.argv), 'arguments.')
print('Argument List:', str(sys.argv))

def main(argv):
    """Main Program."""
    if len(argv) < 1:
        print("usage: run_rl.py <problem_id> <episodes=10000>")
        exit()

    problem_id = int(argv[0])
    episodes = 10000

    if len(argv) > 1 and str.isdigit(argv[1]):
        episodes = int(argv[1])

    print('Solving with Simple Agent')
    print('Problem: ', problem_id)
    print('Episodes that will run...: ', episodes)
    print("\n\n")

    agent = SimpleAgent(problem_id=problem_id) 
    agent.solve(episodes=episodes)
    policy = agent.policy        
    arrows = policy_to_arrows(policy, 8, 8)
    
    agent.env.reset()
    print("This is the environment: ")
    print(agent.env.render())
    print("This is the final policy: ")    
    print_table(arrows)
    agent.write_eval_files()

if __name__ == '__main__':
    main(sys.argv[1:])

