# Artificial Intelligence (H) 2018-2019
## Assessed Exercise: Individual, 20% of the final grade (∼ 20 hours)

### Problem Statement
Your task is to design, implement, evaluate and document three virtual agents which are (potentially) able to reach a goal in a custom Open AI Gym environment derived from Frozen. Thus, you will need
to install and understand the workings of the Open AI Gym environment to be able solve the task (hint: see AI (H) Lab 2).

The specific environment/problem under consideration is a grid-world with a starting position (S), obstacles (H) and a final goal (G). The task is to get from S to G. The environment is defined in `uofgsocsai.py`. via the class `LochLomondEnv` including documentation relating to the setting, specific states, parameters etc. An example of how to instantiate the environment and navigate it using random actions is provided in `lochlomond_demo.py`.

You must consider three agent types: a senseless/random agent, a simple agent and a reinforcement agent based on the requirements listed in the following sections. Your agents will be tested against other instances of the same problem type, i.e., you can not (successfully) hard-code the solution. You will have access to eighth specific training instances of the environment determined by the value of a single variable `problem_id`.

Your agents and findings should be documented in a short (max 1500 word) technical report accompanied by the actual implementation/code and evaluation scripts.

### Tasks
You should provide a design, implementation, evaluation and documentation of three different agents; each with its own set of specific requirements:

#### Task I: Senseless/Random agent
You should provide a solution for an agent without sensory input which takes random actions. You should initialise the environment as follows:

`env = LochLomondEnv(problem_id=problem_id, is_stochastic=True, reward_hole=0.0)`

where you can use problem_id ∈ [0:7] to evaluate the performance over different instances of the same problem.

**Purpose**: This agent should be used as a naive baseline. Hint: A basic senseless/random agent is already partly provided in lochlomond_demo.py (albeit with output computation of the performance measure...).

**Requirements**:  
-**Sensors**: None (/random/full; it doesn’t matter...)  
-**Action**: Discrete  
-**State-space**: No prior knowledge (i.e. it has not got a map)  
-**Rewards/goal**: No prior knowledge (does not know where the goal is located)  

#### Task II: Simple Agent
You should provide an agent based on a tree/graph-search and justify its use for solving the particular the problem (e.g. using A* search with a suitable heuristic) assuming that the task environment is fully known and observable. You should initialise the environment as follows:
env = LochLomondEnv(problem_id=[0-7], is_stochastic=False, reward_hole=0.0)
where you can use problem_id ∈ [0:7] to evaluate the performance over different instances of the same problem - and to fine-tune your agent to make sure it generalises. We recommend you use existing code (from e.g. the AIMA toolbox) to solve this part).

**Purpose**: This agent is used as an ideal baseline to find the optimal path under ideal circumstances. Hint: if you have attended the Lab sessions you will easily be able to reuse most of the code to solve this part (a parser that maps from env.desc to the lab 3 format will be made available).

**Requirements:**  
-**Sensors**: Oracle (i.e. you’re allowed to read the location of all object in the environment e.g. using env.desc)  
-**Actions**: Discrete and noise-free.  
-**State-space**: Fully observable a priori.  
-**Rewards/goal**: Fully known a priori (you are allowed to inform the problem with the rewards and location of terminal states)  

#### Task III: Reinforcement learning agent
You should provide a reinforcement learning agent to solve the problem with minimal assumptions about the environment (see below). The choice of the RL agent is up to you but we highly recommend using tabular Q-learning. Regardless, the choice should be well-justified and meet the list of requirements listed below. You should instantiate the environment as follows:

`env = LochLomondEnv(problem_id=[0-7], is_stochastic=True, reward_hole=[YOUR-CHOICE])`

where you can use problem_id ∈ [0:7] to evaluate the performance over different instances of the problem - and to fine tune your agent to make sure it generalises. You are encouraged to write your own code but use of 3rd party implementations (e.g. the AIMA toolbox) is allowed, if you demonstrate a sufficient understanding of the algorithms in the accompanying documentation and cite appropriately.

**Requirements:**  
-**Sensors**: Perfect information about the current state and thus available actions in that state; no prior  
knowledge about the state-space in general  
-**Action**: Discrete and noisy. The requested action is only carried out correctly with a certain probability  (as defined by uofgsocsai.py).
-**State-space**: No prior knowledge, but partially observable/learn-able via the sensors/actions.  
-**Rewards**: No prior knowledge, but partially observable via sensors/actions.  

**Notice**: You can restart and replay the same instance of the problem multiple times and maintain the knowledge obtained across repetitions/episodes. The reward should in this case be reported as a) the average across all restarts/repetitions and b) the single best policy (typically after a long learning phase).

### Submission
You should include three items in your submission:

##### Implementation & Code
Your implementation - containing the three different agents (along with any dependencies, except the Open AI Gym) - should be uploaded to Moodle as a zip-file containing the source code. You must provide three separate and executable python scripts/programs named: `run_random.py`, `run_simple.py`
and `run_rl.py` which takes as (only required) argument the `problem_id` . Each script/program should include training/learning phases (including possible repetitions/episodes of the problem) and output a text file (named after the agent, e.g. "random") with any relevant information. Hint: A template will be provided via Moodle.

##### Experiment & Evaluation
An important aspect of RL is assessing and comparing the performance of agents and different policies.

To document the behavior of your agents you should design a suitable set of (computer) experiments which produces a relevant set of graphs/tables to document the behavior of your RL agent (e.g. average performance measure vs number of episodes, etc) and compares its performance against the baselines. 
The evaluation strategy should be implemented in a single Python script (a wrapper) run_eval.py which runs you entire evaluation, that is, it should call your agents, collate the results and produce the figures/tables you have included in your report. The `run_eval.py` should be submitted via Moodle alongside your implementation.

### Report
You should document your work and results in a short technical report of (max 1500 words; excluding figures, tables, captions, references and appendices). Appendices may be used to provide extra information to support the data and arguments in the main document, e.g., detailed simulation results but should not provide crucial information required to understand the principle of the solution and the outcome. You can include as many references as you see fit. The report should be submitted via Moodle as a pdf file alongside your implementation and evaluation scripts.

### Marking Scheme
The assessment is based on the degree to which your submission (implementation, evaluation script and report) concisely, correctly and completely addresses the following aspects:

#### Analysis [15%]
Introduction/motivation and correct PEAS analysis (including task environment characterisation).

#### Method/design [15%]
Presentation of relevant theory and methods (for all your agents) with proper use of citations and justification for choosing specific methods (referencing your analysis).

#### Implementation [25%]
The code for all the agents (not in the report!) should be well-documented, follow best-practices in software development and follow the outlined naming convention. The report must contain a presentation of relevant aspects of the implementation.

#### Experiments / Evaluation
- [20%] Evaluation script/program to reproduce the results (i.e. graphs/tables) adhering to the specified requirements.
- [20%] Relevant presentation of the evaluation strategy, metrics and the obtained simulation results. A suitable presentation and comparison of the performance of the agent with other agents as evaluated across a suitable number of problem variations (e.g. using graphs/tables).

#### Discussion and conclusion [5%]
- Including a critical reflections on the outcome/results. The weighting of the senseless, simple and RL agent - for all marking criterion - is 5,10 and 85%, respectively.

### Collaboration
This is an individual exercise, but discussions related to the general aspects of the problem are perfectly fine. The final solution must, however, reflect an individual effort.