import numpy as np
import sys
from six import StringIO, b

from gym import utils
from gym.envs.toy_text import discrete

LEFT = 0
DOWN = 1
RIGHT = 2
UP = 3

MAPS_BASE = {
    "4x4-base": [
        "SFFF",
        "FHFH",
        "FFFH",
        "HFFF"
    ],  
    "8x8-base": [
        "HFFFFHFF",
        "FFFFFFFF",
        "FFFHFFFF",
        "FFFFFHFF",
        "FFFHFFFF",
        "FHHFFFHF",
        "FHFFHFHF",
        "FFFHFFFH"
    ],
}

class LochLomondEnv(discrete.DiscreteEnv):
    """
    This environment is derived from the FrozenLake from https://gym.openai.com/envs/#toy_text

    Winter is in Scotland. You and your friends were tossing around a frisbee at Loch Lomond
    when you made a wild throw that left the frisbee out in the middle of the lake.
    The water is mostly frozen, but there are a few holes where the ice has melted.
    If you step into one of those holes, you'll fall into the freezing water.
    At this time, there's an international frisbee shortage, so it's absolutely imperative that
    you navigate across the lake and retrieve the disc.
    
    However, the ice is slippery, so you won't always move in the direction you intend.
    The surface is described using a grid like the following (for 8x8) with a unknown `problem_id`:

        HFFFFSFFF"
        "FFFFFFFF"
        "FFFHFFFF"
        "FFFFFHFF"
        "FFFHFFFF"
        "FHHFFFHF"
        "FHFFHFHF"
        "FGFHFFFH"

    S : starting point, safe
    F : frozen surface, safe
    H : hole, fall to your doom (notice: the environment doesn't actuall return negative reward in this case; 
        depending on your approach this might be something to watch out for...)
    G : goal, where the frisbee is located
    X : Shows you where your at a given point (when running/rendering the env)

    The episode ends when you reach the goal or fall in a hole 
    (ends means that env.step will return "done=True"; you will 
    still be able to render the env). Falling in a hole is not fatal, 
    it just means you need to get up and get dry and warm and can't 
    reach the goal in this episode). 

    The rewards from the env are defined as follows:
        - you receive a reward of +1.0 if you reach the goal, 
        - you receive a reward of reward_hole (<0) if you reach the goal (you migth want to set the reward_hole as part of your algorithm design/evaluation) 
        - and zero otherwise (i.e. there is no cost of "living" in the env).

    """

    metadata = {'render.modes': ['human', 'ansi']}

    def __init__(self, problem_id=0, is_stochastic=True, reward_hole = 0.0):
        if reward_hole > 0.0:
            raise ValueError('reward_hole must be equal to 0 or smaller')
    
        # Fetch the base problem (without S and G)
        map_name_base="8x8-base" # for the final submission in AI (H) this should be 8x8-base but you may want to start out with 4x4-base!        
        desc = MAPS_BASE[map_name_base]
        self.nrow, self.ncol = nrow, ncol = np.asarray(desc,dtype='c').shape

        # Check probelm_id value
        if problem_id > ncol-1:
            raise ValueError("problem_id must be in 0:"+str(ncol-1))

        # Seed the random for the problem 
        np.random.seed(problem_id)

        # Set the Start state for this variant of the problem     
        row_s = 0
        col_s = problem_id 
        desc[row_s] = desc[row_s][:col_s] + 'S' + desc[row_s][col_s+1:]
        
        # Set the Goal state for this variant of the problem     
        row_g = nrow-1
        col_g = np.random.randint(0, high=ncol)
        desc[row_g] = desc[row_g][:col_g] + 'G' + desc[row_g][col_g+1:]

        self.desc = desc = np.asarray(desc,dtype='c')        
        self.reward_range = (0, 1)

        nA = 4
        nS = nrow * ncol

        isd = np.array(desc == b'S').astype('float64').ravel()
        isd /= isd.sum()

        P = {s : {a : [] for a in range(nA)} for s in range(nS)}

        def to_s(row, col):
            return row*ncol + col
        
        def inc(row, col, a):
            if a==0: # left
                col = max(col-1,0)
            elif a==1: # down
                row = min(row+1,nrow-1)
            elif a==2: # right
                col = min(col+1,ncol-1)
            elif a==3: # up
                row = max(row-1,0)
            return (row, col)

        for row in range(nrow):
            for col in range(ncol):
                s = to_s(row, col)
                for a in range(4):
                    li = P[s][a]
                    letter = desc[row, col]
                    if letter in b'GH':
                        li.append((1.0, s, 0, True))
                    else:
                        if is_stochastic:
                            for b in [(a-1)%4, a, (a+1)%4]:
                                newrow, newcol = inc(row, col, b)
                                newstate = to_s(newrow, newcol)
                                newletter = desc[newrow, newcol]
                                done = bytes(newletter) in b'GH'
                                rew = 0.0
                                if(newletter == b'G'):
                                    rew = 1.0
                                elif(newletter == b'H'):
                                    rew = reward_hole
                                li.append((1.0/3.0, newstate, rew, done))
                        else:
                            newrow, newcol = inc(row, col, a)
                            newstate = to_s(newrow, newcol)
                            newletter = desc[newrow, newcol]
                            done = bytes(newletter) in b'GH'
                            rew = 0.0
                            if(newletter == b'G'):
                                rew = 1.0       
                            elif(newletter == b'H'):
                                rew = reward_hole                     
                            li.append((1.0, newstate, rew, done))

        super(LochLomondEnv, self).__init__(nS, nA, P, isd)

    def render(self, mode='human'):
        outfile = StringIO() if mode == 'ansi' else sys.stdout
    
        row, col = self.s // self.ncol, self.s % self.ncol
        desc = self.desc.tolist()

        desc = [[c.decode('utf-8') for c in line] for line in desc]
        desc[row][col] = "X"

        #desc[row][col] = utils.colorize(desc[row][col], "red", highlight=True) # note: this does not work on all setups you can try to uncomment asnd see what happends (if it does work you'll see weird symbols)
        
        if self.lastaction is not None:
            outfile.write("  ({})\n".format(["Left","Down","Right","Up"][self.lastaction]))
        else:
            outfile.write("\n")
        outfile.write("\n".join(''.join(line) for line in desc)+"\n")

        if mode != 'human':
            return outfile
