import numpy as np
import gym
from gym import spaces
from gym.utils import seeding

class BitFlipperEnv(gym.Env):
    '''Bit Flipping environment
       The state space is binary strings of length n.
       The action space is an index i from {0,1...n-1} which represents the agent flipping ith bit of the environment.
       Action = n means agent stays in the same state.
       Given an initial state the agent has to reach a goal state.
       Reward: Only goal state has reward 0,rest all states have reward -1
    '''
    metadata = {'render.modes': ['human','ansi']}
    def __init__(self,n=10,space_seed=0):
        self.n=n        
        self.action_space = spaces.Discrete(self.n +1) #there is an option NOT to flip any bit( index = n)
        self.observation_space = spaces.MultiBinary(self.n)
        self.state_space = spaces.MultiBinary(self.n)
        self.reward_range = (-1,0)
        spaces.seed(space_seed)
        self.space_seed = space_seed
        self.initial_state = self.sample_state()
        self.goal = self.sample_state()
        self.state = self.initial_state
        self.envstepcount = 0
        self.seed()
        self.reward_max = -np.sum(np.bitwise_xor(self.initial_state,self.goal))+1
        if(np.array_equal(self.goal,self.initial_state)):
             self.reward_max = 0

    def sample_state(self):
        return self.state_space.sample()
    
    def step(self,action):
        '''
         accepts action and returns obs,reward, b_flag(episode start), info dict(optional)
        '''
        if(self.action_space.contains(action)):
            if(not(action==self.n)):
                self.state = self.bitflip(action)    ## computes s_t1
            reward = self.calculate_reward(self.goal,self.state)
            self.envstepcount += 1
            done = self.compute_done(reward)
            return (self.state,reward,done,{})
        else:
             print("Invalid action")
                
    def reset(self,seed=None):    
        if seed == None:
            seed = self.space_seed
        self.envstepcount = 0
        # Uncomment for fixing state-goal
        # spaces.seed(seed)
        self.initial_state = self.sample_state()
        #self.goal = self.sample_state()
        self.state = self.initial_state
        self.reward_max = -np.sum(np.bitwise_xor(self.initial_state,self.goal))+1
        if(np.array_equal(self.goal,self.initial_state)):
             self.reward_max = 0
        return self.state
    
    def close(self):
        pass
    
    def render(self, mode='human', close=False):
        print_str = str("State: "+str(self.state.T)+" Steps done: "+str(self.envstepcount))
        if(mode=='human'):
            print(print_str)
        if(mode=='ansi'):
            return print_str 
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def bitflip(self,index):
        s2=np.array(self.state)
        s2[index] = not s2[index]
        return s2
    
    def calculate_reward(self,goal,state):
        if(np.array_equal(goal,state)):
            return 0.0
        else:
            return -1.0
        
    def compute_done(self,reward):
        if(reward==0 or self.envstepcount >=self.n):
            return True
        else:
            return False
    
class GoalBitFlipperEnv(BitFlipperEnv):
    def __init__(self, n=80, space_seed=None):
        super().__init__(n, space_seed)
        self.observation_space = spaces.MultiBinary(self.n*2)
    
    def step(self, action):
        obs, reward, done, info = super().step(action)
        return self._convert_observation(obs, self.state, self.goal), reward, done, info
    
    def reset(self, seed=None):
        obs = super().reset()
        return self._convert_observation(obs, self.state, self.goal)

    def compute_reward(self, state, goal):
        return super().calculate_reward(state, goal)

    def change_obs_goal(self, obs, goal):
        return np.concatenate([obs[:self.state.shape[0]], goal])

    def _convert_observation(self, obs, state, goal):
        return {'observation': obs, 'achieved_goal': state, 'desired_goal': goal}
