# from env import environment
from collections import defaultdict
import numpy as np
import gym

np.random.seed(42)

env = gym.make('LunarLander-v2')

def max_ent_policy(Q):
    def policy_fn(observation):
        
        return 
    return policy_fn
    
def run_max_ent_policy():
    Q = defaultdict(lambda: np.zeros(env.action_space.n))
