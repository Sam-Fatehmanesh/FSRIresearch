# from env import environment
from collections import defaultdict
import numpy as np
import gym
import plotting
import env as rlworld
import random

#expsil
class ThompsonSampling:
    def __init__(self, env, n_bandits, seed=42):
        np.random.seed(seed)
        self.n_bandits = n_bandits
        self.env = env

    def thompson_policy(self,reward_arr, n_bandits):
        k_arr = list(range(1,n_bandits))

        successes = reward_arr.sum(axis=1)
        failures = k_arr.sum(axis=1) - successes
                    
        samples_list = [np.random.beta(1 + successes[bandit_id], 1 + failures[bandit_id]) for bandit_id in range(n_bandits)]
                                
        return np.argmax(samples_list)    

    def train(self, num_episodes):
        # Initialize Q-value function and episode statistics
        episode_lengths = np.zeros(num_episodes)
        episode_rewards = np.zeros(num_episodes)

        for episode_idx in range(num_episodes):
            if (episode_idx + 1) % 10 == 0:
                print("\nEpisode {}/{}".format(episode_idx + 1, num_episodes))
            
            observation = self.env.reset()
            done = False
            time = 1
            
            while not done:
                action = self.thompson_policy(episode_rewards, self.n_bandits)
                
                next_observation, reward, done, _ = self.env.step(action)
                                
                # Update episode statistics
                episode_rewards[episode_idx] += reward
                episode_lengths[episode_idx] = time
                

                if done:
                    done = True
                else:
                    observation = next_observation
                    time += 1
        
        return episode_lengths, episode_rewards
