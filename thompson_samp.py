# from env import environment
from collections import defaultdict
import numpy as np
import gymnasium
import plotting
import env as maBandaWorld
import random

#expsil Maxmimzing the number of states visited or the next state transition, action/state joint distribution
class ThompsonSampling:
    def __init__(self, env, seed=42):
        np.random.seed(seed)
        self.env = env

    def thompson_policy(self,reward_arr,n_bandits):
        samples_list = []
        
        success_count = reward_arr.sum(axis=1)
        failure_count = reward_arr.shape[1] - success_count
                    
        samples_list = [np.random.beta(1 + success_count[bandit_id], 1 + failure_count[bandit_id]) for bandit_id in range(n_bandits)]
                                
        return np.argmax(samples_list) 

    def train(self, num_episodes, step_count):
        # Initialize Q-value function and episode statistics
        statistics = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
        numBandits = self.env.env.action_space.n
        


        for episode_idx in range(num_episodes):
            if (episode_idx + 1) % 10 == 0:
                print("\nEpisode {}/{}".format(episode_idx + 1, num_episodes))
            
            observation = self.env.env.reset()
            done = False
            time = 1
            
            while not done:
                for i in range(step_count):
                    action = self.thompson_policy(statistics.episode_rewards,numBandits)
                    
                    next_observation, reward, done, _ = self.env.env.step(action)
                                    
                    # Update episode statistics

                    statistics.episode_rewards[episode_idx] += reward
                    statistics.episode_lengths[episode_idx] = time

                    if done:
                        done = True
                    else:
                        observation = next_observation
                        time += 1
        
        return statistics
