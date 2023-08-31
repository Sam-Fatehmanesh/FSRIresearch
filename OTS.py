# from env import environment
from collections import defaultdict
import numpy as np
import gymnasium
from plotting import *
import env as maBandaWorld
import random
from tqdm import tqdm
from matplotlib import pyplot as plt


#expsil Maxmimzing the number of states visited or the next state transition, action/state joint distribution
class OptimisticThompsonSampling:
    def __init__(self, env, seed=42):
        np.random.seed(seed)
        self.env = env
        self.r_dist = self.env.env.getRDist()

        self.steps = []
        self.running_regret = []
        self.sigma_regret= 0 

        self.array_with_mean_reward_of_each_arm = []
        for i in range(len(self.r_dist)):
            mean_reward_of_each_arm = np.mean(self.r_dist[i])
            self.array_with_mean_reward_of_each_arm.append(mean_reward_of_each_arm)
        
        self.best_arm = np.argmax(self.array_with_mean_reward_of_each_arm)
        self.mean_reward_of_best_arm = np.max(self.array_with_mean_reward_of_each_arm)

    def thompson_policy(self, reward_arr):

        mean = np.array([reward_arr[i][0]/(reward_arr[i][0] + reward_arr[i][1]) for i in range(len(reward_arr))])

        samples_list = [0 for i in range(len(reward_arr))]

        for i in range(len(reward_arr)):
            sample = np.random.beta(1 + reward_arr[i][0], 1 + reward_arr[i][1])
            if  sample > mean[i]:
                samples_list[i] = sample
                mean[i] = sample
            else:
                samples_list[i] = mean[i]
                
        return np.argmax(np.array(samples_list))

    def train(self, num_episodes, step_count):
        # Initialize Q-value function and episode statistics
        statistics = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        step_reward_avg=np.zeros(num_episodes*step_count)) #added this
        numBandits = self.env.env.action_space.n
        data = np.zeros((numBandits, 2))
        rewards = np.zeros(num_episodes*step_count)
        armPulled = np.zeros(numBandits)
        #optimistic_mean = np.ones(numBandits) * (-100000)
        
        


        for episode_idx in tqdm(range(num_episodes), leave=True):
            
            observation = self.env.env.reset()
            done = False
            time = 1
            
            while not done:
                for i in range(step_count):
                    
                    # print(len(statistics.episode_rewards))
                    action = self.thompson_policy(data)#, optimistic_mean)#self.thompson_policy(statistics.episode_rewards, numBandits)
                    #print(optimistic_mean)
                    
                    # print(action) #added to understand what's happening, outputs 14131 on a run
                    
                    next_observation, reward, done, _ = self.env.env.step(action)
                    armPulled[action] += 1
                    if reward > 0:
                        data[action, 0] += reward
                    else:
                        data[action, 1] += abs(reward)

                                    
                    # Update episode statistics
                    rewards[episode_idx*step_count + i] = reward

                    statistics.episode_rewards[episode_idx] += reward
                    statistics.episode_lengths[episode_idx] = time
                    statistics.step_reward_avg[episode_idx*step_count + i] = np.sum(rewards)/(episode_idx*step_count + i + 1)
                
                    self.steps.append(len(self.steps))
                    mean_reward_of_selected_arm = np.mean(self.r_dist[action])
                    regret_of_step = self.mean_reward_of_best_arm - mean_reward_of_selected_arm
                    self.sigma_regret += regret_of_step
                    self.running_regret.append(self.sigma_regret)
                    
                    if done:
                        done = True
                    else:
                        observation = next_observation
                        time += 1

        



        #print("max R: " + str(maxR))
        #plot_episode_stats(statistics, "Thompson Sampling", " ", "b")
        X = [i for i in range(num_episodes*step_count)]
        plotPointGraph(X, self.running_regret, figsize=(50, 50))
        plt.show()
        return statistics

def plotPointGraph(X, Y, figsize=(50,50)):
    fig = plt.figure(figsize=figsize)
    plt.plot(X, Y)#, color = 'blue', width =.4)
    return fig