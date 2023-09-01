# from env import environment
from collections import defaultdict
import numpy as np
import gymnasium
from plotting import *
import env as maBandaWorld
import random
#from tqdm import tqdm
from matplotlib import pyplot as plt


#expsil Maxmimzing the number of states visited or the next state transition, action/state joint distribution
class ThompsonSampling:
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
        samples_list = [np.random.beta(1 + reward_arr[i][0], 1 + reward_arr[i][1]) for i in range(len(reward_arr))]
        #print(reward_arr, "reward array")
        #print(samples_list)
        #print(np.argmax(samples_list), "selected action")
        #print(self.best_arm, "best arm")
        return np.argmax(samples_list)

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


        for episode_idx in range(num_episodes):
            
            observation = self.env.env.reset()
            done = False
            time = 1
            
            while not done:
                for i in range(step_count):
                    
                    # print(len(statistics.episode_rewards))
                    action = self.thompson_policy(data)#self.thompson_policy(statistics.episode_rewards, numBandits)
                    
                    next_observation, reward, done, _ = self.env.env.step(action)
                    armPulled[action] += 1
                    if reward > 0:
                        data[action, 0] += 1 #should not be adding reward
                    else:
                        data[action, 1] += 1 #should not be adding reward

                                    
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

        # regret = np.zeros(num_episodes*step_count)
        # maxR = np.max(self.env.env.getRDist())
        # for i in range(num_episodes):
        #     regret[i] = ((i+1)*maxR) - np.sum(rewards[:i+1])
    
            


        # print("max R: " + str(maxR))
        # plot_episode_stats(statistics, "Thompson Sampling", " ", "b")
        # X = [i for i in range(num_episodes*step_count)]
        # plotPointGraph(X, regret, "Step", "Cumulative Regret", "Thompson Sampling Cumulative Regret", figsize=(50, 50))
        # plt.show()
        return self.running_regret


    def plots(self):

        #regret plot, shows us how close our reward is to actual max reward possible of that step
        #x wlil be steps; y will be regret of each step
        fig5 = plt.figure(figsize = (50,50))
        plt.plot(self.steps, self.running_regret)
        plt.xlabel("Time steps")
        plt.ylabel("Regret")
        plt.title("Time steps vs. Regret")


        plt.show()

        return fig5

# def plotPointGraph(X, Y, xlabel, ylabel, title, figsize=(50, 50)):
#     fig = plt.figure(figsize=figsize)
#     plt.plot(X, Y)#, color = 'blue', width =.4)
#     plt.xlabel(xlabel)
#     plt.ylabel(ylabel)
#     plt.title(title)
#     return fig