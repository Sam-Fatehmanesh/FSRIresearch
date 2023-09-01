# from env import environment
from collections import defaultdict
import numpy as np
import gymnasium
from plotting import *
import env as maBandaWorld
import random
from tqdm import tqdm
from matplotlib import pyplot as plt
from scipy.linalg import fractional_matrix_power



#expsil Maxmimzing the number of states visited or the next state transition, action/state joint distribution
class OptimisticThompsonSamplingFO:
    def __init__(self, env, seed=42):
        np.random.seed(seed)
        self.env = env

    def thompson_policy(self, reward_arr, V_t, param_hat_t):

        # mean = np.array([reward_arr[i][0]/(reward_arr[i][0] + reward_arr[i][1]) for i in range(len(reward_arr))])

        # samples_list = [0 for i in range(len(reward_arr))]

        # for i in range(len(reward_arr)):
        #     sample = np.random.beta(1 + reward_arr[i][0], 1 + reward_arr[i][1])
        #     if  sample > mean[i]:
        #         samples_list[i] = sample
        #         mean[i] = sample
        #     else:
        #         samples_list[i] = mean[i]
                
        # return np.argmax(np.array(samples_list))
        v_negrad = fractional_matrix_power(V_t, -0.5)
        
        samples_list = param_hat_t + (np.matmul(np.array([np.random.beta(1 + reward_arr[i][0], 1 + reward_arr[i][1]) for i in range(len(reward_arr))]), v_negrad) * np.random.normal())

        # print("v_negrad")
        #print((np.matmul(np.array([np.random.beta(1 + reward_arr[i][0], 1 + reward_arr[i][1]) for i in range(len(reward_arr))]), V_t) * np.random.normal()))
        # print("param_hat_t")
        # print(param_hat_t)
        # print("samples_list")
        # print(samples_list)
        return np.argmax(samples_list)

    def train(self, num_episodes, step_count, lambdaconst, graph_color):
        # Initialize Q-value function and episode statistics
        statistics = EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        step_reward_avg=np.zeros(num_episodes*step_count)) #added this
        numBandits = self.env.env.action_space.n
        data = np.zeros((numBandits, 2))
        rewards = np.zeros(num_episodes*step_count)
        armPulled = np.zeros(numBandits)
        armPulled_t = []
        V_t = np.identity(n=numBandits) * lambdaconst
        param_hat_t = np.zeros(numBandits)
        total_arm_r = np.zeros(numBandits)
        self.sigma_regret = 0

        self.array_with_mean_reward_of_each_arm = []
        for i in range(len(self.env.env.r_dist)):
            mean_reward_of_each_arm = np.mean(self.env.env.r_dist[i])
            self.array_with_mean_reward_of_each_arm.append(mean_reward_of_each_arm)
        
        self.best_arm = np.argmax(self.array_with_mean_reward_of_each_arm)
        self.mean_reward_of_best_arm = np.max(self.array_with_mean_reward_of_each_arm)
        self.running_regret = []
        self.steps = []
        #self.maxreward = np.max(env

        #optimistic_mean = np.ones(numBandits) * (-100000)
        
        


        for episode_idx in tqdm(range(num_episodes), leave=True):
            
            observation = self.env.env.reset()
            done = False
            time = 1
            action_reward_product_sum = np.zeros(numBandits)
            
            while not done:
                for i in tqdm(range(step_count), leave=True):
                    
                    # print(len(statistics.episode_rewards))
                    action = self.thompson_policy(data, V_t, param_hat_t)#, optimistic_mean)#self.thompson_policy(statistics.episode_rewards, numBandits)
                    action_vec = np.zeros(numBandits)
                    action_vec[action] += 1
                    armPulled_t.append(action)
                     
                    
                    
                    next_observation, reward, done, _ = self.env.env.step(action)
                    total_arm_r[action] += reward
                    armPulled[action] += 1
                    if reward > 0:
                        data[action, 0] += 1
                    else:
                        data[action, 1] += 1

                    V_t += np.matmul(action_vec, np.transpose(action_vec))
                    param_hat_t = np.matmul(np.linalg.inv(V_t), action_reward_product_sum)
                    action_reward_product_sum += action_vec * reward
                    # print(action_reward_product_sum)R
                                    
                    # Update episode statistics
                    rewards[episode_idx*step_count + i] = reward

                    statistics.episode_rewards[episode_idx] += reward
                    statistics.episode_lengths[episode_idx] = time
                    statistics.step_reward_avg[episode_idx*step_count + i] = np.sum(rewards)/(episode_idx*step_count + i + 1)

                    mean_reward_of_selected_arm = np.mean(self.env.env.r_dist[action])
                    regret_of_step = self.mean_reward_of_best_arm - mean_reward_of_selected_arm
                    self.sigma_regret += regret_of_step

                    self.running_regret.append(self.sigma_regret)
                    self.steps.append(len(self.steps))

                    if done:
                        done = True
                    else:
                        observation = next_observation
                        time += 1

        # cum_regret = np.zeros(num_episodes*step_count)
        # r_dist = self.env.env.getRDist()
        # p_dist = self.env.env.p_dist
        # R = [r_dist[i][0] * p_dist[i] for i in range(numBandits)]
        # # print("R")
        # # print(R)
        # regret = np.zeros(num_episodes*step_count)

        # maxR = np.max(R)
        # for i in tqdm(range(num_episodes*step_count)):
        #     cum_regret[i] = ((i+1)*maxR) - R[armPulled_t[i]]
        #     regret[i] = maxR - R[armPulled_t[i]]




        # print("max R: " + str(maxR))
        # plot_episode_stats(statistics, "Thompson Sampling", " ", "b")
        # X = [i for i in range(num_episodes*step_count)]
        # plotPointGraph(X, cum_regret, "Step", "Cumulative Regret", "Thompson Sampling Cumulative Regret", graph_color)
        # plotPointGraph(X, regret, "Step", "Regret", "Thompson Sampling Regret", graph_color)
        # plt.show()
        return self.running_regret

def plotPointGraph(X, Y, xlabel, ylabel, title, color, figsize=(50,50)):
    fig = plt.figure(figsize=figsize)
    plt.plot(X, Y, color = color)#, color = 'blue', width =.4)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    return fig