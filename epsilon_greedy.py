import numpy as np
import gymnasium as gym
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import env as maBanditWorld

class epsilon_greedy:
    #why does init need __? python needs it
    def __init__(self, n_actions, env, seed):
        self.env = env
        np.random.seed(seed)
        
        self.number_of_rows = 50
        self.table_of_avg_rewards = np.zeros((self.number_of_rows, n_actions))

        #self.average_reward = [0.0 for i in range(n_actions)]

        self.memory_of_each_pull = [0.0 for i in range(n_actions)]
        self.accumulated_rewards = [0.0 for i in range(n_actions)]
        self.arms_array = [i for i in range(n_actions)]
        self.history_of_pulls = []
        self.steps = []
        self.running_avg = []
        self.history_of_action_distributions = []
        self.running_regret = []
        self.optimal_avg_running_reward = []

        self.sigma_regret = 0
        self.sigma_sum = 0
        self.sigma_pulls = 0

        #find the true best arm
        self.r_dist = self.env.env.getRDist()

        #self.reward = self.env.env.getReward()
        
        self.array_with_mean_reward_of_each_arm = []
        for i in range(len(self.r_dist)):
            mean_reward_of_each_arm = np.mean(self.r_dist[i])
            self.array_with_mean_reward_of_each_arm.append(mean_reward_of_each_arm)
        
        self.best_arm = np.argmax(self.array_with_mean_reward_of_each_arm)
        self.mean_reward_of_best_arm = np.max(self.array_with_mean_reward_of_each_arm)

        #print(self.mean_reward_of_best_arm)



    #under the assumption that epsilon is between 0,1
    #under the assumptiong that decay_rate is between 0,1
    #take initial epsilon and then decrease it until we are exploiting near the end of episodes
    def epsilon_decay(self, step_count, decay_rate, epsilon):
        return epsilon * ((1 - decay_rate)**(step_count))



    def epsilon_greedy_policy(self, n_actions, epsilon):
        def policy(observation):
            #exploration; outputs even distribution of all actions then we can select
            if epsilon > np.random.rand(): #rand generates a float
                distribution = [1.0/n_actions for i in range(n_actions)]
                return distribution
            
            #exploitation: figuring out what the best action is and then making sure that it's a peak so the program picks it
            else:
                distribution = [0.0 for i in range(n_actions)]
                best_action_idx = np.argmax(self.table_of_avg_rewards[observation]) #gets the index of the largest value in accumulated_rewards
                #print(best_action_idx)
                distribution[best_action_idx] += 1.0
                return distribution
        return policy



    def train(self, num_episodes, decay_rate, epsilon, step_count):

        numActions = self.env.env.action_space.n
        
        for episode_idx in range(num_episodes):        
            print("\nEpisode {}/{}".format(episode_idx + 1, num_episodes)) 

            observation = self.env.env.reset() #why 2 env?
            done = False
            time = 1

            while not done:
                for i in range(step_count):
                    #choose an action using greedy-epsilon policy
                    
                    decayed_epsilon = self.epsilon_decay(step_count=i+1, decay_rate=decay_rate, epsilon=epsilon) #decays within episodes not out of
                    policy = self.epsilon_greedy_policy(n_actions=numActions, epsilon=decayed_epsilon) #epsilon itself outputs a linear regret curve
                    action_distribution = policy(observation)

                    action = np.random.choice(np.arange(len(action_distribution)), p=action_distribution)

                    next_observation, reward, done, _ = self.env.env.step(action)
                    
                    #appending action to history of action distributions for plotting later
                    self.history_of_action_distributions.append(action_distribution)

                    #adding reward to its slot
                    
                    self.accumulated_rewards[action] += reward

                    self.table_of_avg_rewards[observation][action] += reward

                    #when a specific arm is pulled, add that number to that index
                    self.memory_of_each_pull[action] += 1

                    #updating average_reward
                    self.table_of_avg_rewards[observation][action] = self.table_of_avg_rewards[observation][action]/self.memory_of_each_pull[action]

                    #self.average_reward[action] = self.accumulated_rewards[action]/self.memory_of_each_pull[action]

                    #updating sigma_pulls
                    self.sigma_pulls += 1

                    #updating sigma_sum
                    #reward can be negative
                    self.sigma_sum += reward

                    #updating memory
                    self.history_of_pulls.append(action)

                    #updating steps
                    self.steps.append(len(self.steps))

                    #updating running avg reward for each step
                    self.running_avg.append(self.sigma_sum/self.sigma_pulls)

                    # finds mean_reward so we can calculate regret
                    # mean_reward_of_selected_arm = np.mean(self.r_dist[action])

                    #print(self.table_of_avg_rewards[observation][action])
                    #print(self.table_of_avg_rewards[observation]) #should be printing out a row
                    #print(self.table_of_avg_rewards[:, action]) #should print out the col

                    
                    mean_reward_of_selected_arm = np.mean(self.r_dist[action])
                    regret_of_step = self.mean_reward_of_best_arm - mean_reward_of_selected_arm
                    self.sigma_regret += regret_of_step

                    self.running_regret.append(self.sigma_regret)


                    #self.optimal_avg_running_reward.append(self.mean_reward_of_best_arm)

                    
                    #debug
                    print("step {}/{}".format(i, step_count))
                    #print("decayed epsilon: " + str(decayed_epsilon))
                    #print("action distribution: " + str(action_distribution))
                    #print("History of action distrbution: " + str(self.history_of_action_distributions))
                    #print("Observation: " + str(observation))
                    #print("Sigma sum: " + str(self.sigma_sum))
                    #print("Reward: " + str(reward))
                    #print("best arm {} ___ Mean_reward_of_best_arm {} ".format(self.best_arm, self.mean_reward_of_best_arm))
                    #print("selected arm: {} ___ mean_reward_of_selected_arm {}".format(action, mean_reward_of_selected_arm))
                    #print("average reward array: \n" + str(self.average_reward))
                    #print(self.sigma_regret)
                    #print(self.running_regret)
                    #print(self.history_of_pulls)
                    #print("Regret2: " + str(np.max(self.env.env.getRDist())))


                    if done:
                        done = True
                    else:
                        observation = next_observation
                        time += 1

        return self.running_regret


    def plots(self):
        #normalizing bar graph
        for i in range(len(self.memory_of_each_pull)):
            self.memory_of_each_pull[i] = self.memory_of_each_pull[i]/self.sigma_pulls

        #bar graph
        #x will be 0,1,2,3,,4,5, num of arms; y will be num of pulls
        fig1 = plt.figure(figsize = (50,50))
        plt.bar(self.arms_array, self.memory_of_each_pull, color = 'blue', width =.4)
        plt.xlabel("Arm")
        plt.ylabel("Probability")
        plt.title("Probability per arm")

        #converging dots plot arm pulledv. steps
        #x will be steps; y will be num of arms
        fig2 = plt.figure(figsize = (50,50))
        plt.plot(self.steps, self.history_of_pulls)
        plt.xlabel("Time steps")
        plt.ylabel("Arm pulled")
        plt.title("Arm pulled at each time step")

        #optimal reward
        fig3 = plt.figure(figsize = (50,50))
        plt.plot(self.steps, self.optimal_avg_running_reward)
        plt.xlabel("Time steps")
        plt.ylabel("Optimal running avg reward")
        plt.title("Optimal average running reward")

        #continous plot timestepv. reward
        #x will be steps; y will be avg reward
        fig4 = plt.figure(figsize = (50,50))
        plt.plot(self.steps, self.running_avg)
        plt.xlabel("Time steps")
        plt.ylabel("Running Average")
        plt.title("Running average at each time step")

        #regret plot, shows us how close our reward is to actual max reward possible of that step
        #x wlil be steps; y will be regret of each step
        fig5 = plt.figure(figsize = (50,50))
        plt.plot(self.steps, self.running_regret)
        plt.xlabel("Time steps")
        plt.ylabel("Regret")
        plt.title("Time steps vs. Regret")


        plt.show()

        return fig1, fig2, fig4, fig5