import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import env as maBanditWorld

class upper_confidence_bound:
    def __init__(self, n_actions, env, seed):
        self.env = env
        np.random.seed(seed)

        self.rows = 50
        self.table_of_rewards = np.zeros((self.rows, n_actions))
        #self.average_reward = [0.0 for i in range(n_actions)]

        self.memory_of_each_pull = [0.0 for i in range(n_actions)]
        #self.accumulated_rewards = [0.0 for i in range(n_actions)]
    
        self.ucb_of_each_arm = [0.0 for i in range(n_actions)]
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
        
        self.array_with_mean_reward_of_each_arm = []
        for i in range(len(self.r_dist)):
            mean_reward_of_each_arm = np.mean(self.r_dist[i])
            self.array_with_mean_reward_of_each_arm.append(mean_reward_of_each_arm)
        
        self.best_arm = np.argmax(self.array_with_mean_reward_of_each_arm)
        self.mean_reward_of_best_arm = np.max(self.array_with_mean_reward_of_each_arm)

        #print(self.mean_reward_of_best_arm)


# each arm must be played once initially

# constact (c) value of exploration exploitation trade off set by user 

# all arms must be updated after a step

    def ucb_policy(self, n_actions, c):
        def policy():
            #new_ucb_value_of_selected_arm = self.average_reward[selected_arm] + c*(np.sqrt(((2*(np.log(self.sigma_pulls)))/self.memory_of_each_pull[selected_arm])))
            best_action_idx = np.argmax(self.ucb_of_each_arm)
            distribution = [0.0 for i in range(n_actions)]
            distribution[best_action_idx] += 1.0
            for i in range(10):
                new_ucb_value_of_selected_arm = sum(self.table_of_rewards[:, i])/self.memory_of_each_pull[i] + c*(np.sqrt(((2*(np.log(self.sigma_pulls)))/self.memory_of_each_pull[i])))
                self.ucb_of_each_arm[i] = new_ucb_value_of_selected_arm
                #print(self.ucb_of_each_arm)
            #print("best action idx: " + str(best_action_idx))
            #print("ucb array: " + str(self.ucb_of_each_arm))

            return distribution
        return policy



    
    def train(self, num_episodes, step_count, c):
        numActions = self.env.env.action_space.n

        observation = self.env.env.reset()
        time = 1
        counter = 0

        #choosing each arm once
        while counter < 10:
            action = counter
            next_observation, reward, done, _ = self.env.env.step(action)

            #updating important stats
            #self.accumulated_rewards[action] += reward
            self.table_of_rewards[observation][action] += reward

            self.memory_of_each_pull[action] += 1

            #self.average_reward[action] = self.accumulated_rewards[action]/self.memory_of_each_pull[action]
            #self.table_of_rewards[observation][action] = self.table_of_rewards[observation][action]

            self.sigma_pulls += 1

            self.sigma_sum += reward

            self.history_of_pulls.append(action)

            self.steps.append(len(self.steps))

            #updating running avg reward for each step
            self.running_avg.append(self.sigma_sum/self.sigma_pulls)

            #finds mean_reward so we can calculate regret
            mean_reward_of_selected_arm = sum(self.table_of_rewards[:, action])/self.memory_of_each_pull[action]

            #calculating regret
            mean_reward_of_selected_arm = np.mean(self.r_dist[action])
            regret_of_step = self.mean_reward_of_best_arm - mean_reward_of_selected_arm
            self.sigma_regret += regret_of_step

            self.running_regret.append(self.sigma_regret)

            #intial 10 calculations of each arm's ucb value

            ucb_value_of_n_arm = sum(self.table_of_rewards[:, action]) + c*(np.sqrt(((2*(np.log(10)))/self.memory_of_each_pull[action])))

            self.ucb_of_each_arm[action] = ucb_value_of_n_arm

            #debugging
            #print("action: " + str(action))
            #print("reward: " + str(reward))
            #print("calculated ucb value: " + str(ucb_value_of_n_arm))
            #print("ucb array: " + str(self.ucb_of_each_arm))
            #print("memory of pulls array: " + str(self.memory_of_each_pull))
            #print("sigma pulls: " + str(self.sigma_pulls))

            #self.optimal_avg_running_reward.append(self.optimal_avg_running_reward)

            observation = next_observation
            time += 1
            counter += 1

        for episode_idx in range(num_episodes):        
            print("\nEpisode {}/{}".format(episode_idx + 1, num_episodes)) #episode_idx + 1 b/c we start counting from 0
            
            #the following line resets the enviornment so we transition out of the inner loop (stops us from local minimas in other things than MAB)
            observation = self.env.env.reset()
            done = False
            print("ucb array: " + str(self.ucb_of_each_arm))

            while not done:
                for i in range(step_count-10):
                    print("step {}/{}".format(i+10, step_count))
                    policy = self.ucb_policy(c=c, n_actions=numActions)
                    action_distribution = policy()

                    #self.average_reward =  [self.accumulated_rewards[i]/(self.memory_of_each_pull[i]+1) for i in range(len(self.memory_of_each_pull))]

                    action = np.random.choice(np.arange(len(action_distribution)), p=action_distribution)

                    next_observation, reward, done, _ = self.env.env.step(action)
                    
                    #adding reward to its slot
                    #self.accumulated_rewards[action] += reward
                    self.table_of_rewards[observation][action] += reward

                    #when a specific arm is pulled, add that number to that index
                    #so now we know how many times an arm has been pulled
                    self.memory_of_each_pull[action] += 1

                    #updating average_reward
                    #self.average_reward[action] = self.accumulated_rewards[action]/self.memory_of_each_pull[action]
                    #self.table_of_rewards[observation][action] = self.table_of_rewards[observation][action]
                    
                    #updating sigma_pulls
                    self.sigma_pulls += 1

                    #updating sigma_sum
                    self.sigma_sum += reward

                    #updating memory
                    self.history_of_pulls.append(action)

                    #updating steps
                    self.steps.append(len(self.steps))

                    #updating running avg reward for each step
                    self.running_avg.append(self.sigma_sum/self.sigma_pulls)

                    #regret calculations
                    mean_reward_of_selected_arm = np.mean(self.r_dist[action])
                    regret_of_step = self.mean_reward_of_best_arm - mean_reward_of_selected_arm
                    self.sigma_regret += regret_of_step

                    self.running_regret.append(self.sigma_regret)

                    
                    #debug
                    #print("decayed epsilon: " + str(decayed_epsilon))
                    #print("action distribution: " + str(action_distribution))
                    #print("History of action distrbution: " + str(self.history_of_action_distributions))
                    #("Observation: " + str(observation))
                    #print("Sigma sum: " + str(self.sigma_sum))
                    #print("Reward: " + str(reward))
                    #print(self.table_of_rewards[0])
                    #print("best arm {} ___ Mean_reward_of_best_arm {} ".format(self.best_arm, self.mean_reward_of_best_arm))
                    #print("selected arm: {} ___ mean_reward_of_selected_arm {}".format(action, mean_reward_of_selected_arm))
                    #print("regret: " + str(self.sigma_regret))
                    #print("ucb_of_each arm" + str(self.ucb_of_each_arm))
                    #print("average reward array: \n" + str(self.average_reward))
                    #print(self.sigma_regret)
                    #print(self.running_regret)
                    #print(self.history_of_pulls)
                    

                    if done:
                        done = True
                    else:
                        observation = next_observation 
                        time += 1 
    
                

    def plots(self):

        #normalizing bar graph
        for i in range(len(self.memory_of_each_pull)):
            self.memory_of_each_pull[i] = self.memory_of_each_pull[i]/self.sigma_pulls
       
        #print(self.memory_of_each_pull)
        #print(sum_of_distribution)

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

        # #optimal reward
        # fig3 = plt.figure(figsize = (50,50))
        # plt.plot(self.steps, self.optimal_avg_running_reward)
        # plt.xlabel("Time steps")
        # plt.ylabel("Optimal running avg reward")
        # plt.title("Optimal average running reward")

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