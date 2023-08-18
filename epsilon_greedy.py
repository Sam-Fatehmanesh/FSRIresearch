import numpy as np
import gymnasium as gym
from matplotlib import animation
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import env as maBanditWorld

#epsilon_greedy with decaying epsilon without Q-learning because I'm not sure what the states of the table are (actions are pulling "arms")

class epsilon_greedy:
    #creates a randomly generated seed of the bandit world, think minecraft
    #why does init need __? python needs it for init methods
    def __init__(self, n_actions, env, seed):
        self.env = env
        np.random.seed(seed)

        self.memory_of_each_pull = [0.0 for i in range(n_actions)]
        self.accumulated_rewards = [0.0 for i in range(n_actions)]
        self.average_reward = [0.0 for i in range(n_actions)]
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

        #how to calculate regret at a given time step:
        #highest mean reward of the 10 arm reward distributions - mean reward of the selected arm reward distribution
    
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

        

    #whats the difference between the above line and np.random.seed(<seed>) \n env.seed(17)
    #don't really need self.env = env
    #when creating the mabanditworld object, we can set agent to epsilon-greedy
    #since we don't use env anywhere, it's another reason to get rid of self.env = env
    #need self.env = env because need to access the self.actions = [] that contains a history of all actions


    #question: what does np.random.seed do? I don't see how it randomly makes a bandit world when env and np.random.seed isn't connected together - done

    #in python, methods of a class needs the parameter self
    #self parameter allows instance (objects) of a class to have their own attributes (defs); if no self, multiple instances can't have same value
    
    '''
    Question: when creating an object of the greedy_epsilon class, our parameters are what?
    is it: epsilon_greedy(maBanditWorld, 47)?
    if so, where does the program get the parameters for the methods in the class?
    like where does n_actions and observations come from?
    '''
    
#try 1/t decay rate

#regret is not calculated correctly




    #under the assumption that epsilon is between 0,1
    #under the assumptiong that decay_rate is between 0,1
    #take initial epsilon and then decrease it until we are exploiting near the end of episodes
    def epsilon_decay(self, step_count, decay_rate, epsilon):
        #as episodes happen, i want epsilon to get smaller, decay, so need to track where num-epsidoes is at
        return 1/step_count
        
        #1/step_count #wtf does this decay have lower regret than ucb?

        #epsilon * ((1 - decay_rate)**(step_count))

    #what is q? q table numbers
    def epsilon_greedy_policy(self, n_actions, epsilon):
        def policy():

            #question: what is the specific observation? in maze it would be position
            #if observation is a number, applying MDP here, how do you transition from one # to another? 
            #it's easier to think about transiitoning from 1 position on a maze to another after an action but not with a numerical observation (that's random since slot machine gives random reward amts (right?))

            #exploration; outputs even distribution of all actions then we can select
            if epsilon > np.random.rand(): #rand generates a float
                #distribution is an array, each index of the array is the chance that each action will be chosen
                #the for i in range(n_actions) populates the array with equal probs for all actions
                #1.0 is used to guarantee system knows its a float output like 1/7
                distribution = [1.0/n_actions for i in range(n_actions)]
                return distribution
            #exploitation: figuring out what the best action is and then making sure that it's a peak so the program picks it
            else:
                #what does idx stand for? index position
                #deciding best arm by avg reward of each arm (could be much better if we did standard curve math)
                
                #when a specific output is observed from pulling an arm, add that number to that index and then divide by the corresponding pull count in memory of pulls
                distribution = [0.0 for i in range(n_actions)]
                best_action_idx = np.argmax(self.average_reward) #gets the index of the largest value in accumulated_rewards
                distribution[best_action_idx] += 1.0
                return distribution
        return policy



    #question: what is happening with obsevation, reward, done, info = env.step(action)

    #each step is 1 pull
    #each episode is 1000 pulls
    #separate them so the RL doesn't get "stuck" in harder problems

    def train(self, num_episodes, decay_rate, epsilon, step_count):

        numActions = self.env.env.action_space.n #need some help understanding this one, .n part, is the .n part just a numerical value?
        
        for episode_idx in range(num_episodes):        
            print("\nEpisode {}/{}".format(episode_idx + 1, num_episodes)) #episode_idx + 1 b/c we start counting from 0

            #not infinite loop because the "game" tells us when we're done
            
            #the following line resets the enviornment so we transition out of the inner loop (stops us from local minimas in other things than MAB)
            observation = self.env.env.reset() #why 2 env?
            done = False
            time = 1

            while not done:
                for i in range(step_count):
                    #choose an action using greedy-epsilon policy
                    
                    decayed_epsilon = self.epsilon_decay(step_count=i+1, decay_rate=decay_rate, epsilon=epsilon) #decays within episodes not out of
                    policy = self.epsilon_greedy_policy(n_actions=numActions, epsilon=decayed_epsilon) #epsilon itself outputs a linear regret curve
                    action_distribution = policy()

                    #self.average_reward =  [self.accumulated_rewards[i]/(self.memory_of_each_pull[i]+1) for i in range(len(self.memory_of_each_pull))]

                    action = np.random.choice(np.arange(len(action_distribution)), p=action_distribution)

                    print(np.arange(len(action_distribution)))
                    print(action_distribution)

                    next_observation, reward, done, _ = self.env.env.step(action)
                    
                    #appending action to history of action distributions for plotting later
                    self.history_of_action_distributions.append(action_distribution)

                    #adding reward to its slot
                    self.accumulated_rewards[action] += reward

                    #when a specific arm is pulled, add that number to that index
                    #so now we know how many times an arm has been pulled
                    self.memory_of_each_pull[action] += 1

                    #updating average_reward
                    self.average_reward[action] = self.accumulated_rewards[action]/self.memory_of_each_pull[action]

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

                    # #finds mean_reward so we can calculate regret
                    # #mean_reward_of_selected_arm = np.mean(self.r_dist[action])

                    mean_reward_of_selected_arm = self.average_reward[action]

                    # mean_reward_of_best_arm_idx = np.argmax(self.average_reward)

                    # mean_reward_of_best_arm = self.average_reward[mean_reward_of_best_arm_idx]

                    # #calculating regret
                    # regret_of_step = mean_reward_of_best_arm - mean_reward_of_selected_arm

                    # #calculating sigma_regret
                    # self.sigma_regret += regret_of_step

                    # #updating sigma_regret list
                    # #self.running_regret.append(self.sigma_pulls*self.mean_reward_of_best_arm - self.sigma_regret)
                    
                    #regret after T rounds is calcultaed as 

                    maximal_reward_mean_arm_idx = np.argmax(self.average_reward)
                    
                    regret_after_T_rounds = (self.sigma_pulls * self.average_reward[maximal_reward_mean_arm_idx]) - self.sigma_sum
                    self.sigma_regret += regret_after_T_rounds
                    self.running_regret.append(self.sigma_regret/self.sigma_pulls)


                    self.optimal_avg_running_reward.append(self.mean_reward_of_best_arm)

                    #need help understanding the following line
                    #i know from reading articles that it feeds the envionrment the action but why is there a _?, isn't that slot for info?
                    #this convention looks so weird
                    #use _ if not using that value

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
                    
                    if done: #HOW DOES IT KNOW ITS DONE?? the multiarmed bandit game tells the program when it's done
                        done = True
                    else:
                        observation = next_observation #need explanation; update observation so new stuff
                        time += 1 #what is this used for again?


    def plots(self):
        '''
        For plotting, q we get a bar graph, x is each arm, y is amount of times pulled
        also a graph where x is episode number and y is arm but it's dots instead
        also a gprah where x is epsidoe number and y is reward amt (continous graph)
        '''

        #normalizing bar graph
        for i in range(len(self.memory_of_each_pull)):
            self.memory_of_each_pull[i] = self.memory_of_each_pull[i]/self.sigma_pulls
        #print(self.memory_of_each_pull)

        #sum_of_distribution = np.sum(self.memory_of_each_pull)

        #print(sum_of_distribution)

        #bar graph
        #x will be 0,1,2,3,,4,5, num of arms
        #y will be num of pulls
        fig1 = plt.figure(figsize = (50,50))
        plt.bar(self.arms_array, self.memory_of_each_pull, color = 'blue', width =.4)
        plt.xlabel("Arm")
        plt.ylabel("Probability")
        plt.title("Probability per arm")

        # updating_bar_array = []
        # def update():
        #     plt.bar(self.arms_array, self.memory_of_each_pull, color = 'blue', width =.4)
        #     plt.xlabel("Arm")
        #     plt.ylabel("Number of times pulled")
        #     plt.title("Total pulls by arm")
        #     updating_bar_array = self.memory_of_each_pull

        # ani = FuncAnimation(fig1, update, frames=range(10), repeat=False)


        #converging dots plot arm pulledv. steps
        #x will be steps
        #y will be num of arms
        #dots instead of linear, should converge to 1 y value
        fig2 = plt.figure(figsize = (50,50))
        #used plot because it's just a dot so it's not contiousous (cause matplotlibs isn't conitnous right?)
        plt.plot(self.steps, self.history_of_pulls)
        plt.xlabel("Time steps")
        plt.ylabel("Arm pulled")
        plt.title("Arm pulled at each time step")

        #animated bar graph
        #fig3 = plt.figure(figsize = (50,50))

        #optimal reward
        fig3 = plt.figure(figsize = (50,50))
        plt.plot(self.steps, self.optimal_avg_running_reward)
        plt.xlabel("Time steps")
        plt.ylabel("Optimal running avg reward")
        plt.title("Optimal average running reward")

        #continous plot timestepv. reward
        #x will be steps
        #y will be avg reward
        fig4 = plt.figure(figsize = (50,50))
        plt.plot(self.steps, self.running_avg)
        plt.xlabel("Time steps")
        plt.ylabel("Running Average")
        plt.title("Running average at each time step")

        #regret plot, shows us how close our reward is to actual max reward possible of that step
        #x wlil be steps
        #y will be regret of each step
        fig5 = plt.figure(figsize = (50,50))
        plt.plot(self.steps, self.running_regret)
        plt.xlabel("Time steps")
        plt.ylabel("Regret")
        plt.title("Time steps vs. Regret")


        plt.show()

        #is the bottom right? yes
        return fig1, fig2, fig4, fig5