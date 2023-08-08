import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import env as maBanditWorld

#epsilon_greedy without Q-learning because I'm not sure what the states of the table are (actions are pulling "arms")

class epsilon_greedy:
    #creates a randomly generated seed of the bandit world, think minecraft
    #why does init need __? python needs it for init methods
    def __init__(self, n_actions, seed = 17):
        self.memory_of_each_pull = [0.0 for i in range(n_actions)]
        self.accumulated_rewards = [0.0 for i range(n_actions)]
        self.average_reward = [self.accumulated_rewards[i]/self.memory_of_each_pull[i] for i in range(self.memory_of_each_pull)]
        self.arms_array = [i for i in range(n_actions)]
        self.history_of_pulls = []
        self.steps = []
        self.running_avg = []
        self.sigma_sum = 0
        self.sigma_pulls = 0
        np.random.seed(seed)
        self.env = env

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
    
    #under the assumption that epsilon is between 0,1
    #under the assumptiong that decay_rate is between 0,1
    #take initial epsilon and then decrease it until we are exploiting near the end of episodes
    def epsilon_decay(self, num_episodes, decay_rate, epsilon):
        #as episodes happen, i want epsilon to get smaller, decay, so need to track where num-epsidoes is at
        return epsilon * ((1 - decay_rate)**(num_episodes))

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
                    decayed_epsilon = self.epsilon_decay(num_episodes, decay_rate, epsilon)
                    policy = self.epsilon_greedy_policy(n_actions = numActions, epsilon = decayed_epsilon)
                    action_distribution = policy()
                    action = np.random.choice(np.arange(len(action_distribution)), p=action_distribution)
                    
                    #adding reward (observation) to its slot
                    self.accumulated_rewards[action] += observation

                    #when a specific arm is pulled, add that number to that index
                    #so now we know how many times an arm has been pulled
                    self.memory_of_each_pull[action] += 1

                    #updating sigma_sum
                    self.sigma_sum += observation

                    #updating sigma_pulls
                    self.sigma_pulls += 1

                    #updating memory
                    self.history_of_pulls[episode_idx*step_count + i] = action

                    #updating steps
                    self.steps[episode_idx*step_count + i] = (episode_idx*step_count) + i

                    #updating running avg reward for each step
                    self.running_avg[episode_idx*step_count + i] = self.sigma_sum/self.sigma_pulls


                    #need help understanding the following line
                    #i know from reading articles that it feeds the envionrment the action but why is there a _?, isn't that slot for info?
                    #this convention looks so weird
                    #use _ if not using that value
                    next_observation, reward, done, _ = self.env.env.step(action)

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

        #bar graph
        #x will be 0,1,2,3,,4,5, num of arms
        #y will be num of pulls
        fig1 = plt.figure(figsize = (50,50))
        plt.bar(self.arms_array, self.memory_of_each_pull, color = 'blue', width =.4)
        plt.xlabel("Arm")
        plt.ylabel("Number of times pulled")
        plt.title("Total pulls by arm")


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


        #continous plot timestepv. reward
        #x will be steps
        #y will be avg reward
        fig3 = plt.figure(figsize = (50,50))
        plt.plot(self.steps, self.running_avg)
        plt.xlabel("Time steps")
        plt.ylabel("Running Average")
        plt.title("Running average at each time step")


        plt.show()

        #is the bottom right? yes
        return fig1, fig2, fig3