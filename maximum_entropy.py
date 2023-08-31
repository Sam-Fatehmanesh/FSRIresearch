# from env import environment
from collections import defaultdict
import numpy as np
import gymnasium
import plotting
import env as maBandaWorld
from scipy.stats import entropy
import math
from matplotlib import pyplot as plt
#expsil
class MaxEntropyQLearning:
    def __init__(self, env, seed=42 ):
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

#working version
    def epsilon_decay(self, step_count, decay_rate, epsilon):
            return epsilon * ((1 - decay_rate)**(step_count))
    def eps_policy(self, q, n_actions, epsilon):
        def policy(observation):
            # Epsilon-greedy policy with added exploration noise to break ties
            
            distribution = []
            if epsilon > np.random.rand():
                distribution = [1.0/n_actions for i in range(n_actions)]
                return distribution
            else:
                best_action_idx = np.argmax(q[observation] + 1e-10 * np.random.random(q[observation].shape))
                distribution = [0.0 for i in range(n_actions)]
                if 0 <= best_action_idx < n_actions:
                    distribution[best_action_idx] += 1.0
                return distribution
            
            '''
            distribution = q[observation]
            #print(q, "q")
            for action_idx in range(n_actions):
                probability = epsilon
                if action_idx == best_action_idx:
                    probability += 1 - epsilon
                distribution.append(probability)
            return distribution
            # distribution = []
            #             # for action_idx in range(n_actions):
            #     probability = epsilon / n_actions
            #     if action_idx == best_action_idx:
            #         probability += 1 - epsilon
            #     distribution.append(probability)
            '''
            # return distribution
        return policy

    '''
    def entropy(self, actions):
        # Compute the entropy of the action distribution
        return -np.sum(actions * np.log(actions))
    '''

    def entropy2(self, actions):
        value,counts = np.unique(actions, return_counts=True)
        norm_counts = counts / counts.sum()
        return (norm_counts * np.log(norm_counts)/np.log(2.7182)).sum()
    def train(self, num_episodes, learning_rate, discount_factor, epsilon, step_count, decay_factor):
        # Initialize Q-value function and episode statistics
        statistics = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes),
        step_reward_avg=np.zeros(num_episodes*step_count))

        numActions = self.env.env.action_space.n
        q_size = 10
        q =  np.random.rand(q_size, q_size)
        rewards = np.zeros(num_episodes*step_count)
        numBandits = self.env.env.action_space.n
        data = np.zeros((numBandits, 2))
        rewards = np.zeros(num_episodes*step_count)
        armPulled = np.zeros(numBandits)
        '''
        Initializing the Q-Table with the proper dimensions of the environment, assuming 50 is large enough to discretize it accurately
        '''

        for episode_idx in range(num_episodes):
            if (episode_idx + 1) % 10 == 0:
                print("\nEpisode {}/{}".format(episode_idx + 1, num_episodes))
            
            observation = self.env.env.reset()
            done = False
            time = 1
            
            while not done:
                # Choose an action using the max-entropy policy
                for i in range(step_count):
                    decayed_epsilon = self.epsilon_decay(step_count=i+1, decay_rate=decay_factor, epsilon=epsilon) #decays within episodes not out of

                    policy = self.eps_policy(q, numActions, epsilon)
                    action_distribution = policy(observation)
                    total_probability = sum(action_distribution)
                    normalized_probabilities = [p / total_probability for p in action_distribution]
                    action = np.random.choice(list(range(0,numActions)), p=normalized_probabilities)
                    next_observation, reward, done, _ = self.env.env.step(action)

                    # Update episode statistics
                    rewards[episode_idx*step_count + i] = reward
                    statistics.episode_rewards[episode_idx] += reward
                    statistics.episode_lengths[episode_idx] = time
                    statistics.step_reward_avg[episode_idx*step_count + i] = np.sum(rewards)/(episode_idx*step_count + i + 1)

                    # Compute the best Q-value for the next state
                    best_next_q = np.max(q[next_observation])
                    
                    # Compute the entropy term for the current action distribution
                    entropy_term = entropy(action_distribution,base=math.e)
                    
                    # Update Q-value using the Q-learning update rule with entropy regularization
                    q[observation][action] += learning_rate * (reward + discount_factor * (best_next_q - q[observation][action] + 0.001*entropy_term))


                    '''
                    NEED TO DO: Multiply the entropy term with some fixed beta term from 0 to inf

                    Writing in mathematical form: 

                    Q(s_1,a_1) = Q(s_0,a_0) + alpha * (reward + gamma( max(Q(s_1, a_1)) - Q(s_0,a_0) ))
                    Update the Q-value function, which is basically our utility function that we are trying to optimize for picking the best actions, 
                    alpha is some learning rate term that is arbitrarily defined, discount factor determines how farsighted or nearsighted the agent is, 
                    and the update in Q is just picking the max Q-value given the time step ahead. The interesting part here is the added entropy term in both
                    the update of the Q-Value and the added noise with the epsilon greedy policy. Maixmizing entropy, or approaching an d action distribution 
                    that is more (disorded?) is theoretically optimal. 
                    '''
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
        
        return statistics
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