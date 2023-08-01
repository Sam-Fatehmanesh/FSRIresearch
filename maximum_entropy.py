# from env import environment
from collections import defaultdict
import numpy as np
import gym
import plotting
import torch
import env as rlworld

#expsil
class MaxEntropyQLearning:
    def __init__(self, env, seed=42, ):
        np.random.seed(seed)
        self.env = env

    def eps_policy(self, q, n_actions, epsilon):
        def policy(observation):
            # Epsilon-greedy policy with added exploration noise to break ties
            best_action_idx = np.argmax(q[observation] + 1e-10 * np.random.random(q[observation].shape))
            distribution = []
            for action_idx in range(n_actions):
                probability = epsilon / n_actions
                if action_idx == best_action_idx:
                    probability += 1 - epsilon
                distribution.append(probability)
            return distribution
        return policy

    def entropy(self, actions):
        # Compute the entropy of the action distribution
        return -np.sum(actions * np.log(actions))

    def train(self, num_episodes, learning_rate, discount_factor, epsilon):
        # Initialize Q-value function and episode statistics
        numActions = self.env.action_space.n
        q_size = 50
        q = np.zeros((q_size, q_size, numActions))
        episode_lengths = np.zeros(num_episodes)
        episode_rewards = np.zeros(num_episodes)

        for episode_idx in range(num_episodes):
            if (episode_idx + 1) % 10 == 0:
                print("\nEpisode {}/{}".format(episode_idx + 1, num_episodes))
            
            observation = self.env.reset()
            done = False
            time = 1
            
            while not done:
                # Choose an action using the max-entropy policy
                policy = self.eps_policy(q, numActions, epsilon)
                action_distribution = policy(observation)
                action = np.random.choice(np.arange(len(action_distribution)), p=action_distribution)
                
                next_observation, reward, done, _ = self.env.step(action)
                next_observation = torch.tensor(next_observation)
                
                # Update episode statistics
                episode_rewards[episode_idx] += reward
                episode_lengths[episode_idx] = t
                
                # Compute the best Q-value for the next state
                best_next_q = max(q[next_observation])
                
                # Compute the entropy term for the current action distribution
                entropy_term = self.entropy(action_distribution)
                
                # Update Q-value using the Q-learning update rule with entropy regularization
                q[observation][action] += learning_rate * (reward + discount_factor * best_next_q - q[observation][action] + entropy_term)
                
                if done:
                    done = True
                else:
                    observation = next_observation
                    time += 1
        
        return q, episode_lengths, episode_rewards
