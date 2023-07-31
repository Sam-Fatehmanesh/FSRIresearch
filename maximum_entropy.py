# from env import environment
from collections import defaultdict
import numpy as np
import gym
import plotting
import torch
import env as rlworld

np.random.seed(42)

w = rlworld.env

def max_ent_policy(q, n_actions, epsilon):
    def policy(observation):
        best_action_idx = np.argmax(q[observation] + 1e-10 * np.random.random(q[observation].shape))
        distribution = []
        for action_idx in range(n_actions):
            probability = epsilon / n_actions
            if action_idx == best_action_idx:
                probability += 1 - epsilon
            distribution.append(probability)
        return distribution
    return policy

def entropy(actions):
    return -np.sum(actions*np.log(actions))

def q_learning_base(env, *, num_episodes, alpha, gamma, epsilon, max_entropy):

    statistics = plotting.EpisodeStats(
        episode_lengths=np.zeros(num_episodes),
        episode_rewards=np.zeros(num_episodes))
    nA = env.action_space.n
    q = defaultdict(lambda: np.zeros(nA))
    for episode_idx in range(num_episodes):
        if (episode_idx + 1) % 10 == 0:
            print("\nEpisode {}/{}"
                  .format(episode_idx + 1, num_episodes))
        observation = env.reset()
        terminal = False
        t = 1
        while not terminal:
            policy = max_ent_policy(q, env.action_space.n, epsilon)
            action_distribution = policy(observation)
            action = np.random.choice(np.arange(len(action_distribution)),
                                      p=action_distribution)
            next_observation, reward, done, _ = env.step(action)
            next_observation = torch.tensor(next_observation)
            statistics.episode_rewards[episode_idx] += reward
            statistics.episode_lengths[episode_idx] = t
            next_action_values = [q[next_observation][next_action]
                                  for next_action
                                  in np.arange(nA)]
            best_next_q = max(q[next_observation])
            entropy_term = entropy(action_distribution)
            q[observation][action] += alpha * (reward + gamma * best_next_q - q[observation][action] + entropy_term)

            if done:
                terminal = True
            else:
                observation = next_observation
                t += 1
    return q, statistics
