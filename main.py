from maximum_entropy import MaxEntropyQLearning
from thompson_samp import ThompsonSampling
import plotting
from env import maBanditWorld
from matplotlib import pyplot as plt
import epsilon_greedy

env = maBanditWorld()
max_ent = MaxEntropyQLearning(env)
thomps = ThompsonSampling(env)
eps_greed = epsilon_greedy(env)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

# Epsilon tuning
#for i in range(10):
    #ent_stats = max_ent.train(num_episodes=1000, learning_rate=0.16, discount_factor=0.9, epsilon=0.01, step_count=100, decay_factor=0.98)
    #plotting.plot_episode_stats(ent_stats, "Max Entropy Rewards", "epsilon tuning", color=colors[i])
#EPSILON PREFERRED: 0.095
# Learning rate tuning
#0.16 LEARNING RATE
#thomp_stats = thomps.train(num_episodes=10000, step_count=1000)

#plotting.plot_episode_stats(thomp_stats, "Thompson Smapling Rewards ")
#0.9 DISCOUNT FACTOR


eps_greed.train(num_episodes=10, decay_rate=.005, epsilon = 1, step_count = 1000)
eps_greed.plots()

