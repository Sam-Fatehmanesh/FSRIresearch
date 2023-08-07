from maximum_entropy import MaxEntropyQLearning
from thompson_samp import ThompsonSampling
import plotting
from env import maBanditWorld
from matplotlib import pyplot as plt
env = maBanditWorld()
max_ent = MaxEntropyQLearning(env)
thomps = ThompsonSampling(env)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']

# Epsilon tuning

for i in range(10):
    ent_stats = max_ent.train(num_episodes=1000, learning_rate=0.1, discount_factor=0.9, epsilon=0.1-i*0.005, step_count=1000)
    plotting.plot_episode_stats(ent_stats, "Max Entropy Rewards", "epsilon tuning", color=colors[i])

# Learning rate tuning
for j in range(10):
    ent_stats = max_ent.train(num_episodes=1000, learning_rate=0.1+j*0.01, discount_factor=0.9, epsilon=0.1, step_count=1000)
    plotting.plot_episode_stats(ent_stats, "Max Entropy Rewards", "learning rate tuning", color=colors[j])

# Discount factor tuning
for k in range(10):
    ent_stats = max_ent.train(num_episodes=1000, learning_rate=0.1, discount_factor=0.9+k*0.01, epsilon=0.1, step_count=1000)
    plotting.plot_episode_stats(ent_stats, "Max Entropy Rewards", "discount factor tuning", color=colors[k])

plt.show()
#thomp_stats = thomps.train(num_episodes=10000, step_count=1000)

#plotting.plot_episode_stats(thomp_stats, "Thompson Smapling Rewards ")