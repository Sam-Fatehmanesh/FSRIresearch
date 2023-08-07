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

ent_stats = max_ent.train(num_episodes=1000, learning_rate=0.16, discount_factor=0.9, epsilon=0.095, step_count=10000)
plotting.plot_episode_stats(ent_stats, "Max Entropy Rewards", "epsilon tuning", color=colors[0])
#EPSILON PREFERRED: 0.095
# Learning rate tuning
#0.16 LEARNING RATE

#0.9 DISCOUNT FACTOR
plt.show()
#thomp_stats = thomps.train(num_episodes=10000, step_count=1000)

#plotting.plot_episode_stats(thomp_stats, "Thompson Smapling Rewards ")