from maximum_entropy import MaxEntropyQLearning
from thompson_samp import ThompsonSampling
import plotting
from env import maBanditWorld

env = maBanditWorld()
max_ent = MaxEntropyQLearning(env)
thomps = ThompsonSampling(env)

ent_stats = max_ent.train(num_episodes=1000, learning_rate=0.1, discount_factor=0.99, epsilon=0.1)
thomp_stats = thomps.train(num_episodes=1000)

plotting.plot_episode_stats(ent_stats)
plotting.plot_episode_stats(thomp_stats)