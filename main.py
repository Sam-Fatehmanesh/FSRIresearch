from maximum_entropy import MaxEntropyQLearning
from thompson_samp import ThompsonSampling
from otsFO import OptimisticThompsonSampling
import plotting
import numpy as np
from env import maBanditWorld
from matplotlib import pyplot as plt
from epsilon_greedy import epsilon_greedy
from ucb import upper_confidence_bound
from tqdm import tqdm

env = maBanditWorld()
max_ent = MaxEntropyQLearning(env)
thomps = OptimisticThompsonSampling(env)
eps_greed = epsilon_greedy(10, env, 42)
upp_con = upper_confidence_bound(10, env, 42)

colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'orange', 'purple', 'brown']
# thomps.train(1, 2000, .1, "red")
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
def run_many_test(test_func):
    regret_curves = []
    for i in tqdm(range(512)):
        np.random.seed(i)
        # max_ent = MaxEntropyQLearning(env, seed=i+69)

        regret_curves.append(test_func(i+69))

        


    for i in range(len(regret_curves)):
        plt.plot([i for i in range(len(regret_curves[i]))], regret_curves[i], label = str(i))
    plt.show()



def ex_test(seed):
    # write code to run a single algo test
    # ex
    # thomps = OptimisticThompsonSampling(env, seed=seed)
    # return thomps.train(num_episodes=1, step_count=400, lambdaconst=0.1, graph_color=colors[i%10]))
    pass

#c is the confidence interval, higher c more explore, low c more exploit
#epsilons = []
#for j in range(30):
    #epsilons.append(upp_con.train(num_episodes=1, step_count= 4000, c=.99))
#upp_con.plots()
#print("Averaged epsilon: ", np.average(epsilons) )

#epg 1/t decay results in lowered regret compared to ucb under the same step_count and num_episodes


#1891 steps


