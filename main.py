from maximum_entropy import MaxEntropyQLearning
from thompson_samp import ThompsonSampling
from otsFO import OptimisticThompsonSamplingFO
from OTS import OptimisticThompsonSampling

import plotting
import numpy as np
from env import maBanditWorld
from matplotlib import pyplot as plt
from epsilon_greedy import epsilon_greedy
from ucb import upper_confidence_bound
from tqdm import tqdm
steps = 1540

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
def run_many_test(test_func, bound_func):
    regret_curves = []
    for i in tqdm(range(50)):
        np.random.seed(i)

        regret_curves.append(test_func(i+69))
    
    for i in range(len(regret_curves)):
        plt.plot([i for i in range(len(regret_curves[i]))], regret_curves[i], label = str(i), color="red")

    plt.plot([i for i in range(len(bound_func()))], bound_func(), label = "Bound", color="blue")

    plt.xlabel("Steps")
    plt.ylabel("Total Regret")
    plt.show()

    

def OptimisticThompsonSampling_test(seed):
    ots = OptimisticThompsonSampling(env, seed=seed)
    return ots.train(num_episodes=1, step_count=steps, lambdaconst=0.1, graph_color="red")

def OTbound():
    return [np.sqrt(10*i*np.log(i)) for i in range(steps)]

def OptimisticThompsonSamplingFO_test(seed):
    ots_fo = OptimisticThompsonSamplingFO(env, seed=seed)
    return ots_fo.train(num_episodes=1, step_count=steps, lambdaconst=0.1, graph_color="red")

def OTbound():
    return [np.sqrt(10*i*np.log(i)) for i in range(steps)]

def ThompsonSampling_test(seed):
    thomps = ThompsonSampling(env, seed=seed)
    return thomps.train(num_episodes=1, step_count=steps)

def TSbound():
    return [np.sqrt(10*i*np.log(i)) for i in range(steps)]

def MaxEnt_test(seed):
    max_ent = MaxEntropyQLearning(env)
    return max_ent.train(num_episodes=steps, learning_rate=0.16, discount_factor=0.9, epsilon=0.01, step_count=100, decay_factor=0.98)

def MaxEntBound():
    return [0 for i in range(steps)]

def UCB_test(seed):
    upp_c = upper_confidence_bound(10, env, 42)
    return upp_c.train(num_episodes=1, step_count= steps, c=.99)

def UCBBound():
    return [np.sqrt(i*np.log(i))  for i in range(steps)]

def EpsGreed_test(seed):
    eps_g = epsilon_greedy(10, env, 42)
    return  eps_g.train(num_episodes=1, decay_rate=0.0, epsilon=1, step_count=steps)

def EpsGreedBound():
    return [10*i  for i in range(steps)]

def EpsGreedDec_test(seed):
    eps_g = epsilon_greedy(10, env, 42)
    return  eps_g.train(num_episodes=1, decay_rate=0.0050, epsilon=1, step_count=steps)
    
def EpsGreedDecBound():
    return [i**(2/3) * (10 * np.log(i)) ** (1/3) for i in range(steps)]




run_many_test(OptimisticThompsonSampling_test, OTbound)
run_many_test(OptimisticThompsonSamplingFO_test, OTbound)
run_many_test(ThompsonSampling_test, TSbound)
run_many_test(MaxEnt_test, MaxEntBound)
run_many_test(UCB_test, UCBBound)
run_many_test(EpsGreed_test, EpsGreedBound)
run_many_test(EpsGreedDec_test, EpsGreedDecBound)


#c is the confidence interval, higher c more explore, low c more exploit
#epsilons = []
#for j in range(30):
    #epsilons.append(upp_con.train(num_episodes=1, step_count= 4000, c=.99))
#upp_con.plots()
#print("Averaged epsilon: ", np.average(epsilons) )

#epg 1/t decay results in lowered regret compared to ucb under the same step_count and num_episodes


#1891 steps


