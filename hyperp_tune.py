from tqdm import tqdm
import numpy as np
import pandas as pd
from env import maBanditWorld
from maximum_entropy import MaxEntropyQLearning
import datetime
from matplotlib import pyplot as plt

class HyperRandTuner:
    def __init__(self, parambounds):
        np.random.seed(42)
        self.parambounds = parambounds
        self.paramcount = len(parambounds)
        self.data = pd.DataFrame(columns=['round', 'params', 'score'])
        

    def randomParams(self):
        params = []
        for i in range(self.paramcount):
            params.append(np.random.uniform(self.parambounds[i][0], self.parambounds[i][1]))
        return params


    def tune(self, testFunc, rounds):
        bestparams = np.zeros(self.paramcount)
        top_score = -1000000000.0
        for i in tqdm(range(rounds)):
            np.random.seed(i)
            params = []
            for j in range(self.paramcount):
                params.append(np.random.uniform(self.parambounds[j][0], self.parambounds[j][1]))
            score = testFunc(params)            
            
            self.data = pd.concat([self.data, pd.DataFrame([{'round': i, 'params': params, 'score': score}])])
            if score > top_score:
                top_score = score
                bestparams = params
        
        print("Best parameters: ", bestparams)
        print("Best score: ", top_score)
        # Saves csv with filename and date and time
        self.data.to_csv('hyperparam_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv', index=False)
        plt.show()

class HyperEvoTuner:
    def __init__(self, initparambounds, mutateFactor=0.1):
        np.random.seed(42)
        self.initparambounds = initparambounds
        self.paramcount = len(initparambounds)
        self.data = pd.DataFrame(columns=['round', 'params', 'score'])
        self.mutateFactors = [mutateFactor*((initparambounds[i][0]-initparambounds[i][1])/2.0) for i in range(self.paramcount)]

    def randomParams(self):
        params = []
        for i in range(self.paramcount):
            params.append(np.random.uniform(self.initparambounds[i][0], self.initparambounds[i][1]))
        return params
    
    def mutate(self, params):
        for i in range(self.paramcount):
            params[i] += np.random.uniform(0, 1) * self.mutateFactors[i]
        return params

    def batchmutate(self, params):
        newparams = []
        for i in range(len(params)):
            newparams.append(self.mutate(params[i]))
        return newparams

    def initGen(self, num):
        params = []
        for i in range(num):
            params.append(self.randomParams())
        return params

    def select(self, params, scores, num):
        sortedParams = [x for _,x in sorted(zip(scores, params))]
        return sortedParams[-num:]

    def reproduce(self, params, num):
        newparams = []
        for i in range(num):
            newparams.append(self.mutate(params[np.random.randint(0, len(params))]))
        return newparams

    def evolve(self, testFunc, rounds, initGenSize=10, selectSize=5, reproduceSize=5, mutateSize=5):
        bestparams = np.zeros(self.paramcount)
        top_score = -1000000000.0
        params = self.initGen(initGenSize)
        scores = []
        print("Testing initial generation")
        # Initial generation
        for i in tqdm(range(initGenSize)):
            scores.append(testFunc(params[i]))
        # Evolution loop
        print("Running evolutionary tuning")
        for i in tqdm(range(rounds)):
            params = self.select(params, scores, selectSize)
            params = self.reproduce(params, reproduceSize)
            params = self.batchmutate(params)
            scores = []
            # Tests new generation
            for j in tqdm(range(len(params)), leave=False):
                scores.append(testFunc(params[j]))
            # Saves data
            self.data = pd.concat([self.data, pd.DataFrame([{'round': i, 'params': params, 'score': scores}])])
            if max(scores) > top_score:
                top_score = max(scores)
                bestparams = params[scores.index(top_score)]
        
        print("Best parameters: ", bestparams)
        print("Best score: ", top_score)
        # Saves csv with filename and date and time
        self.data.to_csv('hyperparam_' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + '.csv', index=False)
        plt.show()

# Tests hyperparameters for the maximum entropy algorithm
def testME(params):
    total_steps = 200*100
    avg = 0
    X = [i for i in range(total_steps)]
    Y = [0.0 for i in range(total_steps)]
    for i in tqdm(range(6), leave=False):
        env = maBanditWorld()
        max_ent = MaxEntropyQLearning(env)
        ent_stats = max_ent.train(num_episodes=200, learning_rate=params[0], discount_factor=params[1], epsilon=params[2], step_count=100, decay_factor=params[3])
        Y = [Y[i] + ent_stats.step_reward_avg[i] for i in range(total_steps)]
        avg += ent_stats.step_reward_avg[-1]
    Y = [Y[i]/6.0 for i in range(total_steps)]
    plt.plot(X, Y, color='r')
    return avg/6.0