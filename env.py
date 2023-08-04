import gymnasium
import gym_bandits


class rlworld():
    def __init__(self, env_name, agent=None, render_mode=None):
        self.env = gymnasium.make(env_name, render_mode = render_mode)
        self.observation, self.info = self.env.reset()
        self.reward = None
        self.terminated = None
        self.truncated = None
        self.agent = None
        self.actions = []
        self.episode_Q = 0
        self.observations = []


    def printAS(self):
        print(self.env.action_space.sample())

    def printenvstate(self):
        print(self.observation)
        print(self.info)

    def getObservation(self):
        return self.observation

    def step(self):
        action = self.agent.getAction(self.observation)
        self.actions.append(action)

        self.observation, self.reward, self.terminated, self.truncated, self.info = self.env.step(action)
        self.observations.append(self.observation)

    def run(self, step_count, training_mode=False):
        for _ in range(step_count):
            self.step() 
            if self.terminated or self.truncated:

                self.observation, self.info = self.env.reset()
                
                if training_mode:
                    self.agent.train(self.observations, self.actions, self.episode_Q)
                    self.episode_Q = 0
                    self.actions = []


        self.env.close()

    


# w = rlworld("LunarLander-v2", render_mode="human")

# w.printenvstate()

class maBanditWorld():
    def __init__(self, agent=None):
        self.env = gymnasium.make("BanditTenArmedGaussian-v0")
        self.observation = self.env.reset()
        self.reward = None
        self.terminated = None
        self.truncated = None
        self.agent = None
        self.actions = []
        self.episode_Q = 0
        self.observations = []
        self.info = None


    def printAS(self):
        print(self.env.action_space)

    def printenvstate(self):
        print(self.observation)

    def getObservation(self):
        return self.observation

    def step(self):
        action = self.agent.getAction(self.observation)
        self.actions.append(action)

        self.observation, self.reward, self.terminated, self.truncated, self.info = self.env.step(action)
        self.observations.append(self.observation)

    def run(self, step_count, training_mode=False):
        for _ in range(step_count):
            self.step() 
            if self.terminated or self.truncated:

                self.observation = self.env.reset()
                
                if training_mode:
                    self.agent.train(self.observations, self.actions, self.episode_Q)
                    self.episode_Q = 0
                    self.actions = []


        self.env.close()

