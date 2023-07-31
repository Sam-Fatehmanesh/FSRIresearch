import gymnasium as gym
env = gym.make("LunarLander-v2", render_mode="human")
observation, info = env.reset(seed=42)
for _ in range(1000):
   action = env.action_space.sample()  # this is where you would insert your policy
   observation, reward, terminated, truncated, info = env.step(action)

   if terminated or truncated:
      observation, info = env.reset()

env.close()

# class rlworld():
#     def                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   import gymnasium as gym
# env = gym.make("LunarLander-v2", render_mode="human")
# observation, info = env.reset(seed=42)
# for _ in range(1000):
#    action = env.action_space.sample()  # this is where you would insert your policy
#    observation, reward, terminated, truncated, info = env.step(action)

#    if terminated or truncated:
#       observation, info = env.reset()

# env.close()

class rlworld():
    def __init__(self, env_name, agent=None, render_mode=None):
        self.env = gym.make(env_name, render_mode = render_mode)
        self.observation, self.info = self.env.reset()
        self.reward = None
        self.terminated = None
        self.truncated = None
        self.agent = None


    def printAS(self):
        print(self.env.action_space.sample())

    def printenvstate(self):
        print(self.observation)
        print(self.info)

    def getObservation(self):
        return self.observation

    def step(self):
        action = self.agent.getAction(self.observation)
        self.observation, self.reward, self.terminated, self.truncated, self.info = env.step(action)

    def run(self, step_count, training_mode=False):
        for _ in range(step_count):
            self.step() 
            if self.terminated or self.truncated:
                self.observation, self.info = self.env.reset()

        env.close()

    


w = rlworld("LunarLander-v2", render_mode="human")

w.printenvstate()