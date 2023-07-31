import gymnasium
import pytorch

class agent():
    def __init__(self):
        self.model = None

    def getAction(self, observation):
        return self.ValueAlgo(self.getPredQ(observation))

    def ValueAlgo(self, Q):
        Action = None
        return Action

    def getPredQ(self, observation):
        return self.model.varout(observation)