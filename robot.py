import numpy as np
class Robot:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.human_observations = []
        self.actual_pulls = []

    def update_human_choice(self, human_choice):
        self.human_observations.append(human_choice)

    # def update(self, human_choice):


    def select_arm(self):
        # Simple strategy: follow the human's most common choice
        if self.human_observations:
            return np.argmax(np.bincount(self.human_observations))
        return np.random.choice(self.n_arms)
