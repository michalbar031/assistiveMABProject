import numpy as np
class Robot:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.observations = []

    def update(self, human_choice):
        self.observations.append(human_choice)

    def select_arm(self):
        # Simple strategy: follow the human's most common choice
        if self.observations:
            return np.argmax(np.bincount(self.observations))
        return np.random.choice(self.n_arms)
