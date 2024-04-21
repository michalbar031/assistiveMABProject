import numpy as np
import
class Robot:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.human_observations = []
        self.actual_pulls = []
        self.rnn = None

    def update_human_choice(self, human_choice):
        self.human_observations.append(human_choice)

    def update_actual_pulls(self, robot_pull):
        self.actual_pulls.append(robot_pull)


    def select_arm(self):
        # Simple strategy: follow the human's most common choice
        if self.human_observations:
            return np.argmax(np.bincount(self.human_observations))
        return np.random.choice(self.n_arms)

    def init_rnn(self):
        self.rnn = RNN(self.n_arms)

    def update_train_rnn(self):
        pass