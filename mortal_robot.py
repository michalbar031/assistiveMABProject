import numpy as np
from mortal_bandit import MortalBandit
from mortal_human_policy import MortalHumanPolicy
from rnn import RNN

class MortalRobot:
    def __init__(self, n_arms, lifetimes, input_size, hidden_size=32, epsilon=0.1, c=1):
        self.n_arms = n_arms
        self.lifetimes = lifetimes
        self.bandit = MortalBandit(n_arms, lifetimes)
        self.human_observations = []
        self.actual_pulls = []
        self.hidden_size = hidden_size
        self.rnn = RNN(input_size, hidden_size, n_arms)
        self.epsilon = epsilon
        self.c = c
        self.arm_counts = [0] * n_arms
        self.arm_rewards = [0] * n_arms

    def select_arm(self, human_choice_history, robot_choice_history):
        active_arms = [arm for arm in range(self.n_arms) if self.bandit.is_active(arm)]
        if len(active_arms) == 0:
            return None

        if np.random.rand() < self.epsilon:
            return np.random.choice(active_arms)
        else:
            ucb_values = [self.arm_rewards[arm] / self.arm_counts[arm] +
                          self.c * np.sqrt(np.log(sum(self.arm_counts)) / self.arm_counts[arm])
                          if self.arm_counts[arm] > 0 else float('inf')
                          for arm in active_arms]
            return active_arms[np.argmax(ucb_values)]

    def update_arm_stats(self, arm, reward):
        self.arm_counts[arm] += 1
        self.arm_rewards[arm] += reward

    def prepare_input_sequence(self, human_choice_history, robot_choice_history):
        pass

    def update_human_choice(self, human_choice):
        self.human_observations.append(human_choice)

    def update_actual_pulls(self, robot_pull):
        self.actual_pulls.append(robot_pull)