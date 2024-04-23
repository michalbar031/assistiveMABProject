import numpy as np

class MortalHumanPolicy:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.choices = []
        self.actual_pulls = []
        self.rewards = []

    def select_arm(self, active_arms):
        raise NotImplementedError

    def update_choices(self, chosen_arm):
        self.choices.append(chosen_arm)

    def update_rewards(self, robot_pull, reward):
        self.rewards.append(reward)
        self.actual_pulls.append(robot_pull)

class MortalEpsilonGreedy(MortalHumanPolicy):
    def __init__(self, n_arms, epsilon=0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon

    def select_arm(self, active_arms):
        if np.random.rand() < self.epsilon:
            return np.random.choice(active_arms)
        else:
            success_rates = [self.calculate_success_rate(arm) for arm in active_arms]
            return active_arms[np.argmax(success_rates)]

    def calculate_success_rate(self, arm):
        relevant_pulls = [r for i, r in zip(self.actual_pulls, self.rewards) if i == arm]
        return np.mean(relevant_pulls) if len(relevant_pulls) > 0 else 0