import numpy as np
class HumanPolicy:
    def __init__(self, n_arms):
        self.n_arms = n_arms
        self.choices = []
        self.actual_pulls = []
        self.rewards = []

    def select_arm(self):
        raise NotImplementedError

    def update_choices(self, chosen_arm):
        self.choices.append(chosen_arm)

    def update_rewards(self,robot_pull, reward):
        self.rewards.append(reward)
        self.actual_pulls.append(robot_pull)


class EpsilonGreedy(HumanPolicy):
    def __init__(self, n_arms, epsilon=0.1):
        super().__init__(n_arms)
        self.epsilon = epsilon

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)
        else:
            success_rates = [np.mean([r for i, r in zip(self.choices, self.rewards) if i == arm])
                             if self.choices.count(arm) > 0 else 0 for arm in range(self.n_arms)]
            return np.argmax(success_rates)

# Additional policies (WSLS, TS, UCL, GI, and Epsilon-Optimal) would be implemented similarly
