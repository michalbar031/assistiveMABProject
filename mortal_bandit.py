import numpy as np

class MortalBandit:
    def __init__(self, n_arms, lifetimes):
        self.n_arms = n_arms
        self.lifetimes = lifetimes
        self.remaining_lifetimes = lifetimes.copy()
        self.probabilities = np.random.rand(n_arms)

    def generate_reward(self, arm):
        if self.remaining_lifetimes[arm] > 0:
            reward = np.random.binomial(1, self.probabilities[arm])
            self.remaining_lifetimes[arm] -= 1
            return reward
        else:
            return None

    def is_active(self, arm):
        return self.remaining_lifetimes[arm] > 0