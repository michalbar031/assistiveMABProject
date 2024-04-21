from __future__ import division

import time
import numpy as np

import time
import numpy as np


class Bandit:
    def generate_reward(self, i):
        raise NotImplementedError("Subclasses should implement this!")


class BetaBernoulliBandit(Bandit):
    def __init__(self, n_arms, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.n_arms = n_arms
        self.alphas = [1] * n_arms  # Alpha parameters of Beta distribution
        self.betas = [1] * n_arms  # Beta parameters of Beta distribution

    def generate_reward(self, i):
        theta_i = np.random.beta(self.alphas[i], self.betas[i])
        reward = np.random.rand() < theta_i
        self.update(i, reward)
        return reward

    def update(self, arm, reward):
        if reward:
            self.alphas[arm] += 1
        else:
            self.betas[arm] += 1

    def get_estimated_probas(self):
        return [self.alphas[i] / (self.alphas[i] + self.betas[i]) for i in range(self.n_arms)]


class BernoulliBandit(Bandit):
    def __init__(self, n, probas=None, seed=None):
        if seed is not None:
            np.random.seed(seed)
        assert probas is None or len(probas) == n
        self.n = n
        self.probas = probas if probas is not None else np.random.rand(n)
        self.best_proba = max(self.probas)

    def generate_reward(self, i):
        return np.random.rand() < self.probas[i]


# Example usage
if __name__ == "__main__":
    bernoulli_bandit = BernoulliBandit(4)
    beta_bandit = BetaBernoulliBandit(4)

    print("Initial Bernoulli Probabilities:", bernoulli_bandit.probas)
    print("Initial Beta Means:", beta_bandit.get_estimated_probas())

    for i in range(50):
        arm = np.random.choice(4)
        beta_bandit.generate_reward(arm)
        bernoulli_bandit.generate_reward(arm)
        # print(f"Updated Bernoulli Probabilities:{i}", bernoulli_bandit.probas)
        print(f"Updated Beta Means:{i}", beta_bandit.get_estimated_probas())

    print("Updated Beta Means:", beta_bandit.get_estimated_probas())


# class BernoulliBandit(Bandit):
#
#     def __init__(self, n, probas=None):
#         assert probas is None or len(probas) == n
#         self.n = n
#         if probas is None:
#             np.random.seed(int(time.time()))
#             self.probas = [np.random.random() for _ in range(self.n)]
#         else:
#             self.probas = probas
#
#         self.best_proba = max(self.probas)
#
#     def generate_reward(self, i):
#         # The player selected the i-th machine.
#         if np.random.random() < self.probas[i]:
#             return 1
#         else:
#             return 0