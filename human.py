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
        rand=np.random.rand()
        print("rand:", rand)
        if rand >= self.epsilon:
            success_rates = [self.calculate_success_rate(arm) for arm in range(self.n_arms)]
            print("actual_pulls:", self.actual_pulls)
            print("Success rates:", success_rates)
            print("argmax:", np.argmax(success_rates))
            return np.argmax(success_rates)

        else:
            return np.random.choice(self.n_arms)
    def calculate_success_rate(self, arm):
        return np.mean([r for i, r in zip(self.actual_pulls, self.rewards) if i == arm]) if self.actual_pulls.count(arm) > 0 else 0

    def get_arm_policy_distribution(self):
        return [self.calculate_success_rate(arm) for arm in range(self.n_arms)]

class WSLS(HumanPolicy):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.last_choice = None
        self.last_reward = None

    def select_arm(self):
        if self.last_choice is None or self.last_reward == 0:
            return np.random.choice(self.n_arms)
        else:
            return self.last_choice

    def update_choices(self, chosen_arm):
        self.last_choice = chosen_arm

    def update_rewards(self, robot_pull, reward):
        super().update_rewards(robot_pull, reward)
        if robot_pull == self.last_choice:
            self.last_reward = reward

class ThompsonSampling(HumanPolicy):
    def __init__(self, n_arms, alpha=1, beta=1):
        super().__init__(n_arms)
        self.alpha = [alpha] * n_arms
        self.beta = [beta] * n_arms

    def select_arm(self):
        samples = [np.random.beta(a, b) for a, b in zip(self.alpha, self.beta)]
        return np.argmax(samples)

    def update_rewards(self, robot_pull, reward):
        super().update_rewards(robot_pull, reward)
        if reward:
            self.alpha[robot_pull] += 1
        else:
            self.beta[robot_pull] += 1

class UCL(HumanPolicy):
    def __init__(self, n_arms, c=1):
        super().__init__(n_arms)
        self.c = c
        self.pulls = [0] * n_arms
        self.means = [0] * n_arms

    def select_arm(self):
        if 0 in self.pulls:
            return self.pulls.index(0)
        ucb_values = [mean + self.c * np.sqrt(np.log(sum(self.pulls)) / pull) for mean, pull in zip(self.means, self.pulls)]
        return np.argmax(ucb_values)

    def update_rewards(self, robot_pull, reward):
        super().update_rewards(robot_pull, reward)
        self.pulls[robot_pull] += 1
        self.means[robot_pull] = (self.means[robot_pull] * (self.pulls[robot_pull] - 1) + reward) / self.pulls[robot_pull]

class GittinsIndex(HumanPolicy):
    def __init__(self, n_arms, gamma=0.9):
        super().__init__(n_arms)
        self.gamma = gamma
        self.pulls = [0] * n_arms
        self.means = [0] * n_arms

    def select_arm(self):
        if 0 in self.pulls:
            return self.pulls.index(0)
        gittins_indices = [self.calculate_gittins_index(mean, pull) for mean, pull in zip(self.means, self.pulls)]
        return np.argmax(gittins_indices)

    def update_rewards(self, robot_pull, reward):
        super().update_rewards(robot_pull, reward)
        self.pulls[robot_pull] += 1
        self.means[robot_pull] = (self.means[robot_pull] * (self.pulls[robot_pull] - 1) + reward) / self.pulls[robot_pull]

    def calculate_gittins_index(self, mean, pull):
        # Simplified approximation of the Gittins index
        return mean + np.sqrt(2 * np.log(1 / (1 - self.gamma)) / pull)

class EpsilonOptimal(HumanPolicy):
    def __init__(self, n_arms, epsilon=0.1, reward_params=None):
        super().__init__(n_arms)
        self.epsilon = epsilon
        self.reward_params = reward_params

    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.n_arms)
        else:
            return np.argmax(self.reward_params)