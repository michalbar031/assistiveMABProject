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

class WSLS(HumanPolicy):
    def __init__(self, n_arms):
        super().__init__(n_arms)
        self.last_choice = None

    def select_arm(self):
        if self.last_choice is None:
            return np.random.choice(self.n_arms)
        else:
            return self.last_choice

    def update_choices(self, chosen_arm):
        self.last_choice = chosen_arm


class ThompsonSampling(HumanPolicy):
    pass

class UCL(HumanPolicy):
    pass

class GittinsIndex(HumanPolicy):
    pass

