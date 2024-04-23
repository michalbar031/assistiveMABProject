from mortal_bandit import MortalBandit
from mortal_human_policy import MortalHumanPolicy
from mortal_robot import MortalRobot

class MortalProblem:
    def __init__(self, n_arms, lifetimes, human_policy, n_rounds=1000):
        self.bandit = MortalBandit(n_arms, lifetimes)
        self.human = human_policy(n_arms)
        self.human_policy = human_policy
        self.human_history_vector = [-1] * n_rounds
        self.robot_history_vector = [-1] * n_rounds
        self.n_rounds = n_rounds
        self.input_size = 2
        self.robot = MortalRobot(n_arms, lifetimes, self.input_size)

    def run_round(self, round_number):
        pass

    def simulate(self):
        pass