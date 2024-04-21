from bandit import BetaBernoulliBandit
from human import EpsilonGreedy
from robot import Robot
class Problem:
    def __init__(self, n_arms, human_policy):
        self.bandit = BetaBernoulliBandit(n_arms)
        self.human = human_policy(n_arms)
        self.robot = Robot(n_arms)

    def run_round(self):
        human_choice = self.human.select_arm()
        robot_choice = self.robot.select_arm()
        reward = self.bandit.generate_reward(robot_choice)
        self.human.update(human_choice, reward,robot_choice)
        self.bandit.update(robot_choice, reward)

        robot_choice = self.robot.select_arm()
        self.robot.update(human_choice)  # Robot sees human choice but not the reward

    def simulate(self, rounds):
        for _ in range(rounds):
            self.run_round()


# Example usage
problem = Problem(4, EpsilonGreedy)
problem.simulate(1000)
