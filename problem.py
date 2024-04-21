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
        self.human.update_choices(human_choice)
        self.robot.update_human_choice(human_choice)

        robot_choice = self.robot.select_arm()
        self.robot.update_actual_pulls(robot_choice)
        reward = self.bandit.generate_reward(robot_choice)
        self.human.update_rewards(robot_choice, reward)
        self.bandit.update(robot_choice, reward)


    def simulate(self, rounds):
        for _ in range(rounds):
            self.run_round()


# Example usage
problem = Problem(4, EpsilonGreedy)
problem.simulate(1000)
