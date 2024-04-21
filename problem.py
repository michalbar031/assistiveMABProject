from bandit import BetaBernoulliBandit
from human import *
from robot import Robot
class Problem:
    def __init__(self, n_arms, human_policy,n_rounds=1000):
        self.bandit = BetaBernoulliBandit(n_arms)
        self.human = human_policy(n_arms)

        self.human_history_vector = [-1] * n_rounds
        self.robot_history_vector = [-1] * n_rounds
        self.human_policy_vector = self.get_policy_vector(self.human)
        self.n_rounds = n_rounds
        self.input_size = len(self.human_history_vector) + len(self.robot_history_vector)+len(self.human_policy_vector)
        self.robot = Robot(n_arms,self.input_size)
    def get_policy_vector(self, human_policy):
        # Assuming there are 5 policies in total and human_policy is an instance of one of them
        policy_vector = [0] * 5
        policy_names = [EpsilonGreedy, WSLS, ThompsonSampling, UCL, GittinsIndex]  # Add actual classes here
        # policy_names = ["EpsilonGreedy", "WSLS", "ThompsonSampling", "UCL", "GittinsIndex"]
        policy_index = policy_names.index(type(human_policy))
        policy_vector[policy_index] = 1
        return policy_vector

    def run_round(self, round_number):
        # print("---------------round----------------")
        #HUMAN
        human_choice = self.human.select_arm()
        self.human_history_vector[round_number] = human_choice
        self.human.update_choices(human_choice)
        self.robot.update_human_choice(human_choice)
        #ROBOT
        robot_choice = self.robot.select_arm()
        self.robot.update_actual_pulls(robot_choice)
        self.robot_history_vector[round_number] = robot_choice

        reward = self.bandit.generate_reward(robot_choice)
        self.human.update_rewards(robot_choice, reward)
        self.bandit.update(robot_choice, reward)

        rnn_input = self.human_history_vector + self.robot_history_vector + self.human_policy_vector
        self.robot.update_train_rnn(rnn_input, reward)

    def simulate(self):
        for round_number in range(self.n_rounds):
            self.run_round(round_number)


# Example usage
problem = Problem(4, EpsilonGreedy, n_rounds=1000)
problem.simulate()
