import numpy as np
from rnn import RNN
import torch
import torch.nn as nn
import torch
import torch.nn as nn
import numpy as np
from torch.distributions import Categorical
from ppo import PPO
import torch.optim as optim
from bandit import BernoulliBandit

class Robot:
    def __init__(self, n_arms,input_size,hidden_size=32,trajectories_number_sample=10):
        self.n_arms = n_arms
        self.human_observations = []
        self.actual_pulls = []
        self.hidden_size = hidden_size
        self.rnn = RNN(input_size, hidden_size, 1)
        self.ppo = PPO(self.rnn)
        self.trajectories = []
        self.optimizer = torch.optim.Adam(self.rnn.parameters())
        self.criterion = nn.NLLLoss()
        self.trajectories_number_sample = trajectories_number_sample
        # self.approximate_reward_parameters = [1] * n_arms  # Start with a uniform guess
        self.approximate_reward_parameters = np.random.rand(n_arms)  # Initial guess for theta

    def select_arm(self, human_choice_history, robot_choice_history):
        # Assume there is at least one prior action
        if len(robot_choice_history) == 0:
            robot_choice_history.append(0)  # Default starting action
        if len(human_choice_history) == 0:
            human_choice_history.append(0)  # Default starting action

        input_pair = torch.tensor([[robot_choice_history[-1], human_choice_history[-1]]], dtype=torch.float)
        input_sequence = input_pair.unsqueeze(0)  # Batch size of 1
        probabilities = self.rnn(input_sequence)  # Forward pass through the RNN
        chosen_arm = torch.argmax(probabilities).item()  # Select arm with the highest probability
        return chosen_arm
    def select_arm111(self, human_choice_history, robot_choice_history):
        input_sequence = self.prepare_input_sequence(human_choice_history, robot_choice_history)
        print("Input sequence shape:", input_sequence.shape)

        log_probs = self.rnn(input_sequence)
        probabilities = torch.exp(log_probs).squeeze(0)

        # Choose the arm with the highest probability
        chosen_arm = torch.argmax(probabilities).item()
        return chosen_arm

    def prepare_input_sequence(self, human_choice_history, robot_choice_history):
        sequence_length = len(human_choice_history)
        input_sequence = torch.zeros(1, sequence_length, 2)
        print("Input sequence shape:", input_sequence.shape)

        for i in range(sequence_length):
            if i == 0:
                input_sequence[0, i, 0] = 0  # Assign a default value for the first robot choice
            else:
                input_sequence[0, i, 0] = robot_choice_history[i - 1]
            input_sequence[0, i, 1] = human_choice_history[i]

        return input_sequence

    def select_armg(self, human_choice_history, robot_choice_history):
        # If there is no previous history, use a default value of 0
        last_robot_choice = robot_choice_history[-1] if robot_choice_history else 0
        last_human_choice = human_choice_history[-1] if human_choice_history else 0

        # Prepare the input tensor as a new sequence
        input_pair = torch.tensor([[last_robot_choice, last_human_choice]], dtype=torch.float)

        # Reshape input to match RNN expectations (batch_size, sequence_length, input_size)
        input_sequence = input_pair.unsqueeze(0).unsqueeze(0)
        print("Input sequence shape:", input_sequence.shape)

        # Forward pass through RNN
        log_probs = self.rnn(input_sequence)
        probabilities = torch.exp(log_probs).squeeze(0)

        # Choose the arm with the highest probability
        chosen_arm = torch.argmax(probabilities).item()
        return chosen_arm
    def select_armk(self, human_choice, robot_choice):
        print("--select_arm--")
        input_pair = torch.tensor([[robot_choice, human_choice]], dtype=torch.float).unsqueeze(0)
        print("!!!!!input_sequence:", input_pair)
        log_probs = self.rnn(input_pair)
        probabilities = torch.exp(log_probs).squeeze(0)

        # Choose the arm with the highest probability
        chosen_arm = torch.argmax(probabilities).item()
        return chosen_arm
    def select_arm2(self, human_choice_history, robot_choice_history):
        print("--select_arm--")
        input_pair = torch.tensor([[robot_choice_history[-1], human_choice_history[-1]]], dtype=torch.float)

        # input_sequence = self.prepare_input_sequence(human_choice_history, robot_choice_history)
        print("!!!!!input_sequence:", input_pair)
        log_probs = self.rnn(input_pair)
        probabilities = torch.exp(log_probs).squeeze(0)

        # Choose the arm with the highest probability
        chosen_arm = torch.argmax(probabilities).item()
        return chosen_arm

    def prepare_input_sequence1(self, human_choice_history, robot_choice_history):
        print("--prepare_input_sequence--")
        sequence_length = len(human_choice_history)
        input_sequence = torch.zeros(1, sequence_length, 2)

        for i in range(sequence_length):
            if i == 0:
                input_sequence[0, i, 0] = 0  # Assign a default value for the first robot choice
            else:
                input_sequence[0, i, 0] = robot_choice_history[i - 1]
            input_sequence[0, i, 1] = human_choice_history[i]

        return input_sequence


    def sample_trajectories(self, alphas, betas, T, human_policy):
        print("--sample_trajectories--")
        trajectories = []
        for _ in range(self.trajectories_number_sample):
            theta = [np.random.beta(alpha, beta) for alpha, beta in zip(alphas, betas)]
            sample_bandit = BernoulliBandit(self.n_arms, probas=theta)
            sample_human = human_policy(self.n_arms)

            trajectory = []

            for t in range(T):
                if t == 0:
                    human_choice = self.human_observations[t]
                    r = sample_bandit.generate_reward(human_choice)
                    sample_human.update_choices(human_choice)
                    robot_choice = human_choice
                else:
                    human_choice = sample_human.select_arm()
                    sample_human.update_choices(human_choice)
                    r = sample_bandit.generate_reward(human_choice)
                    robot_choice = self.select_arm([human_choice], [robot_choice])

                print("ROBOT")
                print("Robot choice:", robot_choice)
                print("Human choice:", human_choice)
                trajectory.append((robot_choice, human_choice, r))

            trajectories.append(trajectory)

        return trajectories
    def sample_trajectories1(self, alphas, betas, T, human_policy):
        print("--sample_trajectories--")
        trajectories = []
        for _ in range(self.trajectories_number_sample):
            theta = [np.random.beta(alpha, beta) for alpha, beta in zip(alphas, betas)]
            sample_bandit = BernoulliBandit(self.n_arms, probas=theta)
            sample_human = human_policy(self.n_arms)

            human_choice_history = []
            robot_choice_history = []

            for t in range(T):
                if t == 0:
                    human_choice = self.human_observations[t]
                    r = sample_bandit.generate_reward(human_choice)
                    human_choice_history.append(human_choice)
                    robot_choice_history.append(human_choice)
                    sample_human.update_choices(human_choice)
                    robot_choice = human_choice
                else:
                    human_choice = sample_human.select_arm()
                    sample_human.update_choices(human_choice)
                    r = sample_bandit.generate_reward(human_choice)
                    human_choice_history.append(human_choice)
                    robot_choice = self.select_arm(human_choice_history, robot_choice_history)
                    robot_choice_history.append(robot_choice)

                trajectories.append((robot_choice, human_choice, r))

        return trajectories

    def sample_trajectories_old(self,alphas,betas, T,human_policy):
        for _ in range(self.trajectories_number_sample):
            all_trajectories = []
            theta=[]
            for alpha,beta in zip(alphas,betas):
                theta.append(np.random.beta(alpha,beta))
                # reward = np.random.rand() < theta_i
            sample_bandit=BernoulliBandit(self.n_arms,probas=theta)
            sample_human=human_policy(self.n_arms)
            pairs_for_rnn = []
            ht=None
            at_minus_1=None
            a=0
            h=0
            for t in range(T):
                if t==0:
                    human_choice = self.human_observations[t]
                    r=sample_bandit.generate_reward(human_choice)
                    all_trajectories.append((human_choice, human_choice, r))
                    sample_human.update_choices(human_choice)
                    at_minus_1 = human_choice
                    a+=1
                else:
                    human_choice = sample_human.select_arm()
                    sample_human.update_choices(human_choice)
                    r = sample_bandit.generate_reward(human_choice)
                    ht = human_choice
                    h+=1
                    if a==h:
                        pairs_for_rnn.append((at_minus_1, ht))
                    robot_choice= self.select_arm(pairs_for_rnn)
                    at_minus_1 = robot_choice
                    a+=1
                    all_trajectories.append((robot_choice, human_choice, r))
            self.trajectories.append(all_trajectories)

    def update_human_choice(self, human_choice):
        self.human_observations.append(human_choice)

    def update_actual_pulls(self, robot_pull):
        self.actual_pulls.append(robot_pull)

    # def calculate_likelihood_of_actions(self, theta):
    #     likelihood = 1.0
    #     for (arm_index, reward) in zip(self.actual_pulls, self.approximate_reward_parameters):
    #         prob = theta[arm_index] if reward == 1 else (1 - theta[arm_index])
    #         likelihood *= prob
    #     return likelihood
    #
    # def calculate_proposal_pdf(self, theta_prime, theta):
    #     # Assuming a symmetric normal distribution for proposal; this simplifies to 1 because
    #     # the density function for theta given theta_prime is the same as for theta_prime given theta
    #     return 1.0
    #
    # def sample_from_proposal(self, theta):
    #     # Sampling from a normal distribution centered at current theta values
    #     std_dev = 0.05  # Small standard deviation
    #     theta_prime = np.random.normal(theta, std_dev)
    #     return np.clip(theta_prime, 0, 1)  # Ensure thetas are within valid probability bounds
    #


    # def metropolis_hastings_update(self, current_params, likelihood_function, proposal_distribution, proposal_sampler, n_iterations):
    #     theta = current_params
    #     for _ in range(n_iterations):
    #         # Propose new sample
    #         theta_prime = proposal_sampler(theta)
    #         # Calculate acceptance ratio
    #         acceptance_ratio = min(1, (likelihood_function(theta_prime) * proposal_distribution(theta, theta_prime)) /
    #                                   (likelihood_function(theta) * proposal_distribution(theta_prime, theta)))
    #         # Accept or reject the new sample
    #         if np.random.rand() < acceptance_ratio:
    #             theta = theta_prime
    #     return theta




    # def select_arm_old(self):
    #     # Approximate the posterior of reward parameters using Metropolis-Hastings
    #     self.approximate_reward_parameters = self.metropolis_hastings_update(
    #         self.approximate_reward_parameters,
    #         likelihood_function=self.calculate_likelihood_of_actions,
    #         proposal_distribution=self.calculate_proposal_pdf,
    #         proposal_sampler=self.sample_from_proposal,
    #         n_iterations=100  # for example
    #     )
    #
    #     # Prepare the RNN input
    #     input_tensor = self.prepare_input_for_rnn()
    #     hidden = self.rnn.initHidden()
    #     output, hidden = self.rnn(input_tensor, hidden)
    #
    #     # Use the output to decide on the arm to pull
    #     probabilities = torch.exp(output).detach().numpy().flatten()
    #     chosen_arm = np.random.choice(self.n_arms, p=probabilities)
    #     return chosen_arm
    # def select_arm1(self):
    #     self.metropolis_hastings_update()
    #     # Prepare input tensor from human observations and actual pulls
    #     input_tensor = self.prepare_input(self.human_observations, self.actual_pulls)
    #     hidden = self.rnn.initHidden()
    #
    #     # Forward pass through the RNN
    #     output, hidden = self.rnn(input_tensor, hidden)
    #
    #     # Get the probability distribution over arms and sample an arm
    #     probabilities = torch.exp(output).detach().numpy().flatten()
    #     chosen_arm = np.random.choice(self.n_arms, p=probabilities)
    #     return chosen_arm

    # def prepare_input(self, human_policy_input):
    #     return torch.tensor(human_policy_input, dtype=torch.float).unsqueeze(0)

    # def update_train_rnn(self, rewards):
    #     # rewards is a list of the rewards obtained in each round
    #     loss = 0
    #     hidden = self.rnn.initHidden()
    #     for i in range(len(rewards)):
    #         input_tensor = self.prepare_input(self.human_observations[i], self.actual_pulls[i])
    #         output, hidden = self.rnn(input_tensor, hidden)
    #         loss += self.criterion(output, torch.tensor([rewards[i]], dtype=torch.long))
    #
    #     # Perform backpropagation
    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     self.optimizer.step()
    #
    #     return loss.item()

