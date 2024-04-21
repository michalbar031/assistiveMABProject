import numpy as np
from rnn import RNN
import torch
import torch.nn as nn
class Robot:
    def __init__(self, n_arms,input_size):
        self.n_arms = n_arms
        self.human_observations = []
        self.actual_pulls = []
        hidden_size = 32
        self.rnn = RNN(input_size, hidden_size, n_arms)
        self.optimizer = torch.optim.Adam(self.rnn.parameters())
        self.criterion = nn.NLLLoss()
        self.approximate_reward_parameters = [1] * n_arms  # Start with a uniform guess


    def update_human_choice(self, human_choice):
        self.human_observations.append(human_choice)

    def update_actual_pulls(self, robot_pull):
        self.actual_pulls.append(robot_pull)


    def metropolis_hastings_update(self, current_params, likelihood_function, proposal_distribution, proposal_sampler, n_iterations):
        theta = current_params
        for _ in range(n_iterations):
            # Propose new sample
            theta_prime = proposal_sampler(theta)
            # Calculate acceptance ratio
            acceptance_ratio = min(1, (likelihood_function(theta_prime) * proposal_distribution(theta, theta_prime)) /
                                      (likelihood_function(theta) * proposal_distribution(theta_prime, theta)))
            # Accept or reject the new sample
            if np.random.rand() < acceptance_ratio:
                theta = theta_prime
        return theta

    def select_arm(self):
        # Approximate the posterior of reward parameters using Metropolis-Hastings
        self.approximate_reward_parameters = self.metropolis_hastings_update(
            self.approximate_reward_parameters,
            likelihood_function=self.calculate_likelihood_of_actions,
            proposal_distribution=self.calculate_proposal_pdf,
            proposal_sampler=self.sample_from_proposal,
            n_iterations=100  # for example
        )

        # Prepare the RNN input
        input_tensor = self.prepare_input_for_rnn()
        hidden = self.rnn.initHidden()
        output, hidden = self.rnn(input_tensor, hidden)

        # Use the output to decide on the arm to pull
        probabilities = torch.exp(output).detach().numpy().flatten()
        chosen_arm = np.random.choice(self.n_arms, p=probabilities)
        return chosen_arm
    def select_arm1(self):
        self.metropolis_hastings_update()
        # Prepare input tensor from human observations and actual pulls
        input_tensor = self.prepare_input(self.human_observations, self.actual_pulls)
        hidden = self.rnn.initHidden()

        # Forward pass through the RNN
        output, hidden = self.rnn(input_tensor, hidden)

        # Get the probability distribution over arms and sample an arm
        probabilities = torch.exp(output).detach().numpy().flatten()
        chosen_arm = np.random.choice(self.n_arms, p=probabilities)
        return chosen_arm

    def prepare_input(self, human_observations, actual_pulls):
        encoded_observation = torch.tensor(self.human_observations[-1], dtype=torch.float)
        encoded_parameters = torch.tensor(self.approximate_reward_parameters, dtype=torch.float)

        return torch.cat((encoded_observation.view(1, -1), encoded_parameters.view(1, -1)), 1)

    def update_train_rnn(self, rewards):
        # rewards is a list of the rewards obtained in each round
        loss = 0
        hidden = self.rnn.initHidden()
        for i in range(len(rewards)):
            input_tensor = self.prepare_input(self.human_observations[i], self.actual_pulls[i])
            output, hidden = self.rnn(input_tensor, hidden)
            loss += self.criterion(output, torch.tensor([rewards[i]], dtype=torch.long))

        # Perform backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()
