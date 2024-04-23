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


# class PPO:
#     def __init__(self, policy_network, clip_epsilon=0.2, ppo_epochs=10, mini_batch_size=64):
#         self.policy_network = policy_network
#         self.optimizer = optim.Adam(policy_network.parameters(), lr=1e-3)
#         self.clip_epsilon = clip_epsilon
#         self.ppo_epochs = ppo_epochs
#         self.mini_batch_size = mini_batch_size
#         self.MseLoss = nn.MSELoss()
#
#
#     def sample_trajectories(pi, H, theta, size, T):
#         all_trajectories = []
#         for _ in range(size):
#             trajectory = []
#             # Initial action and observation
#             a_t_minus_1 = None  # Assuming None for the first action
#             h_t = None  # Assuming None for the first human action
#
#             for t in range(T):
#                 # Sample human action based on policy H
#                 h_t = H.sample_action(h_t, a_t_minus_1)
#                 # Prepare input for RNN
#                 input_tensor = prepare_input_for_rnn(a_t_minus_1, h_t)
#                 hidden = pi.initHidden()
#                 # Get action probabilities from RNN
#                 action_probs, hidden = pi(input_tensor, hidden)
#                 # Sample action from the probability distribution
#                 m = Categorical(action_probs)
#                 a_t = m.sample().item()
#                 # Simulate reward from the MAB
#                 r_t = np.random.binomial(1, theta[a_t])
#                 # Store the transition
#                 trajectory.append((h_t, a_t, r_t))
#                 # Update the previous action
#                 a_t_minus_1 = a_t
#
#             all_trajectories.append(trajectory)
#         return all_trajectories
#
#     def select_action(self, state):
#         state = torch.FloatTensor(state).unsqueeze(0)
#         with torch.no_grad():
#             action_probs = self.policy_network(state)
#         distribution = torch.distributions.Categorical(action_probs)
#         action = distribution.sample()
#         return action.item(), distribution.log_prob(action)
#
#     def ppo_update(policy_network, optimizer, all_trajectories, clip_epsilon, ppo_epochs, mini_batch_size):
#         for _ in range(ppo_epochs):
#             # Assume 'generate_batches' is a function to create mini-batches from 'all_trajectories'
#             for states, actions, old_log_probs, returns, advantages in generate_batches(all_trajectories,
#                                                                                         mini_batch_size):
#                 # ... (rest of the PPO update code as mentioned earlier)
#                 # Remember to compute advantages and returns as needed
#                 pass
#
#     def train_policy(pi, H, ppo_epochs, sample_size, T, theta_prior):
#         for _ in range(ppo_epochs):
#             theta = np.random.choice(theta_prior, size=sample_size)  # Sample from prior
#             all_trajectories = sample_trajectories(pi, H, theta, sample_size, T)
#             ppo_update(pi, optimizer, all_trajectories, clip_epsilon=0.2, ppo_epochs=10, mini_batch_size=64)
#
#     # Assuming 'H' is your human policy class with a 'sample_action' method and 'pi' is your policy network
#     # Also assuming 'theta_prior' is an array of possible theta values for the arms
#     # Run the training
#     # train_policy(pi, H, ppo_epochs=50, sample_size=100, T=50, theta_prior=[0.1, 0.2, 0.3, 0.4])
#
#     def update(self, memory):
#         for _ in range(self.ppo_epochs):
#             for states, actions, old_log_probs, returns, advantages in memory.generate_batches(self.mini_batch_size):
#                 # Get the policy's action log probabilities and state values
#                 pi, value = self.policy_network(states)
#                 new_log_probs = pi.log_prob(actions)
#                 state_values = value.squeeze()
#
#                 # Compute the ratio between new and old policy probabilities
#                 ratios = torch.exp(new_log_probs - old_log_probs)
#
#                 # Compute the PPO objective
#                 surr1 = ratios * advantages
#                 surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
#                 policy_loss = -torch.min(surr1, surr2).mean()
#
#                 # Compute value loss
#                 value_loss = self.MseLoss(state_values, returns)
#
#                 # Combine policy and value losses if needed
#                 loss = policy_loss + value_loss
#
#                 # Perform a policy update using gradient ascent
#                 self.optimizer.zero_grad()
#                 loss.backward()
#                 self.optimizer.step()
#
#     def update_old(self, memory):
#         # Assume `memory` is an object that has method `generate_batches`, which yields mini-batches of experience
#         for _ in range(self.ppo_epochs):
#             for states, actions, old_log_probs, returns, advantages in memory.generate_batches(self.mini_batch_size):
#                 # Evaluate new log probabilities and values using current policy
#                 new_log_probs = self.policy_network(states).log_prob(actions)
#                 ratios = torch.exp(new_log_probs - old_log_probs)
#
#                 # Clipped objective function
#                 surr1 = ratios * advantages
#                 surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
#                 policy_loss = -torch.min(surr1, surr2).mean()
#
#                 # Perform policy update
#                 self.optimizer.zero_grad()
#                 policy_loss.backward()
#                 self.optimizer.step()

class Robot:
    def __init__(self, n_arms,input_size,hidden_size=32,trajectories_number_sample=10):
        self.n_arms = n_arms
        self.human_observations = []
        self.actual_pulls = []
        self.hidden_size = hidden_size
        self.rnn = RNN(input_size, hidden_size, n_arms)
        self.ppo = PPO(self.rnn)
        self.trajectories = []
        self.optimizer = torch.optim.Adam(self.rnn.parameters())
        self.criterion = nn.NLLLoss()
        self.trajectories_number_sample = trajectories_number_sample
        # self.approximate_reward_parameters = [1] * n_arms  # Start with a uniform guess
        self.approximate_reward_parameters = np.random.rand(n_arms)  # Initial guess for theta

    def select_arm(self, human_choice_history, robot_choice_history):
        input_sequence = self.prepare_input_sequence(human_choice_history, robot_choice_history)
        arm_distribution = self.rnn(input_sequence)
        chosen_arm = torch.argmax(arm_distribution, dim=1).item()
        return chosen_arm

    def prepare_input_sequence(self, human_choice_history, robot_choice_history):
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
                else:
                    human_choice = sample_human.select_arm()
                    sample_human.update_choices(human_choice)
                    r = sample_bandit.generate_reward(human_choice)
                    human_choice_history.append(human_choice)
                    robot_choice = self.select_arm(human_choice_history, robot_choice_history)
                    robot_choice_history.append(robot_choice)

                trajectories.append((robot_choice, human_choice, r))

        return trajectories

    def train(self):
        trajectories = self.sample_trajectories(self.alphas, self.betas, self.T, self.human_policy)
        self.ppo.update(trajectories)

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

