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

    def update_human_choice(self, human_choice):
        self.human_observations.append(human_choice)

    def update_actual_pulls(self, robot_pull):
        self.actual_pulls.append(robot_pull)


    # def select_arm(self):
    #     # Simple strategy: follow the human's most common choice
    #     if self.human_observations:
    #         return np.argmax(np.bincount(self.human_observations))
    #     return np.random.choice(self.n_arms)

    def select_arm(self):
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
        # Prepare the input tensor, which could be a one-hot encoded vector
        # representing the previous choices of human and robot
        # This is a placeholder and will need to be tailored to your input structure
        return torch.tensor([human_observations[-1], actual_pulls[-1]], dtype=torch.float).view(1, -1)

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
