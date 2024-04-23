import torch
import torch.nn as nn
import torch.optim as optim

class PPO:
    def __init__(self, policy_network, clip_epsilon=0.2, ppo_epochs=10, mini_batch_size=64, learning_rate=1e-3, gamma=0.9):
        self.policy_network = policy_network
        self.optimizer = optim.Adam(policy_network.parameters(), lr=learning_rate)
        self.clip_epsilon = clip_epsilon
        self.ppo_epochs = ppo_epochs
        self.mini_batch_size = mini_batch_size
        self.MseLoss = nn.MSELoss()
        self.gamma = gamma

    def update(self, trajectories):
        states, actions, old_log_probs, rewards, advantages = self.process_trajectories(trajectories)

        for _ in range(self.ppo_epochs):
            for batch_indices in self.generate_batches(len(states), self.mini_batch_size):
                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_old_log_probs = old_log_probs[batch_indices]
                batch_advantages = advantages[batch_indices]

                # Forward pass
                log_probs = self.policy_network(batch_states)
                action_log_probs = log_probs.gather(1, batch_actions.unsqueeze(1)).squeeze(1)

                # Compute the ratio
                ratios = torch.exp(action_log_probs - batch_old_log_probs)

                # Compute the surrogate losses
                surr1 = ratios * batch_advantages
                surr2 = torch.clamp(ratios, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_advantages
                loss = -torch.min(surr1, surr2).mean()

                # Backward pass and optimization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def process_trajectories(self, trajectories):
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        advantages = []

        for trajectory in trajectories:
            # Extract states, actions, and rewards from the trajectory
            # (assuming trajectory is a list of (state, action, reward) tuples)
            states_t, actions_t, rewards_t = zip(*trajectory)

            # Convert states and actions to tensors
            states_t = torch.tensor(states_t, dtype=torch.float)
            actions_t = torch.tensor(actions_t, dtype=torch.long)

            # Compute log probabilities of actions
            log_probs_t = self.policy_network(states_t)
            log_probs_t = log_probs_t.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            # Compute advantages using the rewards and the value function V(s)
            # You can use any advantage estimation method here, such as Generalized Advantage Estimation (GAE)
            # For simplicity, we'll use the rewards-to-go as advantages
            advantages_t = self.compute_advantages(rewards_t)

            states.extend(states_t)
            actions.extend(actions_t)
            old_log_probs.extend(log_probs_t.detach())
            rewards.extend(rewards_t)
            advantages.extend(advantages_t)

        states = torch.stack(states)
        actions = torch.stack(actions)
        old_log_probs = torch.stack(old_log_probs)
        rewards = torch.tensor(rewards, dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)

        return states, actions, old_log_probs, rewards, advantages

    def compute_advantages(self, rewards):
        # Compute advantages using rewards-to-go
        advantages = []
        reward_to_go = 0
        for reward in reversed(rewards):
            reward_to_go = reward + self.gamma * reward_to_go
            advantages.insert(0, reward_to_go)
        return torch.tensor(advantages, dtype=torch.float)



    def process_trajectories1(self, trajectories):
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        advantages = []

        for trajectory in trajectories:
            # Extract states, actions, and rewards from the trajectory
            # (assuming trajectory is a list of (state, action, reward) tuples)
            states_t, actions_t, rewards_t = zip(*trajectory)

            # Compute advantages using the rewards and the value function V(s)
            # This requires implementing a function to estimate V(s), or you could train a separate network for it
            advantages_t = self.compute_advantages(rewards_t, states_t)

            states.extend(states_t)
            actions.extend(actions_t)
            rewards.extend(rewards_t)
            advantages.extend(advantages_t)

            # Compute old log probabilities of actions, which requires you to have saved the log probs during trajectory generation
            old_log_probs.extend(self.get_log_probs(states_t, actions_t))

        states = torch.tensor(states, dtype=torch.float)
        actions = torch.tensor(actions, dtype=torch.long)
        old_log_probs = torch.tensor(old_log_probs, dtype=torch.float)
        rewards = torch.tensor(rewards, dtype=torch.float)
        advantages = torch.tensor(advantages, dtype=torch.float)

        return states, actions, old_log_probs, rewards, advantages

    def generate_batches(self, data_size, batch_size):
        indices = torch.randperm(data_size).tolist()
        for i in range(0, data_size, batch_size):
            yield indices[i:i+batch_size]