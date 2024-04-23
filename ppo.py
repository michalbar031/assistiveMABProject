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

    def updateGPT(self, trajectories):
        for _ in range(self.ppo_epochs):
            for states, actions, old_log_probs, returns, advantages in self.generate_batches(self.process_trajectories(trajectories),
                                                                                             self.mini_batch_size):
                # Calculate action probabilities and entropy
                log_probs = self.policy_network(states)
                action_log_probs = log_probs.gather(1, actions.unsqueeze(1)).squeeze(1)
                ratios = torch.exp(action_log_probs - old_log_probs)

                # Compute the PPO objective
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantages
                policy_loss = -torch.min(surr1, surr2).mean()

                # Perform backward pass and optimization
                self.optimizer.zero_grad()
                policy_loss.backward()
                self.optimizer.step()

    def generate_batches1GPT(self, trajectories, batch_size):
        # Unpack trajectories
        states, actions, old_log_probs, returns, advantages = zip(*trajectories)
        total_size = len(states)
        indices = torch.randperm(total_size)
        for i in range(0, total_size, batch_size):
            batch_indices = indices[i:i + batch_size]
            yield states[batch_indices], actions[batch_indices], old_log_probs[batch_indices], returns[batch_indices], \
            advantages[batch_indices]
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
                action_log_probs = log_probs.gather(2, batch_actions.unsqueeze(2)).squeeze(2)

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
            states_traj, actions_traj, rewards_traj = zip(*trajectory)

            states_t = torch.tensor(states_traj, dtype=torch.float).unsqueeze(0)
            actions_t = torch.tensor(actions_traj, dtype=torch.long)
            rewards_t = torch.tensor(rewards_traj, dtype=torch.float)

            log_probs_all = self.policy_network(states_t)
            log_probs_t = log_probs_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            advantages_t = self.compute_advantages(rewards_t)

            states.append(states_t)
            actions.append(actions_t)
            old_log_probs.append(log_probs_t.detach())
            rewards.append(rewards_t)
            advantages.append(advantages_t)

        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        old_log_probs = torch.cat(old_log_probs, dim=0)
        rewards = torch.cat(rewards, dim=0)
        advantages = torch.cat(advantages, dim=0)

        return states, actions, old_log_probs, rewards, advantages
    def process_trajectoriesk(self, trajectories):
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        advantages = []

        for trajectory in trajectories:
            states_traj, actions_traj, rewards_traj = zip(*trajectory)

            states_t = torch.tensor(states_traj, dtype=torch.float).unsqueeze(0)
            actions_t = torch.tensor(actions_traj, dtype=torch.long)
            rewards_t = torch.tensor(rewards_traj, dtype=torch.float)

            log_probs_all = self.policy_network(states_t)
            log_probs_t = log_probs_all.gather(2, actions_t.view(1, -1, 1)).squeeze(2)

            advantages_t = self.compute_advantages(rewards_t)
            print("States shape:", states_t.shape)
            print("Actions shape:", actions_t.shape)
            print("Log probs shape:", log_probs_t.shape)
            print("Advantages shape:", advantages_t.shape)
            states.append(states_t)
            actions.append(actions_t)
            old_log_probs.append(log_probs_t.detach())
            rewards.append(rewards_t)
            advantages.append(advantages_t)

        states = torch.cat(states, dim=1)
        actions = torch.cat(actions, dim=0)
        old_log_probs = torch.cat(old_log_probs, dim=0)
        rewards = torch.cat(rewards, dim=0)
        advantages = torch.cat(advantages, dim=0)
        print("Concatenated states shape:", states.shape)
        print("Concatenated actions shape:", actions.shape)
        print("Concatenated old log probs shape:", old_log_probs.shape)
        print("Concatenated rewards shape:", rewards.shape)
        print("Concatenated advantages shape:", advantages.shape)
        return states, actions, old_log_probs, rewards, advantages
    def update2(self, trajectories):
        states, actions, old_log_probs, rewards, advantages = self.process_trajectories(trajectories)
        print("flag3")
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

    def process_trajectories2(self, trajectories):
        # Initialize lists to hold processed trajectory data
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        advantages = []

        # Loop over each trajectory to process
        for trajectory in trajectories:
            # Extract states, actions, and rewards from the trajectory
            states_traj, actions_traj, rewards_traj = zip(*trajectory)

            # Convert each component to a tensor
            states_t = torch.tensor(states_traj, dtype=torch.float).unsqueeze(
                1)  # Assumes states_traj is a list of scalars
            actions_t = torch.tensor(actions_traj, dtype=torch.long)
            rewards_t = torch.tensor(rewards_traj, dtype=torch.float)

            log_probs_all = self.policy_network(states_t)
            log_probs_t = log_probs_all.gather(1, actions_t.view(-1, 1)).squeeze(1)

            advantages_t = self.compute_advantages(rewards_t)

            # Store the tensors
            states.append(states_t)
            actions.append(actions_t)
            old_log_probs.append(log_probs_t.detach())
            rewards.append(rewards_t)
            advantages.append(advantages_t)

        # Once all trajectories are processed, concatenate the list of tensors
        states = torch.cat(states, dim=0)
        actions = torch.cat(actions, dim=0)
        old_log_probs = torch.cat(old_log_probs, dim=0)
        rewards = torch.cat(rewards, dim=0)
        advantages = torch.cat(advantages, dim=0)

        # Return the concatenated tensors
        return states, actions, old_log_probs, rewards, advantages
    def process_trajectories1(self, trajectories):
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        advantages = []

        for trajectory in trajectories:
            states_t, actions_t, rewards_t = zip(*trajectory)

            states_t = torch.tensor([s[0] for s in states_t], dtype=torch.float).unsqueeze(1)
            actions_t = torch.tensor([a for a in actions_t], dtype=torch.long)
            rewards_t = torch.tensor([r for r in rewards_t], dtype=torch.float)
            # advantages_t = self.compute_advantages(rewards_t)
            # states_t = torch.tensor(states_t, dtype=torch.float)

            # states_t = torch.tensor(states, dtype=torch.float)
            # actions_t = torch.tensor(actions_t, dtype=torch.long)

            # Add a dimension to states_t if it's a 1D tensor
            if states_t.dim() == 1:
                states_t = states_t.unsqueeze(0)
            print("Shape of states_t:", states_t.shape)
            log_probs_t = self.policy_network(states_t)
            log_probs_t = log_probs_t.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            advantages_t = self.compute_advantages(rewards_t)

            states.append(states_t)
            actions.append(actions_t)
            old_log_probs.append(log_probs_t.detach())
            rewards.append(rewards_t)
            advantages.append(advantages_t)

        max_seq_length = max(s.shape[1] for s in states)
        padded_states = [torch.nn.functional.pad(s, (0, 0, 0, max_seq_length - s.size(1))) for s in states]
        # states = torch.cat(states)
        states = torch.cat(padded_states, dim=0)  # Concatenate along batch dimension
        actions = torch.cat(actions)
        old_log_probs = torch.cat(old_log_probs)
        rewards = torch.cat([torch.tensor(r, dtype=torch.float) for r in rewards])
        advantages = torch.cat(advantages)

        return states, actions, old_log_probs, rewards, advantages
    def process_trajectories4(self, trajectories):
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        advantages = []

        for trajectory in trajectories:
            states_t, actions_t, rewards_t = zip(*trajectory)

            states_t = torch.tensor(states_t, dtype=torch.float)
            actions_t = torch.tensor(actions_t, dtype=torch.long)

            # Add a dimension to states_t if it's a 1D tensor
            if states_t.dim() == 1:
                states_t = states_t.unsqueeze(0)
            print("!!!!!self.policy_network:", self.policy_network)
            print("Shape of states_t:", states_t.shape)
            log_probs_t = self.policy_network(states_t)
            log_probs_t = log_probs_t.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            advantages_t = self.compute_advantages(rewards_t)

            states.append(states_t)
            actions.append(actions_t)
            old_log_probs.append(log_probs_t.detach())
            rewards.append(rewards_t)
            advantages.append(advantages_t)

        states = torch.cat(states)
        actions = torch.cat(actions)
        old_log_probs = torch.cat(old_log_probs)
        rewards = torch.cat([torch.tensor(r, dtype=torch.float) for r in rewards])
        advantages = torch.cat(advantages)

        return states, actions, old_log_probs, rewards, advantages
    def process_trajectories3(self, trajectories):
        states = []
        actions = []
        old_log_probs = []
        rewards = []
        advantages = []

        for trajectory in trajectories:
            states_t, actions_t, rewards_t = zip(*trajectory)

            states_t = torch.tensor(states_t, dtype=torch.float)
            actions_t = torch.tensor(actions_t, dtype=torch.long)

            log_probs_t = self.policy_network(states_t)
            log_probs_t = log_probs_t.gather(1, actions_t.unsqueeze(1)).squeeze(1)

            advantages_t = self.compute_advantages(rewards_t)

            states.append(states_t)
            actions.append(actions_t)
            old_log_probs.append(log_probs_t.detach())
            rewards.append(rewards_t)
            advantages.append(advantages_t)

        states = torch.cat(states)
        actions = torch.cat(actions)
        old_log_probs = torch.cat(old_log_probs)
        rewards = torch.cat([torch.tensor(r, dtype=torch.float) for r in rewards])
        advantages = torch.cat(advantages)

        return states, actions, old_log_probs, rewards, advantages
    def process_trajectories2(self, trajectories):
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