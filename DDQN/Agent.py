import math
from collections import namedtuple, deque
import random

import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import gymnasium as gym

# Device configuration
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

# Hyperparameters
BATCH_SIZE = 128
GAMMA = 0.7
EPS_START = 0.9
EPS_END = 0.05
EPS_DECAY = 1000
TAU = 0.005

class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.conv2 = nn.Conv2d(16, 32, 3)
        self.dropout1 = nn.Dropout(0.15)
        
        # Shared feature extractor
        self.fc_features = nn.Linear(800, 128)
        
        # Action heads using ModuleList for easier iteration
        self.action_heads = nn.ModuleList([
            nn.Linear(128, n_actions[i]) for i in range(len(n_actions))
        ])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        
        # Shared features
        features = F.relu(self.fc_features(x))
        
        # Get all action outputs using list comprehension
        actions = [head(features) for head in self.action_heads]
        
        return actions

Transition = namedtuple('Transition',
                       ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class Agent:
    def __init__(self):
        self.lr = 1e-4
        self.env = gym.make('AssemblyLine',disable_env_checker=True)
        self.n_actions = [space.n for space in self.env.action_space.spaces]
        
        # Initialize networks
        self.policy_net = DQN(self.n_actions).to(device)
        self.target_net = DQN(self.n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(10000)
        self.episode_durations = []
        self.episode_rewards = []

    def turn(self, state, steps_done):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
        #print('turn',(sample > eps_threshold))
        if sample > eps_threshold:
            with torch.no_grad():
                # Get all actions from policy network
                action_values = self.policy_net(state)
                # Select best action for each output using list comprehension
                action = torch.tensor([
                    values.max(1)[1].item() for values in action_values
                ], device=device)
                return action
        else:
            # Random actions using list comprehension
            return torch.tensor([
                random.randint(0, n - 1) for n in self.n_actions
            ], device=device)

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        
        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                              batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        
        state_batch = torch.cat(batch.state)
        action_batch = torch.stack(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Get current Q values for all actions
        current_q_values = self.policy_net(state_batch)
        
        # Initialize lists to store values
        state_action_values = []
        next_state_values = [torch.zeros(BATCH_SIZE, device=device) for _ in self.n_actions]
        expected_state_action_values = []
        
        # Compute Q values for each action
        for i in range(len(self.n_actions)):
            state_action_values.append(
                current_q_values[i].gather(1, action_batch[:, i].unsqueeze(1))
            )

        # Compute next state values using target network
        with torch.no_grad():
            next_q_values = self.target_net(non_final_next_states)
            for i in range(len(self.n_actions)):
                next_state_values[i][non_final_mask] = next_q_values[i].max(1)[0]

        # Compute expected Q values
        expected_state_action_values = [
            (next_state_values[i] * GAMMA) + reward_batch
            for i in range(len(self.n_actions))
        ]

        # Compute Huber loss for each action
        criterion = nn.SmoothL1Loss()
        losses = [
            criterion(state_action_values[i], expected_state_action_values[i].unsqueeze(1))
            for i in range(len(self.n_actions))
        ]
        
        # Combined loss
        total_loss = sum(losses)

        # Optimize the model
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def training_loop(self):
        num_episodes = 10

        for i_episode in range(num_episodes):
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            episode_reward = 0

            for t in count():
                action = self.turn(state, i_episode)
                observation, reward, terminated, truncated, _ = self.env.step(action.tolist())
                reward = torch.tensor([reward], device=device)
                episode_reward += reward.item()  # Accumulate reward
                done = terminated or truncated

                next_state = None if terminated else torch.tensor(
                    observation, dtype=torch.float32, device=device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)
                state = next_state

                # Optimize model
                self.optimize_model()

                # Soft update target network
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    print(f'Episode {i_episode}: {t} steps with reward {episode_reward}')
                    self.episode_durations.append(t + 1)
                    self.episode_rewards.append(episode_reward)  # Store episode reward
                    if is_plot:
                        self.plot_durations()
                    break
        
        with open('episode_rewards.json', 'w') as f:
            json.dump(self.episode_rewards, f)
        with open('episode_durations.json', 'w') as f:
            json.dump(self.episode_durations, f)
        torch.save(self.target_net, 'target_net_state_dict.pt')
                    

    def plot_durations(self, show_result=False):
        
        ax1.clear()
        ax2.clear()
        # Plot episode durations
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        ax1.set_title('Episode Durations')
        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Duration')
        ax1.plot(durations_t.numpy())
        
        # Plot means for durations
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            ax1.plot(means.numpy(), 'r')
            
        # Plot episode rewards
        rewards_t = torch.tensor(self.episode_rewards, dtype=torch.float)
        ax2.set_title('Episode Rewards')
        ax2.set_xlabel('Episode')
        ax2.set_ylabel('Total Reward')
        ax2.plot(rewards_t.numpy())
        
        # Plot means for rewards
        if len(rewards_t) >= 100:
            means = rewards_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            ax2.plot(means.numpy(), 'r')
        
        plt.tight_layout()
        plt.pause(0.001)
        
        """if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())"""

if __name__ == "__main__":
    gym.register(
    id='AssemblyLine-v0',
    entry_point='Game:Game',
    )
    is_plot = False
    if is_plot:
        is_ipython = 'inline' in matplotlib.get_backend()
        if is_ipython:
            from IPython import display
        plt.ion()
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 12))
    Agent().training_loop()
