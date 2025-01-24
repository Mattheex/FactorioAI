import math
from collections import namedtuple, deque
import random
from time import sleep

import numpy as np
import pygame.time
from torch.optim import Adam

import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import gymnasium as gym
from Game import Game

from entities import Case
from var import WIDTH, HEIGHT, RIGHT, DOWN, TOP, LEFT, ITEM_SIZE

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# np.set_printoptions(threshold=np.inf)

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)
BATCH_SIZE = 128
GAMMA = 0.6
EPS_START = 0.95
EPS_END = 0.05
EPS_DECAY = 100
TAU = 0.005
n_actions = 4  # x, y, obj, direction


class DQN(nn.Module):
    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, stride=2)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2)
        self.dropout1 = nn.Dropout(0.25)
        self.fc1 = nn.Linear(288, n_actions)
        # self.fc2 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # print(x.shape)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        # x = F.relu(self.fc1(x))
        return self.fc1(x)


# Register the environment
gym.register(
    id='AssemblyLine',
    entry_point=Game,
)

"""env = gym.make('AssemblyLine')
obs = env.reset()
env.render()"""

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent:
    def __init__(self) -> None:
        # self.game = game
        self.lr = 1e-4

        self.env = gym.make('AssemblyLine')

        n_actions = self.env.action_space.n
        print('n_actions', n_actions)

        self.policy_net = DQN(n_actions).to(device)
        self.target_net = DQN(n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)

        """self.cov_var = torch.full(size=(n_actions,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)"""

        # self.memory = ReplayMemory(10000)
        self.step_done = 0
        # self.action = [["B", RIGHT, RIGHT], ["B", DOWN, DOWN], ["B", TOP, TOP], ["B", LEFT, LEFT]]

        self.memory = ReplayMemory(10000)
        self.episode_durations = []

    """def create_entities(self, x, y, type, dir):
        return self.game.add_items(0, 2, type, dir)"""

    def turn(self, state, steps_done):
        sample = random.random()

        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.

                return self.policy_net(state).max(1).indices
        else:
            if sample < 0.005:
                action = self.env.action_space.sample()
            else:
                action = 399

            return torch.tensor([action], device=device, dtype=torch.long)

    def training_loop(self):
        num_episodes = 5000

        for i_episode in range(num_episodes):

            state, info = self.env.reset()

            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = self.turn(state, i_episode)
                observation, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                self.memory.push(state, action, next_state, reward)

                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (
                            1 - TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    print('turn to reach 50', t, 'and steps done',i_episode)
                    self.episode_durations.append(t + 1)
                    self.plot_durations()
                    break
                self.env.render()

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return
        transitions = self.memory.sample(BATCH_SIZE)

        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # print(batch.action)

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net

        action_batch = torch.unsqueeze(action_batch, 1)

        #print(reward_batch)

        # print(self.policy_net(state_batch))
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        """ img = self.board_to_image(grid, items)

        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                # t.max(1) will return the largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor([[env.action_space.sample()]], device=device, dtype=torch.long)
        learning = uniform(0, 1)

        if learning > self.learningFactor:
            self.createEntities(choice(self.action))
        else:
            self.createEntities(np.max(self.states, 0))

        self.learningFactor += self.learningRate"""

    """def reward(self, g) -> int:
        r = 0
        for row, col in itertools.product(range(n), range(n)):
            if isinstance(g[row, col], Belt):

                neighbors, _ = get_neighbors(g, row, col)

                contains_instance = any(
                    isinstance(item, Mine) for items in neighbors for item in items
                )

                if contains_instance:
                    r += 10

        return r"""

    def board_to_image(self, grid, items):
        img = np.zeros((2, WIDTH, HEIGHT), dtype=np.integer)
        for obj in sum(grid, []):
            if type(obj) is not Case:
                coor = obj.coor.astype(int)
                img[0, coor[0]:coor[0] + obj.size, coor[1]:coor[1] + obj.size] = obj.class_id
        for item in items:
            coor = item.coor.astype(int)
            img[1, coor[0]:coor[0] + item.size, coor[1]:coor[1] + item.size] = item.class_id

        trunc = round(ITEM_SIZE / 2)
        img = img[:, ::trunc, ::trunc]

        print(img.shape)

        """for i in img[0]:
            print(" ".join(map(str, i)))"""

        return img


Agent().training_loop()
