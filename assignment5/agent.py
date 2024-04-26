import random
import torch
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from memory import ReplayMemory, ReplayMemoryLSTM
from model import DQN, DQN_LSTM
from utils import find_max_lives, check_live, get_frame, get_init_state
from config import *
import os

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent():
    def __init__(self, action_size):
        self.action_size = action_size

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1
        self.epsilon_min = 0.01
        self.explore_step = 500000
        self.epsilon_decay = (
            self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 100000

        # Generate the memory
        self.memory = ReplayMemory()

        # Create the policy net
        self.policy_net = DQN(action_size)
        self.policy_net.to(device)

        self.target_net = DQN(action_size)
        self.target_net.to(device)
        self.update_targetnet()

        self.optimizer = optim.Adam(
            params=self.policy_net.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.StepLR(
            self.optimizer, step_size=scheduler_step_size, gamma=scheduler_gamma)

    def load_policy_net(self, path):
        self.policy_net = torch.load(path)

    def update_targetnet(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    """Get action using policy net using epsilon-greedy policy"""

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            ### CODE ####
            action = torch.randint(0, self.action_size, (1,))[0]
        else:
            ### CODE ####
            # state = torch.FloatTensor(state).unsqueeze(0).to(device)
            state = torch.from_numpy(state).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = self.policy_net(state)
                action = q_values.argmax()
        return action

    # pick samples randomly from replay memory (with batch_size)
    def train_policy_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch, dtype=object).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        states = torch.from_numpy(states).cuda()
        actions = list(mini_batch[1])
        actions = torch.LongTensor(actions).cuda()
        rewards = list(mini_batch[2])
        rewards = torch.FloatTensor(rewards).cuda()
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3]  # checks if the game is over
        mask = torch.tensor(list(map(int, dones == False)), dtype=torch.uint8)

        # Compute Q(s_t, a), the Q-value of the current state
        ### CODE ####
        qv = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Compute Q function of next state
        ### CODE ####
        with torch.no_grad():
            next_qv = self.target_net(torch.from_numpy(next_states).to(device)).detach()
        

        # Find maximum Q-value of action at next state from policy net
        ### CODE ####
        next_qv = next_qv.max(1)[0]
        eq = rewards + next_qv * self.discount_factor * mask.to(device)
        # Compute the Huber Loss
        ### CODE ####
        loss = F.smooth_l1_loss(qv, eq)

        # Optimize the model, .step() both the optimizer and the scheduler!
        ### CODE ####
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.scheduler.step()
