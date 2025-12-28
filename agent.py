import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from network import SimpleNN
from replay_buffer import ReplayBuffer

class Agent:
    def __init__(
        self,
        env,
        gamma=0.99,
        lr=1e-3,
        epsilon=1.0,
        epsilon_decay=0.0001,
        final_epsilon=0.05,
        buffer_size=100_000,
        batch_size=32,
        target_update_freq=1000,
    ):
        self.env = env
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon

        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.n

        self.online_net = SimpleNN(obs_dim, act_dim)
        self.target_net = SimpleNN(obs_dim, act_dim)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn = nn.MSELoss()

        self.replay_buffer = ReplayBuffer(buffer_size)
        self.step_count = 0

    def get_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.env.action_space.sample()

        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            return self.online_net(state).argmax(dim=1).item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return

        states, actions, rewards, next_states, dones = \
            self.replay_buffer.sample(self.batch_size)

        # Current Q(s,a)
        q_values = self.online_net(states)
        q_sa = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

        # -------- Double DQN target --------
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(dim=1)
            next_q_values = self.target_net(next_states)
            max_next_q = next_q_values.gather(
                1, next_actions.unsqueeze(1)
            ).squeeze(1)

            target = rewards + (1 - dones) * self.gamma * max_next_q

        loss = self.loss_fn(q_sa, target)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()

        self.step_count += 1
        if self.step_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(
            self.final_epsilon,
            self.epsilon - self.epsilon_decay
        )

    def save(self, path="dqn_lunarlander.pth"):
        torch.save(self.online_net.state_dict(), path)

    def load(self, path="dqn_lunarlander.pth"):
        self.online_net.load_state_dict(torch.load(path))
        self.target_net.load_state_dict(self.online_net.state_dict())
