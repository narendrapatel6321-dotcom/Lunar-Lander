from collections import deque
import random
import numpy as np
import torch
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            torch.from_numpy(np.array(states)).float(),
            torch.LongTensor(actions),
            torch.FloatTensor(rewards),
            torch.from_numpy(np.array(next_states)).float(),
            torch.FloatTensor(dones),
        )

    def __len__(self):
        return len(self.buffer)

