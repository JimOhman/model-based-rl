from collections import namedtuple
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch


NetworkOutput = namedtuple('network_output', ('value', 'reward', 'policy_logits', 'hidden_state'))


class MinMaxStats(object):

  def __init__(self):
    self.maximum = -float("inf")
    self.minimum = float("inf")

  def update(self, state):
    self.maximum = max(self.maximum, torch.max(state).item())
    self.minimum = min(self.minimum, torch.min(state).item())

  def normalize(self, state):
    if self.maximum > self.minimum:
      return (state - self.minimum) / (self.maximum - self.minimum)
    return state

class FCRepresentation(nn.Module):

    def __init__(self, input_dim):
        super(FCRepresentation, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, input_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

class FCDynamics(nn.Module):

    def __init__(self, input_dim):
        super(FCDynamics, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.out = nn.Linear(32, input_dim)
        self.reward = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x), self.reward(x)


class FCPrediction(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(FCPrediction, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.policy = nn.Linear(32, output_dim)
        self.value = nn.Linear(32, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.policy(x), self.value(x)


class FCNetwork(nn.Module):

    def __init__(self, input_dim, action_space, device):
        super(FCNetwork, self).__init__()
        self.device = device
        self.action_space = action_space

        self.representation = FCRepresentation(input_dim)
        self.prediction = FCPrediction(input_dim, action_space)
        self.dynamics = FCDynamics(input_dim)
        self.min_max_stats = MinMaxStats()
        self.to(device)

    def initial_inference(self, image):
        hidden_state = self.representation(image)
        # hidden_state = self.scale_state(hidden_state)
        policy_logits, value = self.prediction(hidden_state)
        reward = torch.zeros_like(value)
        return NetworkOutput(value, reward, policy_logits, hidden_state)

    def recurrent_inference(self, hidden_state, action):
        # hidden_state = self.attach_action(hidden_state, action)
        hidden_state, reward = self.dynamics(hidden_state)
        # hidden_state = self.scale_state(hidden_state)
        policy_logits, value = self.prediction(hidden_state)
        return NetworkOutput(value, reward, policy_logits, hidden_state)

    def attach_action(self, hidden_state, action):
        batch_size, = np.shape(action)
        action_plane = torch.ones((batch_size, 1), device=self.device)
        for i, a in enumerate(action):
            normed_action = (a / self.action_space)
            action_plane[i, ...] *= normed_action
        hidden_state = torch.cat((hidden_state, action_plane), dim=1)
        return hidden_state

    def scale_state(self, state):
        self.min_max_stats.update(state)
        state = self.min_max_stats.normalize(state)
        return state

    def load_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return {key: value.cpu() for key, value in self.state_dict().items()}

    def get_gradients(self):
        grads = []
        for p in self.parameters():
            grad = None if p.grad is None else p.grad.data.cpu().numpy()
            grads.append(grad)
        return grads

    def set_gradients(self, gradients):
        for g, p in zip(gradients, self.parameters()):
            if g is not None:
                p.grad = torch.from_numpy(g)


class ResidualBlock(nn.Module):

    def __init__(self, num_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = F.relu(out)
        return out


class MuZeroRepresentation(nn.Module):

    def __init__(self, input_channels):
        super(MuZeroRepresentation, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1) # dims 48, 48

        self.resblocks1 = nn.ModuleList([ResidualBlock(64) for _ in range(2)])
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # dims 24, 24

        self.resblocks2 = nn.ModuleList([ResidualBlock(128) for _ in range(3)])
        self.avg_pool1 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1) # dims 12, 12

        self.resblocks3 = nn.ModuleList([ResidualBlock(128) for _ in range(3)])
        self.avg_pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1) # dims 6, 6

        self.conv = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.resblocks = nn.ModuleList([ResidualBlock(128) for _ in range(16)])

    def forward(self, x):
        out = self.conv1(x)
        for block in self.resblocks1:
            out = block(out)
        out = self.conv2(out)
        for block in self.resblocks2:
            out = block(out)
        out = self.avg_pool1(out)
        for block in self.resblocks3:
            out = block(out)
        out = self.avg_pool2(out)

        out = self.conv(out)
        out = self.bn(out)
        out = F.relu(out)
        for block in self.resblocks:
            out = block(out)
        return out


class MuZeroDynamics(nn.Module):

    def __init__(self):
        super(MuZeroDynamics, self).__init__()
        self.conv = nn.Conv2d(128 + 1, 128, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(128)
        self.resblocks = nn.ModuleList([ResidualBlock(128) for _ in range(16)])

        self.fc1 = nn.Linear(6 * 6 * 128, 512)
        self.fc2 = nn.Linear(512, 1)

    def forward(self, x):
        batch_size = x.size(0)
        out = self.conv(x)
        # out = self.bn(out)
        out = F.relu(out)
        for block in self.resblocks:
            out = block(out)
        state = out
        reward = F.relu(self.fc1(out.view(batch_size, -1)))
        reward = self.fc2(reward)
        return state, reward


class MuZeroPrediction(nn.Module):

    def __init__(self, output_dim):
        super(MuZeroPrediction, self).__init__()
        self.resblocks = nn.ModuleList([ResidualBlock(128) for _ in range(16)])

        self.conv1 = nn.Conv2d(128, 64, 1)

        self.fc_value = nn.Linear(6 * 6 * 64, 512)
        self.fc_value_o = nn.Linear(512, 1)

        self.fc_policy = nn.Linear(6 * 6 * 64, 512)
        self.fc_policy_o = nn.Linear(512, output_dim)

    def forward(self, x):
        batch_size = x.size(0)
        out = x
        for block in self.resblocks:
            out = block(out)
        out = self.conv1(out)
        out = out.view(batch_size, -1)

        value = F.relu(self.fc_value(out))
        value = self.fc_value_o(value)

        policy = F.relu(self.fc_policy(out))
        policy = self.fc_policy_o(policy)

        return policy, value


class MuZeroNetwork(nn.Module):

    def __init__(self, input_channels, action_space, device):
        super(MuZeroNetwork, self).__init__()
        self.device = device
        self.action_space = action_space

        self.representation = MuZeroRepresentation(input_channels)
        self.prediction = MuZeroPrediction(action_space)
        self.dynamics = MuZeroDynamics()
        self.min_max_stats = MinMaxStats()
        self.to(device)

    def initial_inference(self, image):
        hidden_state = self.representation(image)
        # hidden_state = self.scale_state(hidden_state)
        policy_logits, value = self.prediction(hidden_state)
        reward = torch.zeros_like(value)
        return NetworkOutput(value, reward, policy_logits, hidden_state)

    def recurrent_inference(self, hidden_state, action):
        hidden_state = self.attach_action(hidden_state, action)
        hidden_state, reward = self.dynamics(state)
        # hidden_state = self.scale_state(hidden_state)
        policy_logits, value = self.prediction(hidden_state)
        return NetworkOutput(value, reward, policy_logits, hidden_state)

    def attach_action(self, hidden_state, action):
        batch_size, = np.shape(action)
        action_plane = torch.ones((batch_size, 1, 6, 6), device=self.device)
        for i, a in enumerate(action):
            normed_action = (a / self.action_space)
            action_plane[i, ...] *= normed_action
        hidden_state = torch.cat((hidden_state, action_plane), dim=1)
        return hidden_state

    def scale_state(self, state):
        self.min_max_stats.update(state)
        state = self.min_max_stats.normalize(state)
        return state

    def load_weights(self, weights):
        self.load_state_dict(weights)

    def get_weights(self):
        return {key: value.cpu() for key, value in self.state_dict().items()}
