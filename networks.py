from collections import namedtuple
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch
import math


NetworkOutput = namedtuple('network_output', ('value', 'reward', 'policy_logits', 'hidden_state'))


class BaseNetwork(nn.Module):

  def __init__(self):
    super(BaseNetwork, self).__init__()

  def representation(self, observation):
    raise NotImplementedError

  def prediction(self, state):
    raise NotImplementedError

  def dynamics(self, hidden_state, action):
    raise NotImplementedError

  def initial_inference(self, observation):
    hidden_state = self.representation(observation)
    policy_logits, value = self.prediction(hidden_state)
    return NetworkOutput(value, 0, policy_logits, hidden_state)

  def recurrent_inference(self, hidden_state, action):
    hidden_state, reward = self.dynamics(hidden_state, action)
    policy_logits, value = self.prediction(hidden_state)
    return NetworkOutput(value, reward, policy_logits, hidden_state)

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


class FCRepresentation(nn.Module):

  def __init__(self, input_dim, hidden_dim):
    super(FCRepresentation, self).__init__()

    self.fc1 = nn.Linear(input_dim, 512)
    self.out = nn.Linear(512, hidden_dim)

  def forward(self, x):
    batch_size = x.shape[0]
    x = x.view(batch_size, -1)
    x = F.relu(self.fc1(x))
    return self.out(x)


class FCDynamicsState(nn.Module):

  def __init__(self, input_dim, action_space, hidden_dim):
    super(FCDynamicsState, self).__init__()

    self.fc1 = nn.Linear(hidden_dim+action_space, 512)
    self.out = nn.Linear(512, hidden_dim)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    return self.out(x)


class FCDynamicsReward(nn.Module):

  def __init__(self, input_dim, action_space, reward_support_size, hidden_dim):
    super(FCDynamicsReward, self).__init__()

    self.fc1 = nn.Linear(hidden_dim+action_space, 512)
    self.reward = nn.Linear(512, reward_support_size)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    return self.reward(x)


class FCPredictionValue(nn.Module):

  def __init__(self, input_dim, value_support_size, hidden_dim):
    super(FCPredictionValue, self).__init__()

    self.fc1 = nn.Linear(hidden_dim, 512)
    self.value = nn.Linear(512, value_support_size)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    return self.value(x)


class FCPredictionPolicy(nn.Module):

  def __init__(self, input_dim, action_space, hidden_dim):
    super(FCPredictionPolicy, self).__init__()
    
    self.fc1 = nn.Linear(hidden_dim, 512)
    self.policy = nn.Linear(512, action_space)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    return self.policy(x)


class FCNetwork(BaseNetwork):

  def __init__(self, input_dim, action_space, device, config):
    super(FCNetwork, self).__init__()
    self.device = device

    self.no_support = config.no_support
    self.action_space = action_space

    self.inverse_reward_transform = config.inverse_reward_transform
    self.inverse_value_transform = config.inverse_value_transform

    hidden_dim = 50
    value_out = config.value_support_size if not config.no_support else 1
    reward_out = config.reward_support_size if not config.no_support else 1
    self.representation_head = FCRepresentation(input_dim, hidden_dim)
    self.value_head = FCPredictionValue(input_dim, value_out, hidden_dim)
    self.policy_head = FCPredictionPolicy(input_dim, action_space, hidden_dim)
    self.reward_head = FCDynamicsReward(input_dim, action_space, reward_out, hidden_dim)
    self.transition_head = FCDynamicsState(input_dim, action_space, hidden_dim)
    self.to(device)

    self.LN = nn.LayerNorm([hidden_dim], elementwise_affine=True)

  def representation(self, observation):
    hidden_state = self.representation_head(observation)
    hidden_state = F.relu(self.LN(hidden_state))
    return hidden_state

  def prediction(self, hidden_state):
    value = self.value_head(hidden_state)
    if not self.training and not self.no_support:
        value = self.inverse_value_transform(value)
    policy = self.policy_head(hidden_state)
    return policy, value

  def dynamics(self, hidden_state, action):
    hidden_state_with_action = self.attach_action(hidden_state, action)
    reward = self.reward_head(hidden_state_with_action)
    if not self.training and not self.no_support:
        reward = self.inverse_reward_transform(reward)
    next_hidden_state = self.transition_head(hidden_state_with_action)
    next_hidden_state = F.relu(self.LN(next_hidden_state))
    return next_hidden_state, reward

  def attach_action(self, hidden_state, action):
    batch_size = np.shape(action)[0]
    action = np.array(action, dtype=np.int64)[:, np.newaxis]
    a = torch.from_numpy(action).to(self.device)
    one_hot = torch.zeros((batch_size, self.action_space), dtype=torch.float32, device=self.device)
    one_hot.scatter_(1, a, 1.0)
    hidden_state = torch.cat((hidden_state, one_hot), dim=1)
    return hidden_state

  def load_weights(self, weights):
    self.load_state_dict(weights)

  def get_weights(self):
    return {key: value.cpu() for key, value in self.state_dict().items()}


class AttentionRepresentation(nn.Module):

  def __init__(self, input_dim, out_dim, config):
    super(AttentionRepresentation, self).__init__()
    self.num_heads = config.num_heads
    self.use_binary_pe = config.use_binary_pe

    if len(input_dim) == 1:
      raise ValueError("--stack_obs must be larger than 1 with AttentionNetwork")
    else:
      self.time_steps = input_dim[0]
      input_dim = input_dim[1]
    
    self.attention_dim = config.attention_dim if config.attention_dim is not None else input_dim
    self.value_dim = config.value_dim if config.value_dim is not None else out_dim

    self.scale = math.sqrt(self.attention_dim)

    self.softmax = nn.Softmax(dim=-1)

    if self.use_binary_pe:
      pe = torch.eye(self.time_steps)
      self.register_buffer('pe', pe, persistent=False)
      input_dim += self.time_steps

    self.keys = nn.Linear(input_dim, self.attention_dim*self.num_heads, bias=False)
    self.queries = nn.Linear(input_dim, self.attention_dim*self.num_heads, bias=False)
    self.values = nn.Linear(input_dim, self.value_dim*self.num_heads, bias=False)

    self.out = nn.Linear(self.num_heads*self.time_steps*self.value_dim, out_dim)

  def forward(self, x):
    bs = x.shape[0]

    if self.use_binary_pe and self.num_heads > 1:
      x = torch.cat((x, self.pe.expand(bs, -1, -1)), dim=-1)

    keys = self.keys(x).view(bs, self.num_heads, -1, self.attention_dim)
    queries = self.queries(x).view(bs, self.num_heads, -1, self.attention_dim)
    values = self.values(x).view(bs, self.num_heads, -1, self.value_dim)

    attention = torch.matmul(queries, keys.transpose(-2, -1))

    new_x = torch.matmul(self.probs, values).view(bs, -1)

    out = self.out(new_x)
    return out 


class AttentionNetwork(BaseNetwork):

  def __init__(self, input_dim, action_space, device, config):
    super(AttentionNetwork, self).__init__()
    self.device = device

    self.no_support = config.no_support
    self.action_space = action_space

    self.inverse_reward_transform = config.inverse_reward_transform
    self.inverse_value_transform = config.inverse_value_transform

    hidden_dim = 50
    value_out = config.value_support_size if not config.no_support else 1
    reward_out = config.reward_support_size if not config.no_support else 1
    self.representation_head = AttentionRepresentation(input_dim, hidden_dim, config)
    self.value_head = FCPredictionValue(input_dim, value_out, hidden_dim)
    self.policy_head = FCPredictionPolicy(input_dim, action_space, hidden_dim)
    self.reward_head = FCDynamicsReward(input_dim, action_space, reward_out, hidden_dim)
    self.transition_head = FCDynamicsState(input_dim, action_space, hidden_dim)
    self.to(device)

    self.LN = nn.LayerNorm([hidden_dim], elementwise_affine=True)

  def representation(self, observation):
    hidden_state = self.representation_head(observation)
    hidden_state = F.relu(self.LN(hidden_state))
    return hidden_state

  def prediction(self, hidden_state):
    value = self.value_head(hidden_state)
    if not self.training and not self.no_support:
        value = self.inverse_value_transform(value)
    policy = self.policy_head(hidden_state)
    return policy, value

  def dynamics(self, hidden_state, action):
    hidden_state_with_action = self.attach_action(hidden_state, action)
    reward = self.reward_head(hidden_state_with_action)
    if not self.training and not self.no_support:
        reward = self.inverse_reward_transform(reward)
    next_hidden_state = self.transition_head(hidden_state_with_action)
    next_hidden_state = F.relu(self.LN(next_hidden_state))
    return next_hidden_state, reward

  def attach_action(self, hidden_state, action):
    batch_size = np.shape(action)[0]
    action = np.array(action, dtype=np.int64)[:, np.newaxis]
    a = torch.from_numpy(action).to(self.device)
    one_hot = torch.zeros((batch_size, self.action_space), dtype=torch.float32, device=self.device)
    one_hot.scatter_(1, a, 1.0)
    hidden_state = torch.cat((hidden_state, one_hot), dim=1)
    return hidden_state

  def load_weights(self, weights):
    self.load_state_dict(weights)

  def get_weights(self):
    return {key: value.cpu() for key, value in self.state_dict().items()}


class HopfieldRepresentation(nn.Module):

  def __init__(self, input_dim, hidden_dim, config):
    super(HopfieldRepresentation, self).__init__()
    self.num_heads = config.num_heads
    self.num_keys = config.num_keys
    self.scale = math.sqrt(input_dim)

    self.full_shape = (self.num_keys*self.num_heads, input_dim)
    self.value_shape = (self.full_shape[0], hidden_dim)

    self.keys = nn.Parameter(torch.empty(*self.full_shape))
    bound = 1 / math.sqrt(input_dim)
    torch.nn.init.uniform_(self.keys, a=-bound, b=bound)

    self.values = nn.Parameter(torch.empty(*self.value_shape))
    bound = 1 / math.sqrt(hidden_dim)
    torch.nn.init.uniform_(self.values, a=-bound, b=bound)

    self.softmax = nn.Softmax(dim=1)
    
  def _get_probs(self, x, beta=8.):
    batch_size = x.shape[0]
    weights = torch.matmul(x, self.keys.T) / self.scale
    weights = weights.view(batch_size, self.num_keys, self.num_heads)
    probs = self.softmax(beta*weights)
    probs = probs.view(batch_size, self.full_shape[0])
    return probs

  def forward(self, x, beta=8., n_updates=1):
    for n in range(n_updates):
      probs = self._get_probs(x, beta)
      if n_updates > 1:
        x = torch.matmul(probs, self.keys)
    out = torch.matmul(probs, self.values)
    return out 


class HopfieldNetwork(BaseNetwork):

  def __init__(self, input_dim, action_space, device, config):
    super(HopfieldNetwork, self).__init__()
    self.device = device

    self.no_support = config.no_support
    self.action_space = action_space

    self.inverse_reward_transform = config.inverse_reward_transform
    self.inverse_value_transform = config.inverse_value_transform

    value_out = config.value_support_size if not config.no_support else 1
    reward_out = config.reward_support_size if not config.no_support else 1

    hidden_dim = 50 
    self.representation_head = HopfieldRepresentation(input_dim, hidden_dim, config)
    self.value_head = FCPredictionValue(value_out, hidden_dim)
    self.policy_head = FCPredictionPolicy(action_space, hidden_dim)
    self.reward_head = FCDynamicsReward(action_space, reward_out, hidden_dim)
    self.transition_head = FCDynamicsState(action_space, hidden_dim)
    self.to(device)

    self.layer_norm = nn.LayerNorm([hidden_dim], elementwise_affine=True)

  def representation(self, observation):
    hidden_state = self.representation_head(observation)
    hidden_state = F.relu(self.layer_norm(hidden_state))
    return hidden_state

  def prediction(self, hidden_state):
    value = self.value_head(hidden_state)
    if not self.training and not self.no_support:
      value = self.inverse_value_transform(value)
    policy = self.policy_head(hidden_state)
    return policy, value

  def dynamics(self, hidden_state, action):
    hidden_state_with_action = self.attach_action(hidden_state, action)
    reward = self.reward_head(hidden_state_with_action)
    if not self.training and not self.no_support:
      reward = self.inverse_reward_transform(reward)
    next_hidden_state = self.transition_head(hidden_state_with_action)
    next_hidden_state = F.relu(self.layer_norm(next_hidden_state))
    return next_hidden_state, reward

  def attach_action(self, hidden_state, action):
    batch_size = np.shape(action)[0]
    action = np.array(action, dtype=np.int64)[:, np.newaxis]
    a = torch.from_numpy(action).to(self.device)
    one_hot = torch.zeros((batch_size, self.action_space), dtype=torch.float32, device=self.device)
    one_hot.scatter_(1, a, 1.0)
    hidden_state = torch.cat((hidden_state, one_hot), dim=1)
    return hidden_state

  def load_weights(self, weights):
    self.load_state_dict(weights)

  def get_weights(self):
    return {key: value.cpu() for key, value in self.state_dict().items()}


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

    for block in self.resblocks:
        out = block(out)
    return out


class MuZeroDynamics(nn.Module):

  def __init__(self, reward_support_size):
    super(MuZeroDynamics, self).__init__()
    self.conv = nn.Conv2d(128 + 1, 128, kernel_size=3, stride=1, padding=1)
    self.bn = nn.BatchNorm2d(128)
    self.resblocks = nn.ModuleList([ResidualBlock(128) for _ in range(16)])

    self.fc1 = nn.Linear(6 * 6 * 128, 512)
    self.fc2 = nn.Linear(512, reward_support_size)

  def forward(self, x):
    batch_size = x.size(0)
    out = self.conv(x)
    out = self.bn(out)
    out = F.relu(out)
    for block in self.resblocks:
        out = block(out)
    state = out
    reward = F.relu(self.fc1(out.view(batch_size, -1)))
    reward = self.fc2(reward.view(batch_size, -1))
    return state, reward


class MuZeroPrediction(nn.Module):

  def __init__(self, action_space, value_support_size):
    super(MuZeroPrediction, self).__init__()
    self.resblocks = nn.ModuleList([ResidualBlock(128) for _ in range(16)])

    self.fc_value = nn.Linear(6 * 6 * 128, 512)
    self.fc_value_o = nn.Linear(512, value_support_size)

    self.fc_policy = nn.Linear(6 * 6 * 128, 512)
    self.fc_policy_o = nn.Linear(512, action_space)

  def forward(self, x):
    batch_size = x.size(0)
    out = x
    for block in self.resblocks:
        out = block(out)
    out = out.view(batch_size, -1)

    value = F.relu(self.fc_value(out))
    value = self.fc_value_o(value)

    policy = F.relu(self.fc_policy(out))
    policy = self.fc_policy_o(policy)
    return policy, value


class MuZeroNetwork(BaseNetwork):

  def __init__(self, input_channels, action_space, device, config):
    super(MuZeroNetwork, self).__init__()
    self.device = device
    
    self.no_support = config.no_support
    self.action_space = action_space
    value_output_dim = config.value_support_size if not config.no_support else 1
    reward_output_dim = config.reward_support_size if not config.no_support else 1

    self.representation_head = MuZeroRepresentation(input_channels)
    self.prediction_head = MuZeroPrediction(action_space, value_output_dim)
    self.dynamics_head = MuZeroDynamics(reward_output_dim)
    self.to(device)

    self.inverse_reward_transform = config.inverse_reward_transform
    self.inverse_value_transform = config.inverse_value_transform

  def representation(self, observation):
    hidden_state = self.representation_head(observation)
    hidden_state = self.scale_state(hidden_state)
    return hidden_state

  def prediction(self, hidden_state):
    policy, value = self.prediction_head(hidden_state)
    if not self.training and not self.no_support:
      value = self.inverse_value_transform(value)
    return policy, value

  def dynamics(self, hidden_state, action):
    hidden_state_with_action = self.attach_action(hidden_state, action)
    next_hidden_state, reward = self.dynamics_head(hidden_state_with_action)
    if not self.training and not self.no_support:
      reward = self.inverse_reward_transform(reward)
    next_hidden_state = self.scale_state(next_hidden_state)
    return next_hidden_state, reward

  def attach_action(self, hidden_state, action):
    batch_size, _, x, y = hidden_state.shape
    action = torch.tensor(action, device=self.device).unsqueeze(-1)
    action_plane = torch.ones((batch_size, 1, x, y), dtype=torch.float, device=self.device)
    action_plane = action[:, :, None, None] * action_plane / self.action_space
    hidden_state = torch.cat((hidden_state, action_plane), dim=1)
    return hidden_state

  def scale_state(self, state):
    _min = state.min(dim=1, keepdim=True)[0]
    _max = state.max(dim=1, keepdim=True)[0]
    state = (state - _min) / (_max - _min)
    return state

  def load_weights(self, weights):
    self.load_state_dict(weights)

  def get_weights(self):
    return {key: value.cpu() for key, value in self.state_dict().items()}


class TinyBlock(nn.Module):

  def __init__(self, num_channels, stride=1):
    super(TinyBlock, self).__init__()
    self.conv1 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=1, bias=False)
    self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, stride=stride, padding=1, bias=False)

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = F.relu(self.conv2(out) + x)
    return out


class TinyRepresentation(nn.Module):

  def __init__(self, input_channels):
    super(TinyRepresentation, self).__init__()
    self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, stride=2, padding=1) # dims 48, 48

    self.max_pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # dims 24, 24

    self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # dims 12, 12

    self.max_pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # dims 6, 6

    self.block2 = TinyBlock(64) # dims 6, 6

    self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1) # dims 6, 6

  def forward(self, x):
    out = F.relu(self.conv1(x))
    out = self.max_pool1(out)
    out = F.relu(self.conv2(out))
    out = self.max_pool2(out)

    out = self.block2(out)
    out = torch.tanh(self.conv3(out))
    return out


class TinyValue(nn.Module):

  def __init__(self, value_support_size):
    super(TinyValue, self).__init__()
    self.block1 = TinyBlock(64)
    self.fc_value = nn.Linear(6 * 6 * 64, 512)
    self.fc_value_o = nn.Linear(512, value_support_size)

  def forward(self, x):
    batch_size = x.size(0)
    out = self.block1(x)
    value = F.relu(self.fc_value(out.view(batch_size, -1)))
    return self.fc_value_o(value)


class TinyPolicy(nn.Module):

  def __init__(self, action_space):
    super(TinyPolicy, self).__init__()
    self.block1 = TinyBlock(64)
    self.fc_policy = nn.Linear(6 * 6 * 64, 512)
    self.fc_policy_o = nn.Linear(512, action_space)

  def forward(self, x):
    batch_size = x.size(0)
    out = self.block1(x)
    out = F.relu(self.fc_policy(out.view(batch_size, -1)))
    policy = self.fc_policy_o(out)
    return policy


class TinyReward(nn.Module):

  def __init__(self, reward_support_size):
    super(TinyReward, self).__init__()
    self.block1 = TinyBlock(64 + 1)
    self.fc1 = nn.Linear(6 * 6 * (64 + 1), 512)
    self.fc2 = nn.Linear(512, reward_support_size)

  def forward(self, x):
    batch_size = x.size(0)
    out = self.block1(x)
    out = F.relu(self.fc1(out.view(batch_size, -1)))
    reward = self.fc2(out)
    return reward


class TinyHidden(nn.Module):

  def __init__(self):
    super(TinyHidden, self).__init__()
    self.block1 = TinyBlock(65)
    self.conv1 = nn.Conv2d(65, 64, kernel_size=3, stride=1, padding=1)

  def forward(self, x):
    batch_size = x.size(0)
    out = self.block1(x)
    return torch.tanh(self.conv1(out))


class TinyNetwork(BaseNetwork):

  def __init__(self, input_channels, action_space, device, config):
    super(TinyNetwork, self).__init__()
    self.device = device

    self.no_support = config.no_support
    self.action_space = action_space
    value_output_dim = config.value_support_size if not config.no_support else 1
    reward_output_dim = config.reward_support_size if not config.no_support else 1

    self.representation_head = TinyRepresentation(input_channels)
    self.value_head = TinyValue(value_output_dim)
    self.reward_head = TinyReward(reward_output_dim)
    self.policy_head = TinyPolicy(action_space)
    self.transition_head = TinyHidden()
    self.to(device)

    self.inverse_reward_transform = config.inverse_reward_transform
    self.inverse_value_transform = config.inverse_value_transform


  def representation(self, observation):
    hidden_state = self.representation_head(observation)
    hidden_state = self.scale_state(hidden_state)
    return hidden_state

  def prediction(self, hidden_state):
    value = self.value_head(hidden_state)
    if not self.training and not self.no_support:
      value = self.inverse_value_transform(value)
    policy = self.policy_head(hidden_state)
    return policy, value

  def dynamics(self, hidden_state, action):
    hidden_state_with_action = self.attach_action(hidden_state, action)
    reward = self.reward_head(hidden_state_with_action)
    if not self.training and not self.no_support:
      reward = self.inverse_reward_transform(reward)
    next_hidden_state = self.transition_head(hidden_state_with_action)
    next_hidden_state = self.scale_state(next_hidden_state)
    return next_hidden_state, reward

  def attach_action(self, hidden_state, action):
    batch_size, _, x, y = hidden_state.shape
    action = torch.tensor(action, device=self.device).unsqueeze(-1)
    action_plane = torch.ones((batch_size, 1, x, y), dtype=torch.float, device=self.device)
    action_plane = action[:, :, None, None] * action_plane / self.action_space
    hidden_state = torch.cat((hidden_state, action_plane), dim=1)
    return hidden_state

  def scale_state(self, state):
    _min = state.min(dim=1, keepdim=True)[0]
    _max = state.max(dim=1, keepdim=True)[0]
    state = (state - _min) / (_max - _min)
    return state

  def load_weights(self, weights):
    self.load_state_dict(weights)

  def get_weights(self):
    return {key: value.cpu() for key, value in self.state_dict().items()}

