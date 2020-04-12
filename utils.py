from atari_wrappers import wrap_atari
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import RMSprop, Adam, SGD
from torch.nn import MSELoss, LogSoftmax
from networks import MuZeroNetwork, FCNetwork
from torchsummary import summary
import numpy as np
import torch
import gym

def print_network_summary(network, input_shapes):
    input_shapes = []
    replace_characters = ['(', ')', ' ']
    for input_shape in config.input_shapes:
      for character in replace_characters:
        input_shape = input_shape.replace(character, '')
      input_shape = tuple(map(lambda x: int(x), input_shape.split(',')))
      input_shapes.append(input_shape)

    summary(network.representation, input_shapes[0])
    summary(network.prediction, input_shapes[1])
    summary(network.dynamics, input_shapes[2])

def get_environment(config):
    if 'atari' in config.environment_type: 
      if 'box' in config.environment_type:
        raise NotImplementedError
      env = gym.make(config.environment_id)
      if config.wrap_atari:
        env = wrap_atari(env, config)
    else:
      raise NotImplementedError
    return env

def get_network(config, device):
    env = get_environment(config)
    action_space = env.action_space.n

    if config.architecture == 'MuzeroNetwork':
      input_channels = config.stack_frames
      if config.stack_actions:
        input_channels *= 2
      network = MuzeroNetwork(input_channels, action_space, device)

    elif config.architecture == 'EquivNetwork':
      raise NotImplementedError

    elif config.architecture == 'FCNetwork':
      input_dim = np.shape(env.observation_space)[0]
      network = FCNetwork(input_dim, action_space, device)

    else:
      raise NotImplementedError
    return network

def get_loss_functions(config):
    def cross_entropy_loss(policy_logits, target_policy):
        loss = (-target_policy * LogSoftmax(dim=1)(policy_logits)).sum(1)
        return loss
    if config.scalar_loss == 'MSELoss':
        scalar_loss = MSELoss(reduction='none')
    else:
        raise NotImplementedError
    if config.policy_loss == 'CrossEntropyLoss':
        policy_loss = cross_entropy_loss
    else:
        raise NotImplementedError
    return scalar_loss, policy_loss

def get_optimizer(config, parameters):
    if config.optimizer == 'RMSprop':
        optimizer = RMSprop(parameters, lr=config.lr_init, momentum=config.momentum, eps=0.01, weight_decay=config.weight_decay)
    elif config.optimizer == 'Adam':
        optimizer = Adam(parameters, lr=config.lr_init, weight_decay=config.weight_decay, eps=0.00015)
    elif config.optimizer == 'SGD':
        optimizer = SGD(parameters, lr=config.lr_init, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def to_numpy(x):
  return x.detach().cpu().numpy()

def to_torch(x, device, dtype=torch.float32):
  x = np.array(x)
  return torch.tensor(x, dtype=dtype, device=device)# / 255.0

def get_error(root, initial_inference):
    return (root.value() - initial_inference.value.item())


class MuZeroLR():

  def __init__(self, optimizer, config):
    self.optimizer = optimizer
    self.lr_decay_steps = config.lr_decay_steps
    self.lr_decay_rate = config.lr_decay_rate
    self.lr_init = config.lr_init
    self.lr_step = 0

  def step(self):
    self.lr_step += 1
    lr = self.lr_init * self.lr_decay_rate ** (self.lr_step / self.lr_decay_steps)
    for param_group in self.optimizer.param_groups:
        param_group["lr"] = lr


def get_lr_scheduler(config, optimizer):
    if config.lr_scheduler:
        if config.lr_scheduler == 'ExponentialLR':
          lr_scheduler = ExponentialLR(optimizer, config.lr_decay_rate)
        elif config.lr_scheduler == 'MuZeroLR':
          lr_scheduler = MuZeroLR(optimizer, config)
        elif config.lr_scheduler == '':
          lr_scheduler = None
        else:
            raise NotImplementedError
    return lr_scheduler

def select_action(node, temperature=0):
    visit_counts = [child.visit_count for child in node.children]
    if temperature:
      distribution = np.array([visit_count**(1 / temperature) for visit_count in visit_counts])
      distribution = distribution / sum(distribution)
      action = np.random.choice(len(visit_counts), p=distribution)
    else:
      action = np.argmax(visit_counts)
    return action
