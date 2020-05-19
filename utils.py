from atari_wrappers import wrap_atari
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import RMSprop, Adam, SGD
from torch.nn import MSELoss, LogSoftmax, SmoothL1Loss
from networks import MuZeroNetwork, FCNetwork, TinyNetwork
from torchsummary import summary
import numpy as np
import torch
import gym

def print_network_summary(config):
    network = get_network(config)
    
    summary(network.representation_head, (4, 96, 96))
    summary(network.policy_head, (128, 6, 6))
    summary(network.reward_head, (128+1, 6, 6))
    summary(network.value_head, (128, 6, 6))
    summary(network.transition_head, (128+1, 6, 6))

def get_environment(config):
    env = gym.make(config.environment)
    if config.wrap_atari:
      env = wrap_atari(env, config)
    return env

def get_network(config, device=None):
    if device is None:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = get_environment(config)
    action_space = env.action_space.n

    if config.architecture == 'MuZeroNetwork':
      input_channels = config.stack_frames
      if config.stack_actions:
        input_channels *= 2
      network = MuZeroNetwork(input_channels, action_space, device, config)
    elif config.architecture == 'TinyNetwork':
      input_channels = config.stack_frames
      if config.stack_actions:
        input_channels *= 2
      network = TinyNetwork(input_channels, action_space, device, config)
    elif config.architecture == 'TinierNetwork':
      input_channels = config.stack_frames
      if config.stack_actions:
        input_channels *= 2
      network = TinierNetwork(input_channels, action_space, device, config)
    elif config.architecture == 'FCNetwork':
      input_dim = np.shape(env.observation_space)[0]
      network = FCNetwork(input_dim, action_space, device, config)
    else:
      raise NotImplementedError
    return network

def get_loss_functions(config):
    def cross_entropy_loss(policy_logits, target_policy):
      loss = (-target_policy * LogSoftmax(dim=1)(policy_logits)).sum(1)
      return loss
    if config.policy_loss == 'CrossEntropyLoss':
      policy_loss = cross_entropy_loss
    else:
      raise NotImplementedError
    if not config.no_support:
      scalar_loss = policy_loss
    else:
      if config.scalar_loss == 'MSE':
        scalar_loss = MSELoss(reduction='none')
      elif config.scalar_loss == 'Huber':
        scalar_loss = SmoothL1Loss(reduction='none')
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


class MuZeroLR():

  def __init__(self, optimizer, config):
    self.optimizer = optimizer
    self.lr_decay_steps = config.lr_decay_steps
    self.lr_decay_rate = config.lr_decay_rate
    self.lr_init = config.lr_init
    self.lr_step = 0

  def step(self):
    self.lr_step += 1
    if self.lr_step <= self.lr_decay_steps:
      lr = self.lr_init * self.lr_decay_rate ** (self.lr_step / self.lr_decay_steps)
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = lr


def get_lr_scheduler(config, optimizer):
    if config.lr_scheduler == 'ExponentialLR':
      lr_scheduler = ExponentialLR(optimizer, config.lr_decay_rate)
    elif config.lr_scheduler == 'MuZeroLR':
      lr_scheduler = MuZeroLR(optimizer, config)
    elif config.lr_scheduler == '':
      lr_scheduler = None
    else:
      raise NotImplementedError
    return lr_scheduler


