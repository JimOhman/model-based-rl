from wrappers import wrap_game
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import SGD, RMSprop, Adam, AdamW
from torch.nn import MSELoss, LogSoftmax, SmoothL1Loss
from networks import MuZeroNetwork, FCNetwork, TinyNetwork, HopfieldNetwork, AttentionNetwork
import numpy as np
import random
import torch
import gym


def get_environment(config):
    if config.environment == 'TicTacToe':
      from custom_environments.tic_tac_toe import TicTacToe
      environment = TicTacToe()
    else:
      environment = gym.make(config.environment)
      environment = wrap_game(environment, config)
    return environment

def get_network(config, device=None):
    if device is None:
      device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    env = get_environment(config)
    action_space = env.action_space.n

    if config.architecture == 'MuZeroNetwork':
      input_channels = config.stack_obs
      if config.stack_actions:
        input_channels *= 2
      network = MuZeroNetwork(input_channels, action_space, device, config)
    elif config.architecture == 'TinyNetwork':
      input_channels = config.stack_obs
      if config.stack_actions:
        input_channels *= 2
      network = TinyNetwork(input_channels, action_space, device, config)
    elif config.architecture == 'FCNetwork':
      input_dim = np.prod(env.observation_space.shape)
      network = FCNetwork(input_dim, action_space, device, config)
    elif config.architecture == 'HopfieldNetwork':
      input_dim = np.prod(env.observation_space.shape)
      network = HopfieldNetwork(input_dim, action_space, device, config)
    elif config.architecture == 'AttentionNetwork':
      input_dim = env.observation_space.shape
      if len(input_dim) == 1: 
        input_dim = input_dim[0]
      network = AttentionNetwork(input_dim, action_space, device, config)
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
    elif config.optimizer == 'AdamW':
      optimizer = AdamW(parameters, lr=config.lr_init, weight_decay=config.weight_decay, eps=0.00015)
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
    self.lr = self.lr_init

  def step(self):
    self.lr_step += 1
    self.lr = self.lr_init * self.lr_decay_rate ** (self.lr_step / self.lr_decay_steps)
    for param_group in self.optimizer.param_groups:
      param_group["lr"] = self.lr


class WarmUpLR():

  def __init__(self, optimizer, config):
    self.optimizer = optimizer
    self.max_lr = config.lr_init
    self.warm_up_steps = 5000
    self.lr_step = 0

    self.lr = (1 / self.warm_up_steps) * self.max_lr
    for param_group in self.optimizer.param_groups:
      param_group["lr"] = self.lr

  def step(self):
    self.lr_step += 1
    if self.lr_step <= self.warm_up_steps:
      self.lr = (self.lr_step / self.warm_up_steps) * self.max_lr
      for param_group in self.optimizer.param_groups:
        param_group["lr"] = self.lr


def get_lr_scheduler(config, optimizer):
  if config.lr_scheduler is None:
    return None
  if config.lr_scheduler == 'ExponentialLR':
    lr_scheduler = ExponentialLR(optimizer, config.lr_decay_rate)
  elif config.lr_scheduler == 'MuZeroLR':
    lr_scheduler = MuZeroLR(optimizer, config)
  elif config.lr_scheduler == 'WarmUpLR':
    lr_scheduler = WarmUpLR(optimizer, config)
  else:
    raise NotImplementedError
  return lr_scheduler

def set_all_seeds(seed=None):
  if seed is None:
    seed = random.randint(0, 1000)
  torch.manual_seed(seed)
  torch.cuda.manual_seed_all(seed+1)
  random.seed(seed+2)
  np.random.seed(seed+3)
  torch.backends.cudnn.deterministic = True
  torch.backends.cudnn.benchmark = False
