from shared_storage import SharedStorage
from replay_buffer import PrioritizedReplay
from utils import get_environment
import numpy as np
from learners import Learner
from actors import Actor
from config import make_config
from copy import deepcopy
import datetime
import pickle
import torch
import pytz
import time
import ray
import os


def print_launch_message(config, date):
  print("\n\033[92mStarting date: {}\033[0m".format(date))
  print("Using environment: {}.".format(config.environment))
  print("Using architecture: {}.".format(config.architecture))
  print("Using replay memory with max capacity: {}.".format(config.window_size))
  if config.window_step is not None:
    print("   - with {} as step-size.".format(config.window_step))
  print("   - {} stored before learner starts.".format(config.stored_before_train))
  print("Using optimizer: {}.".format(config.optimizer))
  print("   - initial lr: {}.".format(config.lr_init))
  if config.optimizer == 'SGD':
    print("   - momentum: {}".format(config.momentum))
  if config.weight_decay:
    print("   - weight decay: {},".format(config.weight_decay))
  if config.lr_scheduler:
    print("Using lr scheduler: {},".format(config.lr_scheduler))
    if config.lr_scheduler == 'MuZeroLR':
      print("   - lr decay steps: {},".format(config.lr_decay_steps))
      print("   - lr decay rate: {},".format(config.lr_decay_rate))
    elif config.lr_scheduler == 'ExponentialLR':
      print("   - lr decay rate: {},".format(config.lr_decay_rate))
  if not config.no_target_transform:
    print("Using target transform.")
  print("Using {} as policy loss.".format(config.policy_loss))
  if not config.no_support:
    print("Using value support between {} and {}.".format(config.value_support_min,
                                                          config.value_support_max))
    print("Using reward support between {} and {}.".format(config.reward_support_min,
                                                           config.reward_support_max))
    print("Using {} as value and reward loss.".format(config.policy_loss))
  else:
    print("Using {} as value and reward loss.".format(config.scalar_loss))
  print("Using batch size {}.".format(config.batch_size))
  print("Using discount {}.".format(config.discount))
  print("Using {} simulations per step.".format(config.num_simulations))
  print("Using td-steps {}.".format(config.td_steps))
  print("Using {} actors.".format(config.num_actors))
  if config.fixed_temperatures:
    print("   - with fixed temperatures: {}".format(config.fixed_temperatures))
  else:
    print("   - with dynamic temperatures: {}".format(config.visit_softmax_temperatures))
    print("   - changing at training steps: {}".format(config.visit_softmax_steps))
  print("\033[92mLaunching...\033[0m\n")

def launch(config, date, state=None):
  os.environ["OMP_NUM_THREADS"] = "1"
  ray.init()

  env = get_environment(config)
  config.action_space = env.action_space.n
  config.obs_space = env.observation_space.shape

  storage = SharedStorage.remote(config)
  replay_buffer = PrioritizedReplay.remote(config)
  actors = [Actor.remote(actor_key, config, storage, replay_buffer, state) for actor_key in range(config.num_actors)]
  learner = Learner.remote(config, storage, replay_buffer, state)
  workers = [learner] + actors

  print_launch_message(config, date)
  ray.get([worker.launch.remote() for worker in workers])
  ray.shutdown()

def set_tags(meta_config, config):
  tz = pytz.timezone(config.time_zone)
  date = datetime.datetime.now(tz=tz).strftime("%d-%b-%Y_%H-%M-%S")
  if config.run_tag is None:
    run_tag = ''
    if meta_config.create_run_tag_from:
      for key in meta_config.create_run_tag_from:
        tag = "{}={}".format(key, getattr(config, key))
        run_tag = os.path.join(run_tag, tag)
    run_tag = os.path.join(run_tag, date)
    config.run_tag = run_tag
  return date

def config_generator(meta_config):
  for seed in meta_config.seed:
    for num_actors in meta_config.num_actors:
      for lr_init in meta_config.lr_init:
        for discount in meta_config.discount:
          for window_size in meta_config.window_size:
            for window_step in meta_config.window_step:
              for batch_size in meta_config.batch_size:
                for num_simulations in meta_config.num_simulations:
                  for num_unroll_steps in meta_config.num_unroll_steps:
                    for td_steps in meta_config.td_steps:

                      config = deepcopy(meta_config)

                      if seed is None:
                        config.seed = np.random.randint(10000)
                      else:
                        config.seed = seed

                      config.num_actors = num_actors
                      config.lr_init = lr_init
                      config.discount = discount
                      config.batch_size = batch_size
                      config.num_simulations = num_simulations
                      config.num_unroll_steps = num_unroll_steps
                      config.window_size = window_size
                      config.window_step = window_step
                      config.td_steps = td_steps

                      date = set_tags(meta_config, config)

                      yield config, date


if __name__ == '__main__':
  meta_config = make_config()

  if meta_config.load_state:
    tz = pytz.timezone(meta_config.time_zone)
    date = datetime.datetime.now(tz=tz).strftime("%d-%b-%Y_%H-%M-%S")
    state = torch.load(meta_config.load_state, map_location='cpu')
    launch(state['config'], date, state=state)
  else:
    for config, date in config_generator(meta_config):
      launch(config, date)

