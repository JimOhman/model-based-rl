from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer, PrioritizedReplay
from actors import Actor
from learners import Learner
from config import make_config
from utils import print_network_summary
from copy import deepcopy
import numpy as np
import datetime
import random
import torch
import pytz
import time
import ray
import os

def launch(config, run_tag, group_tag, date):
  ray.init()

  if config.load_state:
    state = torch.load(config.load_state)
    if not config.override_loaded_config:
      config = state['config']
  else:
    state = None

  storage = SharedStorage.remote(config.num_actors)
  replay_buffer = PrioritizedReplay.remote(config)

  actors = [Actor.remote(actor_id, config, storage, replay_buffer, state, run_tag, group_tag, date) for actor_id in range(config.num_actors)]
  learner = Learner.remote(config, storage, replay_buffer, state, run_tag, group_tag, date)
  workers = actors + [learner]

  if config.print_network_summary:
    print_network_summary(config)

  print("\n\033[92mStarting date: {}\033[0m".format(date))
  print("Using environment: {}.".format(config.environment))
  print("Using architecture: {}.".format(config.architecture))
  print("Using replay memory with capacity: {}.".format(config.window_size))
  print("   with {} stored before learner starts.".format(config.stored_before_train))
  print("Using optimizer: {}.".format(config.optimizer))
  print("   initial lr: {}.".format(config.lr_init))
  if config.weight_decay:
    print("   weight decay: {},".format(config.weight_decay))
  if config.lr_scheduler:
    print("Using lr scheduler: {},".format(config.lr_scheduler))
    if config.lr_scheduler == 'MuZeroLR':
      print("   lr decay steps: {},".format(config.lr_decay_steps))
      print("   lr decay rate: {},".format(config.lr_decay_rate))
    elif config.lr_scheduler == 'ExponentialLR':
      print("   lr decay rate: {},".format(config.lr_decay_rate))
  if not config.no_target_transform:
    print("Using target transform.")
  print("Using {} as policy loss.".format(config.policy_loss))
  if not config.no_support:
    print("Using value support between {} and {}.".format(config.value_support_min, config.value_support_max))
    print("Using reward support between {} and {}.".format(config.reward_support_min, config.reward_support_max))
    print("Using {} as value and reward loss.".format(config.policy_loss))
  else:
    print("Using {} as value and reward loss.".format(config.scalar_loss))
  print("Using discount {}.".format(config.discount))
  print("Using {} simulations per step.".format(config.num_simulations))
  print("Using td-steps {}.".format(config.td_steps))
  print("Using {} actors.".format(config.num_actors))
  if config.sampling_ratio:
    print("Using sampling ratio: {}.".format(config.sampling_ratio))

  print("\033[92mLaunching...\033[0m\n")

  ray.get([worker.launch.remote() for worker in workers])
  ray.shutdown()

def get_run_tag(meta_config, config, date):
  if meta_config.run_tag is None:
    if meta_config.group_tag is not 'default':
      run_tag = ''
      run_tag = '{}'.format(config.architecture)
      run_tag += '/seed={}'.format(config.seed)
      run_tag += '/num_actors={}'.format(config.num_actors)
      run_tag += '/lr={}'.format(config.lr_init)
      run_tag += '/discount={}'.format(config.discount)
      run_tag += '/window_size={}'.format(config.window_size)
      run_tag += '/batch_size={}'.format(config.batch_size)
      run_tag += '/td_steps={}'.format(config.td_steps)
      run_tag += '/num_simulations={}'.format(config.num_simulations)
      run_tag += '/num_unroll_steps={}'.format(config.num_unroll_steps)
      run_tag += '/' + date
    else:
      run_tag = date
  else:
    run_tag = config.run_tag
  return run_tag

if __name__ == '__main__':
  os.environ["OMP_NUM_THREADS"] = "1"
  meta_config = make_config()

  for seed in meta_config.seed:
    for num_actors in meta_config.num_actors:
      for lr_init in meta_config.lr_init:
        for discount in meta_config.discount:
          for window_size in meta_config.window_size:
            for batch_size in meta_config.batch_size:
              for num_simulations in meta_config.num_simulations:
                for num_unroll_steps in meta_config.num_unroll_steps:
                  for td_steps in meta_config.td_steps:

                    config = deepcopy(meta_config)
                    config.seed = seed
                    config.num_actors = num_actors
                    config.lr_init = lr_init
                    config.discount = discount
                    config.batch_size = batch_size
                    config.num_simulations = num_simulations
                    config.num_unroll_steps = num_unroll_steps
                    config.window_size = window_size
                    config.td_steps = td_steps

                    date = datetime.datetime.now(tz=pytz.timezone('Europe/Stockholm')).strftime("%d-%b-%Y_%H:%M:%S")
                    run_tag = get_run_tag(meta_config, config, date)

                    launch(config, run_tag, config.group_tag, date)
