from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer, PrioritizedReplay
from actors import Actor
from learners import Learner
from config import make_config
from utils import print_network_summary
from evaluate import Evaluator
import datetime
import torch
import pytz
import time
import ray
import os


def launch(config):
    ray.init()

    if config.load_state:
      state = torch.load(config.load_state)
      if not config.override_loaded_config:
        config = state['config']
    else:
      state = None

    date = datetime.datetime.now(tz=pytz.timezone('Europe/Stockholm')).strftime("%d-%b-%Y_%H:%M:%S")

    run_tag = config.run_tag if config.run_tag else date

    storage = SharedStorage.remote(config.num_actors)
    replay_buffer = PrioritizedReplay.remote(config)

    actors = [Actor.remote(actor_id, config, storage, replay_buffer, state, run_tag, date) for actor_id in range(config.num_actors)]
    learner = Learner.remote(config, storage, replay_buffer, state, run_tag, date)
    workers = actors + [learner]

    if config.print_network_summary:
        print_network_summary(config)

    print("\nStarting date: {}".format(date))
    print("Using environment: {}.".format(config.environment))
    print("Using architecture: {}.".format(config.architecture))
    print("Using replay memory with capacity: {}.".format(config.window_size))
    print("Using optimizer: {}.".format(config.optimizer))
    if config.lr_scheduler:
        print("Using lr scheduler: {},".format(config.lr_scheduler))
        if config.lr_scheduler == 'MuZeroLR':
            print("   lr decay steps: {},".format(config.lr_decay_steps))
            print("   lr decay rate: {},".format(config.lr_decay_rate))
        elif config.lr_scheduler == 'ExponentialLR':
            print("   lr decay rate: {},".format(config.lr_decay_rate))
    print("   lr initial: {}.".format(config.lr_init))
    print("Using {} as policy loss.".format(config.policy_loss))
    print("Using {} as scalar loss.".format(config.scalar_loss))
    print("Using discount {}.".format(config.discount))
    print("Using {} actors.".format(config.num_actors))

    print("Launching...\n")

    ray.get([worker.launch.remote() for worker in workers])

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    config = make_config()
    launch(config)
