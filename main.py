from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer, PrioritizedReplay
from actors import Actor
from learners import Learner
from config import make_atari_config
from utils import print_network_summary
import datetime
import pytz
import time
import ray
import os

def launch(config):
    ray.init()
    storage = SharedStorage.remote(config.num_actors)
    replay_buffer = PrioritizedReplay.remote(config)

    date = datetime.datetime.now(tz=pytz.timezone('Europe/Stockholm')).strftime("%d-%b-%Y_%H:%M:%S")

    actors = [Actor.remote(date, i, config, storage, replay_buffer) for i in range(config.num_actors)]
    learner = Learner.remote(date, config, storage, replay_buffer)
    workers = actors + [learner]

    if config.print_network_summary:
        network = ray.get(learner.get_network.remote())
        print_network_summary(network, config)

    print("\nStarting date: {}".format(date))
    print("Using environment: {}.".format(config.environment_id))
    print("Using architecture: {}.".format(config.architecture))
    print("Using replay memory with capacity = {}.".format(config.window_size))
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
    print("Using {} actors.".format(config.num_actors))

    print("Launching...\n")
    time.sleep(1.)

    ray.get([worker.launch.remote() for worker in workers])

    return storage.latest_weights.remote()

if __name__ == '__main__':
    os.environ["OMP_NUM_THREADS"] = "1"
    config = make_atari_config()
    final_weights = launch(config)
