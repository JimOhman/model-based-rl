import ray
from shared_storage import SharedStorage
from replay_buffer import ReplayBuffer
from actors import Actor
from learners import Learner
from config import make_atari_config
import datetime
import pytz
import time

def launch(config):
    ray.init()
    storage = SharedStorage.remote()
    replay_buffer = ReplayBuffer.remote(config)

    workers = [Actor.remote(i, config, storage, replay_buffer) for i in range(config.num_actors)]
    workers += [Learner.remote(config, storage, replay_buffer)]

    date = datetime.datetime.now(tz=pytz.timezone('Europe/Stockholm')).strftime("%d-%b-%Y_%H:%M:%S")

    print("\nStarting date: {}".format(date))
    print("Using environment: {}.".format(config.environment))
    print("Using architecture: {}.".format(config.architecture))
    print("Using replay memory with capacity = {}.".format(config.window_size))
    print("Using optimizer {} with lr = {}.".format(config.optimizer, config.lr_init))
    print("Using {} as policy loss.".format(config.policy_loss))
    print("Using {} as scalar loss.".format(config.scalar_loss))
    print("Using {} workers.".format(config.num_actors))

    print("Launching...\n")
    time.sleep(1.)

    ray.get([worker.launch.remote() for worker in workers])

    return storage.latest_weights.remote()


if __name__ == '__main__':
    config = make_atari_config()
    final_network = launch(config)
