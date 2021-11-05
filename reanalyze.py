import os
import time
from copy import deepcopy

import numpy as np
import ray
import torch

from logger import Logger
from utils import get_network, set_all_seeds, get_environment


@ray.remote
class Reanalyze(Logger):

  def __init__(self, config, storage, replay_buffer, state=None):
    set_all_seeds(config.seed)

    self.run_tag = config.run_tag
    self.group_tag = config.group_tag
    self.worker_id = 'reanalyze'
    self.replay_buffer = replay_buffer
    self.storage = storage
    self.config = deepcopy(config)
    self.histories_reanalyzed = 0
    self.training_step = 0

    self.environment = get_environment(config)
    self.environment.seed(config.seed)

    if "reanalyze" in self.config.use_gpu_for:
      if torch.cuda.is_available():
        if self.config.learner_gpu_device_id is not None:
          device_id = self.config.learner_gpu_device_id
          self.device = torch.device("cuda:{}".format(device_id))
        else:
          self.device = torch.device("cuda")
      else:
        raise RuntimeError("GPU was requested but torch.cuda.is_available() is False.")
    else:
      self.device = torch.device("cpu")

    self.network = get_network(config, self.device)
    self.network.to(self.device)
    self.network.eval()

    if state is not None:
      self.load_state(state)

    Logger.__init__(self)

  def load_state(self, state):
    self.run_tag = os.path.join(self.run_tag, 'resumed', '{}'.format(state['training_step']))
    self.network.load_state_dict(state['weights'])

  def sync_weights(self, force=False):
    weights, training_step = ray.get(self.storage.get_weights.remote(None, None))
    if training_step != self.training_step or force:
      self.network.load_weights(weights)
      self.training_step = training_step

  def reanalyze(self):
    while ray.get(self.replay_buffer.size.remote()) < self.config.reanalyze_batch_size:
      time.sleep(1)

    while True:
      if self.training_step % self.config.reanalyze_frequency == 0:
        histories = ray.get(self.replay_buffer.sample_game.remote())
        for priority, history in histories.values():
          observations = (torch.tensor(np.array(history.observations)).float().to(self.device))
          network_outputs = self.network.initial_inference(observations)

          for i in range(len(history.root_values)):
            history.root_values[i] = network_outputs.value[i].item()
          self.histories_reanalyzed += 1

          if self.histories_reanalyzed % self.config.reanalyze_log_frequency == 0:
            length_to_log = self.histories_reanalyzed/self.config.reanalyze_log_frequency
            self.log_scalar(tag='reanalyze/length', value=length_to_log, i=self.histories_reanalyzed)

        self.replay_buffer.update_game_history.remote(histories)

      self.sync_weights()

  def launch(self):
    print("Reanalyse is online on {}.".format(self.device))
    self.reanalyze()
