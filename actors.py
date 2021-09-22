from utils import get_network, get_environment, set_all_seeds
from collections import defaultdict
from mcts import MCTS, Node
from logger import Logger
from copy import deepcopy
import numpy as np
import datetime
import pytz
import time
import torch
import ray
import random
import os


@ray.remote
class Actor(Logger):

  def __init__(self, actor_key, config, storage, replay_buffer, state=None):
    set_all_seeds(config.seed + actor_key if config.seed is not None else None)

    self.run_tag = config.run_tag
    self.group_tag = config.group_tag
    self.actor_key = actor_key
    self.config = deepcopy(config)
    self.storage = storage
    self.replay_buffer = replay_buffer

    self.environment = get_environment(config)
    self.environment.seed(config.seed)
    self.mcts = MCTS(config)

    if "actors" in self.config.use_gpu_for:
      if torch.cuda.is_available():
        if self.config.actors_gpu_device_ids is not None:
          device_id = self.config.actors_gpu_device_ids[self.actor_key]
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

    if config.fixed_temperatures:
      self.temperature = config.fixed_temperatures[self.actor_key]
      self.worker_id = 'actors/temp={}'.format(round(self.temperature, 1))
    else:
      self.worker_id = 'actor-{}'.format(self.actor_key)

    if self.config.norm_obs:
      self.obs_min = np.array(self.config.obs_range[::2], dtype=np.float32)
      self.obs_max = np.array(self.config.obs_range[1::2], dtype=np.float32)
      self.obs_range = self.obs_max - self.obs_min

    if self.config.two_players:
      self.stats_to_log = defaultdict(int)

    self.experiences_collected = 0
    self.training_step = 0
    self.games_played = 0
    self.return_to_log = 0
    self.length_to_log = 0
    self.value_to_log = 0

    if state is not None:
      self.load_state(state)

    Logger.__init__(self)

  def load_state(self, state):
    self.run_tag = os.path.join(self.run_tag, 'resumed', '{}'.format(state['training_step']))
    self.network.load_state_dict(state['weights'])
    self.training_step = state['training_step']
    self.games_played = state['actor_games'][self.actor_key]

  def sync_weights(self, force=False):
    weights, training_step = ray.get(self.storage.get_weights.remote(self.games_played, self.actor_key))
    if training_step != self.training_step or force:
      self.network.load_weights(weights)
      self.training_step = training_step

  def run_selfplay(self):

    while not ray.get(self.storage.is_ready.remote()):
      time.sleep(1)

    self.sync_weights(force=True)

    while self.training_step < self.config.training_steps:
      game = self.config.new_game(self.environment)

      self.play_game(game)

      self.value_to_log += (game.sum_values/game.history_idx)
      self.return_to_log += game.sum_rewards
      self.length_to_log += game.step
      self.games_played += 1

      if self.games_played % self.config.actor_log_frequency == 0:
        return_to_log = self.return_to_log / self.config.actor_log_frequency
        length_to_log = self.length_to_log / self.config.actor_log_frequency
        value_to_log = self.value_to_log / self.config.actor_log_frequency
        self.log_scalar(tag='games/return', value=return_to_log, i=self.games_played)
        self.log_scalar(tag='games/length', value=length_to_log, i=self.games_played)
        self.log_scalar(tag='games/value', value=value_to_log, i=self.games_played)
        self.return_to_log = 0
        self.length_to_log = 0
        self.value_to_log = 0

      if self.config.two_players and self.games_played % 100 == 0:
        value_dict = {key:value/100 for key, value in self.stats_to_log.items()}
        self.log_scalars(group_tag='games/stats', value_dict=value_dict, i=self.games_played)
        self.stats_to_log = defaultdict(int)

    self.sync_weights(force=True)

  def play_game(self, game):

    if self.config.fixed_temperatures is not None:
      self.temperature = self.config.visit_softmax_temperature(self.training_step)

    while not game.terminal:
      root = Node(0)

      current_observation = np.float32(game.get_observation(-1))
      if self.config.norm_obs:
        current_observation = (current_observation - self.obs_min) / self.obs_range
      current_observation = torch.from_numpy(current_observation).to(self.device)

      initial_inference = self.network.initial_inference(current_observation.unsqueeze(0))

      legal_actions = game.environment.legal_actions()
      root.expand(initial_inference, game.to_play, legal_actions)
      root.add_exploration_noise(self.config.root_dirichlet_alpha, self.config.root_exploration_fraction)

      self.mcts.run(root, self.network)

      error = root.value() - initial_inference.value.item()
      game.history.errors.append(error)

      action = self.config.select_action(root, self.temperature)

      game.apply(action)
      game.store_search_statistics(root)

      self.experiences_collected += 1

      if self.experiences_collected % self.config.weight_sync_frequency == 0:
        self.sync_weights()

      save_history = (game.history_idx - game.previous_collect_to) == self.config.max_history_length
      if save_history or game.done or game.terminal:
        overlap = self.config.num_unroll_steps + self.config.td_steps
        if not game.history.dones[game.previous_collect_to - 1]:
          collect_from = max(0, game.previous_collect_to - overlap)
        else:
          collect_from = game.previous_collect_to
        history = game.get_history_sequence(collect_from)
        ignore = overlap if not game.done else None
        self.replay_buffer.save_history.remote(history, ignore=ignore, terminal=game.terminal)

      if game.step >= self.config.max_steps:
        self.environment.was_real_done = True
        break

    if self.config.two_players:
      self.stats_to_log[game.info["result"]] += 1

  def launch(self):
    print("{} is online on {}.".format(self.worker_id.capitalize(), self.device))
    with torch.inference_mode():
      self.run_selfplay()

