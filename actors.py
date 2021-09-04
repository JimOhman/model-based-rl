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


@ray.remote(num_cpus=1)
class Actor(Logger):

  def __init__(self, worker_id, config, storage, replay_buffer, state=None, run_tag=None, date=None):
    set_all_seeds(config.seed + worker_id if config.seed is not None else None)

    self.run_tag = run_tag
    self.group_tag = config.group_tag
    self.worker_id = 'actor-{}'.format(worker_id)
    self.config = deepcopy(config)
    self.storage = storage
    self.replay_buffer = replay_buffer

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.environment = get_environment(config)
    if config.seed is not None:
      self.environment.seed(config.seed)

    self.mcts = MCTS(config)

    self.network = get_network(config, self.device)
    self.network.eval()

    if config.fixed_temperatures:
      self.fixed_temperature = config.fixed_temperatures[worker_id]
      self.worker_id = 'actor-{}/temp={}'.format(worker_id, round(self.fixed_temperature, 1))
    else:
      self.fixed_temperature = None

    if self.config.norm_states:
      self.obs_min = np.array(self.config.state_range[::2], dtype=np.float32)
      self.obs_max = np.array(self.config.state_range[1::2], dtype=np.float32)
      self.obs_range = self.obs_max - self.obs_min

    self.training_step, self.games_played = 0, 0
    self.return_to_log, self.length_to_log = 0, 0
    self.value_to_log = 0
    if self.config.two_players:
      self.stats_to_log = defaultdict(int)

    if state is not None:
      self.load_state(state, date)

    self.verbose = True if 'actors' in self.config.verbose else False
    self.log = True if 'actors' in self.config.log else False

    self.histories = []
    self.experiences_collected = 0
    self.prev_revisit_step = 0

    Logger.__init__(self)

  def revisit_history(self):
    game = self.config.new_game(self.environment)

    idx, priority, step, history = ray.get(self.replay_buffer.get_max_history.remote())

    game.history.observations.append(history.observations[step])
    game.environment._elapsed_steps = history.steps[step]
    game.step = history.steps[step]
    game.environment.unwrapped.restore_state(history.env_states[step].copy())

    self.play_game(game)

    self.prev_revisit_step = self.games_played

    if self.log:
      self.log_scalar(tag='revisit/priority', value=priority, i=self.games_played)
      self.log_scalar(tag='revisit/step', value=history.steps[step], i=self.games_played)
      self.log_scalar(tag='revisit/length', value=game.step, i=self.games_played)
      self.log_scalar(tag='revisit/return', value=game.sum_rewards, i=self.games_played)
      self.log_scalar(tag='revisit/value', value=(game.sum_values/game.history_idx), i=self.games_played)

  def load_state(self, state, date):
    self.network.load_state_dict(state['weights'])
    self.training_step = state['step']
    old_run_tag = state['dirs']['base'].split('{}'.format(self.config.group_tag))[1][1:]
    self.run_tag = old_run_tag + '_resumed_' + date

  def sync_weights(self, force=False):
    weights, training_step = ray.get(self.storage.latest_weights.remote())
    if training_step != self.training_step:
      self.network.load_weights(weights)
      self.training_step = training_step
    elif force:
      self.network.load_weights(weights)

  def run_selfplay(self):
    while ray.get(self.storage.latest_weights.remote())[0] is None:
      time.sleep(1)

    self.sync_weights(force=True)

    while self.training_step < self.config.training_steps:
      game = self.config.new_game(self.environment)

      self.play_game(game)

      self.value_to_log += (game.sum_values/game.history_idx)
      self.return_to_log += game.sum_rewards
      self.length_to_log += game.step
      self.games_played += 1

      if self.verbose:
        date = datetime.datetime.now(tz=pytz.timezone('Europe/Stockholm')).strftime("%d-%b-%Y_%H:%M:%S")
        msg = "{}: [{}] Step {}, Game {} --> length: {}, return: {}"
        print(msg.format(self.worker_id.capitalize(), date, self.training_step,
                         self.games_played, game.step, game.sum_rewards))

      if self.log and self.games_played % self.config.actor_log_frequency == 0:
        return_to_log = self.return_to_log / self.config.actor_log_frequency
        length_to_log = self.length_to_log / self.config.actor_log_frequency
        value_to_log = self.value_to_log / self.config.actor_log_frequency
        self.log_scalar(tag='games/return', value=return_to_log, i=self.games_played)
        self.log_scalar(tag='games/length', value=length_to_log, i=self.games_played)
        self.log_scalar(tag='games/value', value=value_to_log, i=self.games_played)
        self.return_to_log = 0
        self.length_to_log = 0
        self.value_to_log = 0

      if self.log and self.config.two_players and self.games_played % 100 == 0:
        value_dict = {key:value/100 for key, value in self.stats_to_log.items()}
        self.log_scalars(group_tag='games/stats', value_dict=value_dict, i=self.games_played)
        self.stats_to_log = defaultdict(int)

      if self.config.revisit:
        if (self.games_played - self.prev_revisit_step) == self.config.revisit_frequency:
          self.revisit_history()

    self.sync_weights(force=True)

  def play_game(self, game):
    if self.fixed_temperature is None:
      temperature = self.config.visit_softmax_temperature(self.training_step)
    else:
      temperature = self.fixed_temperature

    while not game.terminal:
      root = Node(0)

      current_observation = np.float32(game.get_observation(-1))
      if self.config.norm_states:
        current_observation = (current_observation - self.obs_min) / self.obs_range
      current_observation = torch.from_numpy(current_observation).to(self.device)

      initial_inference = self.network.initial_inference(current_observation.unsqueeze(0))

      legal_actions = game.environment.legal_actions()
      root.expand(initial_inference, game.to_play, legal_actions)
      root.add_exploration_noise(self.config.root_dirichlet_alpha, self.config.root_exploration_fraction)

      self.mcts.run(root, self.network)

      error = root.value() - initial_inference.value.item()
      game.history.errors.append(error)

      action = self.config.select_action(root, temperature)

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
    print("{} is online.".format(self.worker_id.capitalize()))
    with torch.no_grad():
      self.run_selfplay()
