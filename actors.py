from utils import get_network, get_environment, set_all_seeds
from mcts import MCTS, Node
from logger import Logger
from copy import deepcopy
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
      self.temperature = config.fixed_temperatures[worker_id]
      self.worker_id = 'actor-{}/temp={}'.format(worker_id, round(self.temperature, 1))
    else:
      self.temperature = None

    self.training_step, self.games_played = 0, 0
    self.return_to_log, self.length_to_log = 0, 0

    if state is not None:
      self.load_state(state, date)

    self.verbose = True if 'actors' in self.config.verbose else False
    self.log = True if 'actors' in self.config.log else False

    self.histories = []
    self.prev_reanalyze_step = 0

    Logger.__init__(self)

  def reanalyze_history(self, history, keep_local=False):
    for idx, observation in enumerate(history.observations[:-1]):
      root = Node(0)

      current_observation = self.config.to_torch(observation, self.device).unsqueeze(0)
      initial_inference = self.network.initial_inference(current_observation)

      root.expand(network_output=initial_inference)
      root.add_exploration_noise(self.config.root_dirichlet_alpha, self.config.root_exploration_fraction)

      self.mcts.run(root, self.network)

      error = root.value() - initial_inference.value.item()
      sum_visits = sum(child.visit_count for child in root.children)
      policy = [child.visit_count / sum_visits for child in root.children]
      root_value = root.value()

      history.child_visits[idx] = policy
      history.root_values[idx] = root_value
      history.errors[idx] = error

    if keep_local:
      self.histories.append(history)

    self.replay_buffer.save_history.remote(history)
    self.prev_reanalyze_step = self.training_step

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
      game = self.play_game()
      self.return_to_log += sum(game.history.rewards)
      self.length_to_log += game.step
      self.games_played += 1

      if self.verbose:
        date = datetime.datetime.now(tz=pytz.timezone('Europe/Stockholm')).strftime("%d-%b-%Y_%H:%M:%S")
        print("{}: [{}] Step {}, Game {} --> length: {}, return: {}".format(self.worker_id.capitalize(), date, self.training_step,
                                                                            self.games_played, game.step, game.sum_rewards))

      if self.log and self.games_played % self.config.actor_log_frequency == 0:
        return_to_log = self.return_to_log / self.config.actor_log_frequency
        length_to_log = self.length_to_log / self.config.actor_log_frequency
        self.log_scalar(tag='games/return', value=return_to_log, i=self.games_played)
        self.log_scalar(tag='games/length', value=length_to_log, i=self.games_played)
        self.return_to_log = 0
        self.length_to_log = 0

      if (self.training_step - self.prev_reanalyze_step) > self.config.reanalyze_frequency:
        if not self.histories:
          self.histories = self.replay_buffer.sample_histories(amount=10)

        for history in self.histories:
          self.reanalyze_history(history)

      self.sync_weights()

  def play_game(self, game=None):
    if game is None:
      game = self.config.new_game(self.environment)

    if self.temperature is None:
      temperature = self.config.visit_softmax_temperature(self.training_step)
    else:
      temperature = self.temperature

    while not game.terminal and game.step < self.config.max_steps:
      root = Node(0)

      current_observation = self.config.to_torch(game.get_observation(-1), self.device, norm=self.config.norm_states).unsqueeze(0)
      initial_inference = self.network.initial_inference(current_observation)

      root.expand(network_output=initial_inference)
      root.add_exploration_noise(self.config.root_dirichlet_alpha, self.config.root_exploration_fraction)

      self.mcts.run(root, self.network)

      error = root.value() - initial_inference.value.item()
      game.history.errors.append(error)

      action = self.config.select_action(root, temperature)

      game.apply(action)
      game.store_search_statistics(root)

      save_history = (game.step - game.previous_collect_to) >= self.config.max_history_length
      save_history = save_history or game.done or game.terminal
      if save_history:
        overlap = self.config.num_unroll_steps + self.config.td_steps
        if not game.history.dones[game.previous_collect_to - 1]:
          collect_from = max(0, game.previous_collect_to - overlap)
        else:
          collect_from = game.previous_collect_to
        history = game.get_history_sequence(collect_from)
        use_ignore = not game.done and not game.terminal
        ignore = overlap if use_ignore else None
        self.replay_buffer.save_history.remote(history, ignore=ignore)
    
    return game

  def launch(self):
    print("{} is online.".format(self.worker_id.capitalize()))
    with torch.no_grad():
      self.run_selfplay()
