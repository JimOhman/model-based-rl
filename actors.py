from utils import get_network, get_environment
from mcts import MCTS, Node
from logger import Logger
from copy import deepcopy
import numpy as np
import datetime
import pytz
import time
import torch
import ray


@ray.remote
class Actor(Logger):

    def __init__(self, worker_id, config, storage, replay_buffer, state=None, run_tag=None, date=None):
        self.run_tag = run_tag
        self.worker_id = 'actor-{}'.format(worker_id)
        self.config = deepcopy(config)
        self.storage = storage
        self.replay_buffer = replay_buffer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.environment = get_environment(config)
        self.mcts = MCTS(config)

        self.network = get_network(config, self.device)
        self.network.eval()

        self.training_step, self.games_played = 0, 0
        self.return_to_log, self.length_to_log = 0, 0

        if state is not None:
          self.load_state(state, date)

        self.verbose = True if 'actors' in self.config.verbose else False
        self.render = True if 'actors' in self.config.render else False
        self.log = True if 'actors' in self.config.log else False

        if self.log:
          Logger.__init__(self)

    def load_state(self, state, date):
        self.network.load_state_dict(state['weights'])
        self.training_step = state['step']
        old_run_tag = state['dirs']['base'].split('/')[-1]
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

            self.return_to_log += game.sum_rewards
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

            self.sync_weights()

    def play_game(self):
      game = self.config.new_game(self.environment)
      temperature = self.config.visit_softmax_temperature(self.training_step)
      while not game.terminal and game.step < self.config.max_steps:
        root = Node(0)

        current_observation = self.config.to_torch(game.get_observation(-1), self.device, scale=255.0).unsqueeze(0)
        initial_inference = self.network.initial_inference(current_observation)

        root.expand(network_output=initial_inference)
        root.add_exploration_noise(self.config.root_dirichlet_alpha, self.config.root_exploration_fraction)

        self.mcts.run(root, self.network)

        error = root.value() - initial_inference.value.item()
        game.history.errors.append(error)

        action = self.config.select_action(root, temperature)

        game.apply(action)
        game.store_search_statistics(root)

        if game.step % self.config.max_sequence_length == 0 or game.done:
          overlap = self.config.num_unroll_steps + self.config.td_steps
          ignore = overlap if not game.done else None
          collect_from = max(0, game.previous_collect_to - overlap)
          history = game.get_history_sequence(collect_from)
          self.replay_buffer.save_history.remote(history, ignore=ignore)

      return game

    def launch(self):
        print("{} is online.".format(self.worker_id.capitalize()))
        with torch.no_grad():
            self.run_selfplay()