from utils import get_network, get_environment, select_action, to_torch, get_error
from mcts import MCTS, Node
from logger import Logger
import numpy as np
import datetime
import pytz
import time
import torch
import ray


@ray.remote(num_cpus=2)
class Actor(Logger):

    def __init__(self, run_tag, worker_id, config, storage, replay_buffer):
        self.run_tag = run_tag
        self.worker_id = 'actor-{}'.format(worker_id)
        self.config = config
        self.storage = storage
        self.replay_buffer = replay_buffer

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.environment = get_environment(config)
        self.network = get_network(config, self.device)
        self.mcts = MCTS(config)
        self.select_action = select_action
        self.get_error = get_error
        self.to_torch = to_torch

        self.training_step = 0
        self.games_played = 0
        self.sum_rewards = 0
        self.sum_lengths = 0

        if 'actors' in self.config.verbose:
          self.verbose = True
        else:
          self.verbose = False

        if 'actors' in self.config.log:
          self.log = True
        else:
          self.log = False

        Logger.__init__(self)

    def run_selfplay(self):

        while ray.get(self.storage.latest_weights.remote()) is None:
            time.sleep(1)

        while True:
            weights = ray.get(self.storage.latest_weights.remote())
            self.network.load_weights(weights)

            game = self.play_game()

            self.games_played += 1
            self.sum_rewards += game.sum_rewards
            self.sum_lengths += game.step

            if self.verbose:
              date = datetime.datetime.now(tz=pytz.timezone('Europe/Stockholm')).strftime("%d-%b-%Y_%H:%M:%S")
              print("[{}] Training step {}, Game {} --> length: {}, return: {}".format(date, self.training_step, self.games_played, game.step, game.sum_rewards))

            if self.log and self.games_played % self.config.actor_log_frequency == 0:
              self.log_scalar(tag='games/return', value=(self.sum_rewards/self.config.actor_log_frequency), i=self.games_played)
              self.log_scalar(tag='games/length', value=(self.sum_lengths/self.config.actor_log_frequency), i=self.games_played)
              self.sum_rewards = 0
              self.sum_lengths = 0

    def play_game(self):
        game = self.config.new_game(self.environment)

        while not game.terminal and game.step < self.config.max_steps:

          root = Node(0)

          current_observation = self.to_torch(game.get_observation(-1), self.device).unsqueeze(0).detach()
          initial_inference = self.network.initial_inference(current_observation)

          root.expand(network_output=initial_inference)
          root.add_exploration_noise(self.config.root_dirichlet_alpha, self.config.root_exploration_fraction)

          self.mcts.run(root, self.network)

          error = self.get_error(root, initial_inference)
          game.history.errors.append(error)

          temperature = self.get_temperature()
          action = self.select_action(root, temperature)

          game.apply(action)
          game.store_search_statistics(root)

          if (game.step % self.config.save_frequency) == 0 or game.terminal:
            overlap = self.config.num_unroll_steps + self.config.td_steps
            length = self.config.save_frequency + overlap
            history = game.get_history_sequence(length)
            ignore = overlap if not game.terminal else None
            history = game.history
            self.replay_buffer.save_history.remote(history, ignore=None)

        return game

    def get_temperature(self):
        training_step = ray.get(self.storage.get_stats.remote(tag="training step"))
        self.training_step = training_step
        temperature = self.config.visit_softmax_temperature(training_step)
        return temperature

    def launch(self):
        print("{} is online.".format(self.worker_id.capitalize()))
        self.run_selfplay()