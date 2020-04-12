from mcts import MCTS, Node
from utils import get_environment, select_action, get_error
import numpy as np


class Evaluator(object):

    def __init__(self):
        self._select_action = select_action
        self._get_error = get_error

        self._environment = get_environment(self.config)
        self._mcts = MCTS(self.config)

        self.returns = {}
        self.errors = {}

        if 'evaluation' in self.config.verbose:
          self._verbose = True
        else:
          self._verbose = False

    def evaluate_network(self):
        games = []
        for n in range(self.config.games_per_evaluation):
            game = self.config.new_game(self._environment)
            while not game.terminal and game.step < self.config.max_steps:
              root = Node(0)

              current_observation = self.to_torch(game.get_observation(-1), self.device).unsqueeze(0).detach()
              initial_inference = self.network.initial_inference(current_observation)
              
              root.expand(network_output=initial_inference)

              self._mcts.run(root, self.network)

              error = self._get_error(root, initial_inference)
              game.history.errors.append(error)

              action = self._select_action(root)

              game.apply(action)

              if self.config.render:
                environment.render()

            games.append(game)

        returns = [game.sum_rewards for game in games]
        average_return, std_return = np.round(np.mean(returns), 1), np.round(np.std(returns), 1)

        errors = [np.abs(np.mean(game.history.errors)) for game in games]
        average_error, std_error = np.round(np.mean(errors), 1), np.round(np.std(errors), 1)

        self.returns[self.training_step] = [average_return, std_return]
        self.errors[self.training_step] = [average_error, std_error]

        if self._verbose:
            print("\nEvaluation at step {}, Average return: {}({}), Average error: {}({})\n".format(self.training_step, average_return, std_return, average_error, std_error))

        if self.log:
          self.log_scalar(tag='evaluation/return', value=average_return, i=self.training_step)
          self.log_scalar(tag='evaluation/length', value=game.step, i=self.training_step)

