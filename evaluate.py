from mcts import MCTS, Node
from utils import get_network, get_environment
from logger import Logger
import numpy as np
import torch
import ray
import random

class ImageViewer(object):
  def __init__(self, display=None):
      self.window = None
      self.isopen = False
      self.display = display

  def imshow(self, arr):
      if self.window is None:
          height, width, _channels = arr.shape
          height, width = 3*height, 3*width
          self.window = pyglet.window.Window(width=width, height=height, 
              display=self.display, vsync=False, resizable=True)            
          self.width = width
          self.height = height
          self.isopen = True

          @self.window.event
          def on_resize(width, height):
              self.width = width
              self.height = height

          @self.window.event
          def on_close():
              self.isopen = False

      assert len(arr.shape) == 3, "Wrong shape! Make sure to include the color channel."
      image = pyglet.image.ImageData(arr.shape[1], arr.shape[0], 
          'RGB', arr.tobytes(), pitch=arr.shape[1]*-3)
      gl.glTexParameteri(gl.GL_TEXTURE_2D, 
          gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
      texture = image.get_texture()
      texture.width = self.width
      texture.height = self.height
      self.window.clear()
      self.window.switch_to()
      self.window.dispatch_events()
      texture.blit(0, 0) # draw
      self.window.flip()

  def close(self):
      if self.isopen and sys.meta_path:
          self.window.close()
          self.isopen = False

  def __del__(self):
      self.close()


@ray.remote
class Evaluator(Logger):

    def __init__(self, config, run_tag=None, directories=None, log_tag=''):
        self.run_tag = run_tag
        self.worker_id = 'evaluator'
        self.config = config

        self.log_tag = log_tag + '_' if log_tag else log_tag

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.environment = get_environment(config)
        self.network = get_network(config, self.device)
        self.mcts = MCTS(config)

        self.returns = {}
        self.errors = {}

        if 'evaluation' in config.render:
          self.render = True
          self.viewer = ImageViewer()
        else:
          self.render = False

        self.verbose = True if 'evaluation' in config.verbose else False
        self.log = True if 'evaluation' in config.log else False

        if self.log:
          Logger.__init__(self, directories)

    def evaluate_network(self, training_step, weights):
        self.network.load_state_dict(weights)
        self.network.eval()

        games = []
        for n in range(self.config.games_per_evaluation):
            game = self.config.new_game(self.environment)
            while not game.terminal and game.step < self.config.max_steps:
              root = Node(0)

              current_observation = self.config.to_torch(game.get_observation(-1), self.device, scale=255.0).unsqueeze(0)
              initial_inference = self.network.initial_inference(current_observation)

              root.expand(network_output=initial_inference)

              self.mcts.run(root, self.network)

              error = root.value() - initial_inference.value.item()
              game.history.errors.append(error)

              action = self.config.select_action(root)
              game.apply(action)

              if self.render:
                try:
                  img = self.environment.obs_buffer.max(axis=0)
                  self.viewer.imshow(img)
                except:
                  self.environment.render()
            games.append(game)

        returns = [game.sum_rewards for game in games]
        average_return, std_return = np.round(np.mean(returns), 1), np.round(np.std(returns), 1)

        lengths = [game.step for game in games]
        average_length, std_lengths = np.round(np.mean(lengths), 1), np.round(np.std(lengths), 1)

        errors = [np.abs(np.mean(game.history.errors)) for game in games]
        average_error, std_error = np.round(np.mean(errors), 1), np.round(np.std(errors), 1)

        self.returns[training_step] = [average_return, std_return]
        self.errors[training_step] = [average_error, std_error]

        if self.verbose:
            print("\n[Evaluation at step {}] --> return: {}({}), error: {}({})\n".format(training_step, average_return, std_return, average_error, std_error))

        if self.log:
          self.log_scalar(tag=(self.log_tag + 'evaluation/return'), value=average_return, i=training_step)
          self.log_scalar(tag=(self.log_tag + 'evaluation/length'), value=average_length, i=training_step)

    def run_one_episode(self, device):
      game = self.config.new_game(self.environment)
      while not game.terminal and game.step < self.config.max_steps:
        root = Node(0)

        current_observation = self.config.to_torch(game.get_observation(-1), device, scale=255.0).unsqueeze(0)
        initial_inference = self.network.initial_inference(current_observation)

        root.expand(network_output=initial_inference)

        self.mcts.run(root, self.network)

        error = root.value() - initial_inference.value.item()
        game.history.errors.append(error)

        action = self.config.select_action(root)
        game.apply(action)

        if self.render:
          try:
            img = self.environment.obs_buffer.max(axis=0)
            self.viewer.imshow(img)
          except:
            self.environment.render()
      return game

@ray.remote
def run(evaluator):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        data = evaluator.run_one_episode(device)
    return data

if __name__ == '__main__':
    from config import make_config
    import pyglet
    import cv2
    from pyglet.gl import *

    ray.init()

    config = make_config()

    state = torch.load(config.load_state)

    state['config'].games_per_evaluation = config.games_per_evaluation
    state['config'].num_simulations = config.num_simulations
    state['config'].verbose = config.verbose
    state['config'].render = config.render
    state['config'].log = config.log

    evaluator = Evaluator.remote(state['config'], log_tag='local', directories=state['dirs'])
    ray.get(evaluator.evaluate_network.remote(state['step'], state['weights']))
