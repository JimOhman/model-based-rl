from mcts import MCTS, Node
from utils import get_network, get_environment, set_all_seeds
from visualize_mcts import write_mcts_as_png
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
plt.style.use('dark_background')
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


class SummaryTools(object):

  @staticmethod
  def moving_average(array, size=3) :
      array = np.cumsum(array, dtype=float)
      array[size:] = array[size:] - array[:-size]
      return array[size - 1:] / size

  def print_summary(self, games):
    lengths = [game.step for game in games]
    average_length, std_lengths = np.round(np.mean(lengths), 1), np.round(np.std(lengths), 1)

    returns = [game.sum_rewards for game in games]
    average_return, std_return = np.round(np.mean(returns), 1), np.round(np.std(returns), 1)

    pred_returns = [sum(game.pred_rewards) for game in games]
    average_pred_return, std_pred_return = np.round(np.mean(pred_returns), 1), np.round(np.std(pred_returns), 1)

    pred_values = [np.mean(game.pred_values) for game in games]
    average_pred_value, std_pred_value = np.round(np.mean(pred_values), 1), np.round(np.std(pred_values), 1)

    root_values = [np.mean(game.history.root_values) for game in games]
    average_root_value, std_root_value = np.round(np.mean(root_values), 1), np.round(np.std(root_values), 1)

    search_depths = [np.mean(game.search_depths) for game in games]
    average_search_depth, std_search_depth = np.round(np.mean(search_depths), 1), np.round(np.std(search_depths), 1)

    print("\n\033[92m[Evaluation of [{}] finished]\033[0m".format(self.config.eval_tag))
    print("   Average length: {}({})".format(average_length, std_lengths))
    print("   Average return: {}({})".format(average_return, std_return))
    print("   Average predicted return: {}({})".format(average_pred_return, std_pred_return))
    print("   Average predicted value: {}({})".format(average_pred_value, std_pred_value))
    print("   Average mcts value: {}({})".format(average_root_value, std_root_value))
    print("   Average search depth: {}({})".format(average_search_depth, std_search_depth))

  def get_quantiles(self, values, smooth=0):
      max_len = len(max(values, key=len))
      extended = list(zip(*[list(vals) + [vals[-1]] * (max_len-len(vals)) for vals in values]))
      center = np.quantile(extended, 0.5, axis=1)
      upper = np.quantile(extended, 0.75, axis=1)
      lower = np.quantile(extended, 0.25, axis=1)
      if smooth:
          center = self.moving_average(center, size=smooth)
          upper = self.moving_average(upper, size=smooth)
          lower = self.moving_average(lower, size=smooth)
      quantiles = {'0.5': center, '0.75': upper, '0.25': lower}
      return quantiles

  def get_axes(self):
    ax1 = plt.subplot2grid((6, 1), (0, 0))
    ax2 = plt.subplot2grid((6, 1), (1, 0))
    ax3 = plt.subplot2grid((6, 1), (2, 0))
    ax4 = plt.subplot2grid((6, 1), (3, 0))
    ax5 = plt.subplot2grid((6, 1), (4, 0))
    ax6 = plt.subplot2grid((6, 1), (5, 0))
    ax1.set_ylabel('Return', color='white', fontsize=15)
    ax1.tick_params(colors='white')
    ax2.set_ylabel('Predicted Return', color='white', fontsize=15)
    ax2.tick_params(colors='white')
    ax3.set_ylabel('Predicted Value', color='white', fontsize=15)
    ax3.tick_params(colors='white')
    ax4.set_ylabel('MCTS Value', color='white', fontsize=15)
    ax4.tick_params(colors='white')
    ax5.set_ylabel('Search Depth', color='white', fontsize=15)
    ax5.tick_params(colors='white')
    ax6.set_ylabel('Policy', color='white', fontsize=15)
    ax6.tick_params(colors='white')
    return ax1, ax2, ax3, ax4, ax5, ax6

  def plot_summary(self, games, config, axes=None):
    ax1, ax2, ax3, ax4, ax5, ax6 = self.get_axes() if axes is None else axes

    rewards = [game.history.rewards for game in games]
    pred_rewards = [game.pred_rewards for game in games]
    pred_values = [game.pred_values for game in games]
    root_values = [game.history.root_values for game in games]
    search_depths = [game.search_depths for game in games]

    return_quantiles = self.get_quantiles([np.cumsum(rews) for rews in rewards], config.smooth)
    pred_return_quantiles = self.get_quantiles([np.cumsum(rews) for rews in pred_rewards], config.smooth)
    pred_value_quantiles = self.get_quantiles(pred_values, config.smooth)
    root_value_quantiles = self.get_quantiles(root_values, config.smooth)
    search_depth_quantiles = self.get_quantiles(search_depths, config.smooth)

    ax1.plot(return_quantiles['0.5'], linewidth=2, label=self.config.eval_tag)
    ax2.plot(pred_return_quantiles['0.5'], linewidth=2, label=self.config.eval_tag)
    ax3.plot(pred_value_quantiles['0.5'], linewidth=2, label=self.config.eval_tag)
    ax4.plot(root_value_quantiles['0.5'], linewidth=2, label=self.config.eval_tag)
    ax5.plot(search_depth_quantiles['0.5'], linewidth=2, label=self.config.eval_tag)

    if config.include_bounds:
      ax1.fill_between(range(len(return_quantiles['0.5'])), y1=return_quantiles['0.25'], y2=return_quantiles['0.75'], alpha=0.4)
      ax2.fill_between(range(len(pred_return_quantiles['0.5'])), y1=pred_return_quantiles['0.25'], y2=pred_return_quantiles['0.75'], alpha=0.4)
      ax3.fill_between(range(len(pred_value_quantiles['0.5'])), y1=pred_value_quantiles['0.25'], y2=pred_value_quantiles['0.75'], alpha=0.4)
      ax4.fill_between(range(len(root_value_quantiles['0.5'])), y1=root_value_quantiles['0.25'], y2=root_value_quantiles['0.75'], alpha=0.4)
      ax5.fill_between(range(len(search_depth_quantiles['0.5'])), y1=search_depth_quantiles['0.25'], y2=search_depth_quantiles['0.75'], alpha=0.4)

    if config.include_policy:
      policy = list(zip(*[zip(*game.history.child_visits) for game in games]))
      policy_quantiles = []
      for action in policy:
        policy_quantiles.append(self.get_quantiles(action, config.smooth))
      for i, action_quantiles in enumerate(policy_quantiles):
        label = '{} - action: {}'.format(self.config.eval_tag, i)
        ax6.plot(action_quantiles['0.5'], linewidth=2, label=label)
        if config.include_bounds:
          ax6.fill_between(steps, y1=action_quantiles['0.25'], y2=action_quantiles['0.75'], alpha=0.4)
      ax6.set_xlabel('Steps', color='white', fontsize=15)
    else:
      ax6.set_visible(False)
      ax5.set_xlabel('Steps', color='white', fontsize=15)

    return ax1, ax2, ax3, ax4, ax5, ax6


class Evaluator(SummaryTools):

    def __init__(self, state):
        self.config = state['config']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.environment = get_environment(self.config)

        self.network = get_network(self.config, self.device)
        self.network.load_state_dict(state['weights'])
        self.network.eval()
        self.mcts = MCTS(self.config)

        if 'evaluation' in self.config.render:
          self.render = True
          self.viewer = ImageViewer()
        else:
          self.render = False

    def play_game(self, device):
      game = self.config.new_game(self.environment)
      game.pred_values = []
      game.pred_rewards = []
      game.search_depths = []
      while not game.terminal and game.step < self.config.max_steps:
        root = Node(0)

        current_observation = self.config.to_torch(game.get_observation(-1), device, scale=True).unsqueeze(0)
        initial_inference = self.network.initial_inference(current_observation)

        root.expand(network_output=initial_inference)

        if self.config.use_exploration_noise:
          root.add_exploration_noise(self.config.root_dirichlet_alpha, self.config.root_exploration_fraction)

        actions = []
        rewards = []
        if self.config.only_prior:
          action = np.argmax([child.prior for child in root.children])
          reward = self.network.recurrent_inference(root.hidden_state, [action]).reward.item()

          actions.append(action)
          rewards.append(reward)

          root.children[action].visit_count += 1
          game.search_depths.append(0)
        elif self.config.only_value:
          outputs = [self.network.recurrent_inference(root.hidden_state, [action]) for action in range(len(root.children))]
          action = np.argmax([out.reward.item() + self.config.discount * out.value.item() for out in outputs])
          reward = outputs[action].reward.item()

          actions.append(action)
          rewards.append(reward)
          
          root.children[action].visit_count += 1
          game.search_depths.append(1)
        else:
          mcts = self.mcts.run(root, self.network)
          if self.config.save_mcts_to_path:
            write_mcts_as_png(mcts, path_to_file=self.config.save_mcts_to_path + str(game.step))
          game.search_depths.append(max(map(len, mcts)) - 1)

          depth = 0
          node = root
          while node.expanded():
            action = self.config.select_action(root, temperature=self.config.temperature)
            reward = node.children[action].reward

            actions.append(action)
            rewards.append(reward)

            node = node.children[action]
            depth += 1

            if depth == self.config.apply_mcts_steps:
              break

        game.pred_values.append(initial_inference.value.item())
        game.store_search_statistics(root)

        for action, reward in zip(actions, rewards):
          game.apply(action)
          game.pred_rewards.append(reward)

          if self.render:
            try:
              img = self.environment.obs_buffer.max(axis=0)
              self.viewer.imshow(img)
            except:
              self.environment.render()

          if game.terminal or game.step > self.config.max_steps:
            break

      tag = "\033[92m[Game done]\033[0m"
      print("\n" + tag + " --> length: {}, return: {}, pred return: {}, pred value: {}, mcts value: {}\n".format(game.step,
                                                                                                                 round(game.sum_rewards, 1),
                                                                                                                 round(sum(game.pred_rewards), 1),
                                                                                                                 np.round(np.mean(game.pred_values), 1),
                                                                                                                 np.round(np.mean(game.history.root_values), 1)))
      return game


def plot_aggregate(data, config):
  ax1 = plt.subplot2grid((5, 1), (0, 0))
  ax2 = plt.subplot2grid((5, 1), (1, 0))
  ax3 = plt.subplot2grid((5, 1), (2, 0))
  ax4 = plt.subplot2grid((5, 1), (3, 0))
  ax5 = plt.subplot2grid((5, 1), (4, 0))
  ax1.set_ylabel('Return', color='white', fontsize=15)
  ax1.tick_params(colors='white')
  ax2.set_ylabel('Predicted Return', color='white', fontsize=15)
  ax2.tick_params(colors='white')
  ax3.set_ylabel('Predicted Value', color='white', fontsize=15)
  ax3.tick_params(colors='white')
  ax4.set_ylabel('MCTS Value', color='white', fontsize=15)
  ax4.tick_params(colors='white')
  ax5.set_ylabel('Lenght', color='white', fontsize=15)
  ax5.tick_params(colors='white')

  return_quantiles = {'0.25': [], '0.5': [], '0.75': []}
  pred_return_quantiles = {'0.25': [], '0.5': [], '0.75': []}
  pred_values_quantiles = {'0.25': [], '0.5': [], '0.75': []}
  root_values_quantiles = {'0.25': [], '0.5': [], '0.75': []}
  lengths_quantiles = {'0.25': [], '0.5': [], '0.75': []}
  for games in data.values():
    returns = [sum(game.history.rewards) for game in games]
    return_quantiles['0.25'].append(np.quantile(returns, 0.25))
    return_quantiles['0.5'].append(np.quantile(returns, 0.5))
    return_quantiles['0.75'].append(np.quantile(returns, 0.75))

    pred_returns = [sum(game.pred_rewards) for game in games]
    pred_return_quantiles['0.25'].append(np.quantile(pred_returns, 0.25))
    pred_return_quantiles['0.5'].append(np.quantile(pred_returns, 0.5))
    pred_return_quantiles['0.75'].append(np.quantile(pred_returns, 0.75))

    pred_values = [np.mean(game.pred_values) for game in games]
    pred_values_quantiles['0.25'].append(np.quantile(pred_values, 0.25))
    pred_values_quantiles['0.5'].append(np.quantile(pred_values, 0.5))
    pred_values_quantiles['0.75'].append(np.quantile(pred_values, 0.75))

    root_values = [np.mean(game.history.root_values) for game in games]
    root_values_quantiles['0.25'].append(np.quantile(root_values, 0.25))
    root_values_quantiles['0.5'].append(np.quantile(root_values, 0.5))
    root_values_quantiles['0.75'].append(np.quantile(root_values, 0.75))

    lengths = [len(game.history.rewards) for game in games]
    lengths_quantiles['0.25'].append(np.quantile(lengths, 0.25))
    lengths_quantiles['0.5'].append(np.quantile(lengths, 0.5))
    lengths_quantiles['0.75'].append(np.quantile(lengths, 0.75))

  steps = list(data.keys())
  ax1.plot(steps, return_quantiles['0.5'], linewidth=2)
  ax2.plot(steps, pred_return_quantiles['0.5'], linewidth=2)
  ax3.plot(steps, pred_values_quantiles['0.5'], linewidth=2)
  ax4.plot(steps, root_values_quantiles['0.5'], linewidth=2)
  ax5.plot(steps, lengths_quantiles['0.5'], linewidth=2)
  if config.include_bounds:
    ax1.fill_between(steps, y1=return_quantiles['0.25'], y2=return_quantiles['0.75'], alpha=0.4)
    ax2.fill_between(steps, y1=pred_return_quantiles['0.25'], y2=pred_return_quantiles['0.75'], alpha=0.4)
    ax3.fill_between(steps, y1=pred_values_quantiles['0.25'], y2=pred_values_quantiles['0.75'], alpha=0.4)
    ax4.fill_between(steps, y1=root_values_quantiles['0.25'], y2=root_values_quantiles['0.75'], alpha=0.4)
    ax5.fill_between(steps, y1=lengths_quantiles['0.25'], y2=lengths_quantiles['0.75'], alpha=0.4)

def get_eval_tag(state, config, idx):
    if config.detailed_eval_tag:
      tag = 'path: {}'.format(idx)
      tag += ', net: {}'.format(state['step'])

      if state['config'].only_value:
        tag += ', only value'
      elif state['config'].only_prior:
        tag += ', only prior'
      else:
        tag += ', sims: {}'.format(state['config'].num_simulations)
        tag += ', mcts-steps: {}'.format(state['config'].apply_mcts_steps)
        if state['config'].temperature:
          tag += ', temp: {}'.format(state['config'].temperature)
        if state['config'].use_exploration_noise:
          tag += ', with noise'.format(state['config'].use_exploration_noise)

    else:
      tag = '{}'.format(state['step'])
    return tag

def get_states(config):
  state_paths = [(saves_dir + net) for saves_dir in config.saves_dir for net in config.evaluate_nets]

  states = []
  for idx, saves_dir in enumerate(config.saves_dir):
    for net in config.evaluate_nets:
      meta_state = torch.load(saves_dir + net)
      for temperature in config.eval_temperatures:
        for num_simulations in config.num_simulations:
          for only_prior in config.only_prior:
            for only_value in config.only_value:
              for use_exploration_noise in config.use_exploration_noise:
                for apply_mcts_steps in config.apply_mcts_steps:
                  if only_prior is True and only_value is True:
                    continue
                  state = deepcopy(meta_state)
                  state['config'].num_simulations = num_simulations
                  state['config'].temperature = temperature
                  state['config'].only_value = only_value
                  state['config'].only_prior = only_prior
                  state['config'].use_exploration_noise = use_exploration_noise
                  state['config'].apply_mcts_steps = apply_mcts_steps
                  state['config'].eval_tag = get_eval_tag(state, config, idx)
                  state['config'].render = config.render
                  state['config'].save_mcts_to_path = config.save_mcts_to_path
                  states.append(state)
  return states

@ray.remote
def run(evaluator, seed=None):
    if seed is None:
      seed = np.random.randint(0, 1000)
    set_all_seeds(seed)
    evaluator.environment.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        game = evaluator.play_game(device)

    game.history._replace(observations = [])

    return game

if __name__ == '__main__':
    from config import make_config
    import pyglet
    import cv2
    from pyglet.gl import *

    ray.init()

    config = make_config()

    states = get_states(config)

    evaluators = []
    for state in states:
      evaluators.append(Evaluator(state))

    print("\033[92mStarting a {} episode evaluation of {} configurations\033[0m...".format(config.games_per_evaluation, len(states)))
    
    if config.plot_summary:
      axes = None

    base_seed = config.seed[0] if config.seed[0] is not None else 0
    
    data = {}
    for evaluator in evaluators:
      print("\n\033[92mEvaluating [{}]\033[0m...".format(evaluator.config.eval_tag))

      games = ray.get([run.remote(evaluator, (base_seed + seed)) for seed in range(1, config.games_per_evaluation + 1)])
      data[evaluator.config.eval_tag] = games

      evaluator.print_summary(games)

      if config.plot_summary:
        axes = evaluator.plot_summary(games, config, axes)
      
    if config.plot_summary:
        [ax.legend(framealpha=0.2) for ax in axes]
        [ax.grid(alpha=0.3) for ax in axes]
        plt.show()

    if config.plot_aggregate:
      plot_aggregate(data, config)
      plt.show()
