from mcts import MCTS, Node
from utils import get_network, get_environment, set_all_seeds
from visualize_mcts import write_mcts_as_png
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
plt.style.use('dark_background')
import numpy as np
import torch
import ray
import random
import os


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
      texture.blit(0, 0)
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

    print("\n\033[92mEvaluation finished! - label: ({})\033[0m".format(self.config.label))
    print("   Average length: {}({})".format(average_length, std_lengths))
    print("   Average return: {}({})".format(average_return, std_return))
    print("   Average predicted return: {}({})".format(average_pred_return, std_pred_return))
    print("   Average predicted value: {}({})".format(average_pred_value, std_pred_value))
    print("   Average mcts value: {}({})".format(average_root_value, std_root_value))
    print("   Average search depth: {}({})\n".format(average_search_depth, std_search_depth))

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

  def get_axes(self, config):
    labels = ['Return', 'Pred Return', 'Value', 
              'Pred Value', 'MCTS Value', 'Search Depth']
    if config.include_policy:
      labels.append('Policy')
    n_subplots = len(labels)
    axes = []
    for i, label in enumerate(labels):
      ax = plt.subplot2grid((n_subplots, 1), (i, 0))
      ax.set_ylabel(label, color='white', fontsize=13)
      ax.tick_params(colors='white', labelsize=9)
      ax.grid(alpha=0.3)
      axes.append(ax)
    axes[-1].set_xlabel('Steps', color='white', fontsize=13)
    plt.subplots_adjust(hspace=0.4)
    return axes

  def plot_summary(self, games, config, axes=None):
    axes = self.get_axes(config) if axes is None else axes

    rewards = [game.history.rewards for game in games]
    pred_rewards = [game.pred_rewards for game in games]

    returns = [np.cumsum(rews) for rews in rewards]
    pred_returns = [np.cumsum(rews) for rews in pred_rewards]

    discounts = [self.config.discount**n for n in range(max(map(len, rewards)))]
    values = []
    for rews in rewards:
      steps = len(rews)
      values.append([np.dot(rews[i:], discounts[:steps-i]) for i in range(steps)])
    pred_values = [game.pred_values for game in games]
    mcts_values = [game.history.root_values for game in games]
    search_depths = [game.search_depths for game in games]

    data = [returns, pred_returns, values, pred_values, mcts_values, search_depths]
    data_quantiles = [[self.config.label, self.get_quantiles(d, config.smooth)] for d in data]

    if config.include_policy:
      policy = list(zip(*[zip(*game.history.child_visits) for game in games]))
      for i, action in enumerate(policy):
        label = '{} - action: {}'.format(self.config.label, i)
        data_quantiles.append([label, self.get_quantiles(action, config.smooth)])
      axes.extend([axes[-1]] * len(policy))

    for ax, [label, quantiles] in zip(axes, data_quantiles):
      ax.plot(quantiles['0.5'], linewidth=2, label=label)
      if config.include_bounds:
        ax.fill_between(range(len(quantiles['0.5'])), y1=quantiles['0.25'], y2=quantiles['0.75'], alpha=0.4)
      ax.legend(framealpha=0.2, fontsize=8)
    return axes

class Evaluator(SummaryTools):

    def __init__(self, state):
        self.config = state['config']

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = get_network(self.config, self.device)
        self.network.load_state_dict(state['weights'])
        self.network.eval()

        self.mcts = MCTS(self.config)

        if self.config.render:
          self.viewer = ImageViewer()

    def play_game(self, environment, device):
      game = self.config.new_game(environment)

      if self.config.save_mcts:
        path_to_mcts_folder = os.path.split(os.path.normpath(self.config.saves_dir))[0]
        path_to_mcts_folder = os.path.join(path_to_mcts_folder, 'mcts')
        os.makedirs(path_to_mcts_folder, exist_ok=True)

      game.pred_values = []
      game.pred_rewards = []
      game.search_depths = []
      while not game.terminal and game.step < self.config.max_steps:
        root = Node(0)

        current_observation = self.config.to_torch(game.get_observation(-1), device).unsqueeze(0)
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
          if self.config.save_mcts:
            path_to_file = os.path.join(path_to_mcts_folder, str(game.step)+'.png')
            write_mcts_as_png(mcts, path_to_file=path_to_file)

          game.search_depths.append(max(map(len, mcts)) - 1)

          times_applied = 0
          node = root
          while node.expanded():
            action = self.config.select_action(root, temperature=self.config.temperature)
            reward = node.children[action].reward

            actions.append(action)
            rewards.append(reward)

            node = node.children[action]
            times_applied += 1

            if times_applied == self.config.apply_mcts_steps:
              break

        game.pred_values.append(initial_inference.value.item())
        game.store_search_statistics(root)

        for action, reward in zip(actions, rewards):
          game.apply(action)
          game.pred_rewards.append(reward)

          if self.config.render:
            try:
              img = environment.obs_buffer.max(axis=0)
              self.viewer.imshow(img)
            except:
              environment.render()

          if game.terminal or game.step > self.config.max_steps:
            break

      msg = "\033[92m[Game done]\033[0m --> "
      msg += "length: {}, return: {}, pred return: {}, pred value: {}, mcts value: {}"
      print(msg.format(game.step, np.round(game.sum_rewards, 1),
                       np.round(np.sum(game.pred_rewards), 1),
                       np.round(np.mean(game.pred_values), 1),
                       np.round(np.mean(game.history.root_values), 1)))
      return game

def get_label(state, config, idx):
    if config.detailed_label:
      label = 'path:{}'.format(idx)
      label += ', net:{}'.format(state['step'])
      if state['config'].only_value:
        label += ', only value'
      elif state['config'].only_prior:
        label += ', only prior'
      else:
        label += ', sims:{}'.format(state['config'].num_simulations)
        label += ', mcts-steps:{}'.format(state['config'].apply_mcts_steps)
        if state['config'].temperature:
          label += ', temp:{}'.format(state['config'].temperature)
        if state['config'].use_exploration_noise:
          label += ', with noise'.format(state['config'].use_exploration_noise)
    else:
      label = '{}'.format(state['step'])
    return label

def get_states(config):
  state_paths = [(saves_dir + net) for saves_dir in config.saves_dir for net in config.evaluate_nets]
  states = []
  for idx, saves_dir in enumerate(config.saves_dir):
    for net in config.evaluate_nets:
      meta_state = torch.load(saves_dir + net, map_location=torch.device('cpu'))
      for temperature in config.eval_temperatures:
        for num_simulations in config.num_simulations:
          for only_prior in config.only_prior:
            for only_value in config.only_value:
              for use_exploration_noise in config.use_exploration_noise:
                for apply_mcts_steps in config.apply_mcts_steps:
                  if only_prior is True and only_value is True:
                    continue
                  state = deepcopy(meta_state)
                  state['config'].saves_dir = saves_dir
                  state['config'].num_simulations = num_simulations
                  state['config'].temperature = temperature
                  state['config'].only_value = only_value
                  state['config'].only_prior = only_prior
                  state['config'].use_exploration_noise = use_exploration_noise
                  state['config'].apply_mcts_steps = apply_mcts_steps
                  state['config'].label = get_label(state, config, idx)
                  state['config'].render = config.render
                  state['config'].save_mcts = config.save_mcts
                  states.append(state)
  return states

def run(evaluator, seed=None):
    environment = get_environment(evaluator.config)
    if seed is not None:
      environment.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
      game = evaluator.play_game(environment, device)

    game.history._replace(observations = [])
    game.environment = None
    return game

@ray.remote
def run_parallel(evaluator, seed=None):
    game = run(evaluator, seed)
    return game


if __name__ == '__main__':
    from config import make_config
    import pyglet
    import cv2
    from pyglet.gl import *

    os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

    config = make_config()

    if config.parallel:
      ray.init()

    states = get_states(config)

    evaluators = []
    for state in states:
      evaluators.append(Evaluator(state))

    print("\n\033[92mStarting a {} episode evaluation of {} configurations\033[0m...".format(config.games_per_evaluation, len(states)))
    
    if config.plot_summary:
      axes = None

    base_seed = config.seed[0] if config.seed[0] is not None else 0
    seeds = range(base_seed, config.games_per_evaluation + base_seed)
    
    for evaluator in evaluators:
      print("\n\033[92mEvaluating... - label: ({})\033[0m".format(evaluator.config.label))

      if config.parallel:
        games = ray.get([run_parallel.remote(evaluator, seed) for seed in seeds])
      else:
        games = [run(evaluator, seed) for seed in seeds]

      evaluator.print_summary(games)

      if config.plot_summary:
        axes = evaluator.plot_summary(games, config, axes)

    if config.plot_summary:
      wm = plt.get_current_fig_manager()
      wm.window.state('zoomed')
      plt.show()
