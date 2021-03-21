from mcts import MCTS, Node
from utils import get_network, get_environment, set_all_seeds
from visualize_mcts import write_mcts_as_png
from matplotlib import animation
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
plt.style.use('dark_background')
import numpy as np
import torch
import time
import ray
import random
import sys
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
    average_length, std_lengths = np.mean(lengths), np.std(lengths)

    returns = [sum(game.history.rewards) for game in games]
    average_return, std_return = np.mean(returns), np.std(returns)

    pred_returns = [sum(game.pred_rewards) for game in games]
    average_pred_return, std_pred_return =np.mean(pred_returns), np.std(pred_returns)

    pred_values = [np.mean(game.pred_values) for game in games]
    average_pred_value, std_pred_value = np.mean(pred_values), np.std(pred_values)

    root_values = [np.mean(game.history.root_values) for game in games]
    average_root_value, std_root_value = np.mean(root_values), np.std(root_values)

    search_depths = [np.mean(max(game.search_depths)) for game in games]
    average_search_depth, std_search_depth = np.mean(search_depths), np.std(search_depths)

    print("\n\033[92mEvaluation finished! - label: ({})\033[0m".format(self.config.label))
    print("   Average length: {:.1f}({:.1f})".format(average_length, std_lengths))
    print("   Average return: {:.1f}({:.1f})".format(average_return, std_return))
    print("   Average predicted return: {:.1f}({:.1f})".format(average_pred_return, std_pred_return))
    print("   Average predicted value: {:.1f}({:.1f})".format(average_pred_value, std_pred_value))
    print("   Average mcts value: {:.1f}({:.1f})".format(average_root_value, std_root_value))
    print("   Average search depth: {:.1f}({:.1f})\n".format(average_search_depth, std_search_depth))

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
    axes = []
    for i, label in enumerate(labels):
      ax = plt.subplot2grid((len(labels), 1), (i, 0))
      ax.set_ylabel(label, color='white', fontsize=13)
      ax.tick_params(colors='white', labelsize=9)
      ax.grid(alpha=0.3)
      axes.append(ax)
    axes[-1].set_xlabel('Steps', color='white', fontsize=13)
    plt.subplots_adjust(hspace=0.4)
    return axes

  def get_values(self, dones, rewards):
    values = []
    longest_game = max(map(len, rewards))
    discounts = [self.config.discount**n for n in range(longest_game)]
    for life_loss, rews in zip(dones, rewards):
      life_loss_idxs, = np.where(np.array(life_loss)==True)
      vals = []
      if not len(life_loss_idxs):
        life_loss_idxs = [len(rews) - 1]
      step = 0
      life_loss_idx = life_loss_idxs[step]
      for i in range(len(rews)):
        if life_loss_idx < i:
          step = min(step + 1, len(life_loss_idxs) - 1)
        life_loss_idx = life_loss_idxs[step]
        r = rews[i:life_loss_idx + 1]
        d = discounts[:len(r)]
        vals.append(np.dot(r, d))
      values.append(vals)
    return values

  def plot_summary(self, games, config, axes=None):
    axes = self.get_axes(config) if axes is None else axes

    if config.clip_rewards:
      rewards = [np.sign(game.history.rewards) for game in games]
    else:
      rewards = [game.history.rewards for game in games]

    pred_rewards = [game.pred_rewards for game in games]

    dones = [game.history.dones for game in games]
    values = self.get_values(dones, rewards)

    returns = [np.cumsum(rews) for rews in rewards]
    pred_returns = [np.cumsum(rews) for rews in pred_rewards]

    pred_values = [game.pred_values for game in games]
    mcts_values = [game.history.root_values for game in games]
    search_depths = [np.max(game.search_depths, axis=1) for game in games]

    datas = [[self.config.label, self.get_quantiles(returns, config.smooth)],
             [self.config.label, self.get_quantiles(pred_returns, config.smooth)],
             [self.config.label, self.get_quantiles(values, config.smooth)],
             [self.config.label, self.get_quantiles(pred_values, config.smooth)],
             [self.config.label, self.get_quantiles(mcts_values, config.smooth)],
             [self.config.label, self.get_quantiles(search_depths, config.smooth)]]

    if config.include_policy:
      policy = list(zip(*[zip(*game.history.child_visits) for game in games]))
      for i, prob_action in enumerate(policy):
        label = '{} - action: {}'.format(self.config.label, i)
        datas.append([label, self.get_quantiles(prob_action, config.smooth)])
        axes.append(axes[-1])

    for ax, [label, quantiles] in zip(axes, datas):
      ax.plot(quantiles['0.5'], linewidth=2, label=label)
      if config.include_bounds:
        ax.fill_between(range(len(quantiles['0.5'])), y1=quantiles['0.25'], y2=quantiles['0.75'], alpha=0.4)
      ax.legend(framealpha=0.2, fontsize=8)
    return axes

  def save_frames_as_gif(self, frames, path='./', filename='default.gif'):
    plt.figure(figsize=(frames[0].shape[1]/72.0, frames[0].shape[0]/72.0), dpi=72)

    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
      patch.set_data(frames[i])

    path_to_file = os.path.join(path, filename)
    anim = animation.FuncAnimation(plt.gcf(), animate, frames=len(frames), interval=50)
    anim.save(path_to_file, writer='imagemagick', fps=60)
    plt.close()


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

        if self.config.norm_states:
          self.obs_min = np.array(self.config.state_range[::2], dtype=np.float32)
          self.obs_max = np.array(self.config.state_range[1::2], dtype=np.float32)
          self.obs_range = self.obs_max - self.obs_min

    def play_game(self, environment, device):
      game = self.config.new_game(environment)

      if self.config.save_mcts:
        path_to_mcts_folder = os.path.split(os.path.normpath(self.config.saves_dir))[0]
        path_to_mcts_folder = os.path.join(path_to_mcts_folder, 'mcts')
        os.makedirs(path_to_mcts_folder, exist_ok=True)

      if self.config.save_gif_as:
        path_to_gif_folder = os.path.split(os.path.normpath(self.config.saves_dir))[0]
        path_to_gif_folder = os.path.join(path_to_gif_folder, 'gifs')
        os.makedirs(path_to_gif_folder, exist_ok=True)

      frames = []
      game.pred_values = []
      game.pred_rewards = []
      game.search_depths = []
      while not game.terminal:
        root = Node(0)

        current_observation = np.float32(game.get_observation(-1))
        if self.config.norm_states:
          current_observation = (current_observation - self.obs_min) / self.obs_range
        current_observation = torch.from_numpy(current_observation).to(device)

        initial_inference = self.network.initial_inference(current_observation.unsqueeze(0))
        
        root.expand(network_output=initial_inference)

        if self.config.use_exploration_noise:
          root.add_exploration_noise(self.config.root_dirichlet_alpha, self.config.root_exploration_fraction)

        actions_to_apply, corresponding_rewards = [], []
        if self.config.only_prior:
          action = np.argmax([child.prior for child in root.children])
          reward = self.network.recurrent_inference(root.hidden_state, [action]).reward
          actions_to_apply.append(action)
          corresponding_rewards.append(reward)
          root.children[action].visit_count += 1
          game.search_depths.append([0])

        elif self.config.only_value:
          pred_rewards = []
          q_values = []

          for action in range(len(root.children)):
            output = self.network.recurrent_inference(root.hidden_state, [action])
            pred_rewards.append(output.reward)
            q_values.append(output.reward + self.config.discount * output.value)
            root.children[action].visit_count += 1

          action = np.argmax(q_values)
          reward = pred_rewards[action]
          actions_to_apply.append(action)
          corresponding_rewards.append(reward)
          game.search_depths.append([1])

        else:
          search_paths = self.mcts.run(root, self.network)
          search_depths = [len(search_path) for search_path in search_paths]
          game.search_depths.append(search_depths)

          if self.config.save_mcts and game.step >= self.config.save_mcts_after_step:
            path_to_file = os.path.join(path_to_mcts_folder, str(game.step) + '.png')
            write_mcts_as_png(search_paths, path_to_file=path_to_file)

          node = root
          steps_applied = 0
          while node.expanded():
            action = self.config.select_action(root, temperature=self.config.temperature)
            reward = node.children[action].reward
            actions_to_apply.append(action)
            corresponding_rewards.append(reward)

            node = node.children[action]
            steps_applied += 1

            if steps_applied == self.config.apply_mcts_steps:
              break

        game.pred_values.append(initial_inference.value.item())
        game.store_search_statistics(root)

        for action, reward in zip(actions_to_apply, corresponding_rewards):
          game.pred_rewards.append(reward)
          game.apply(action)

          if self.config.render:
            try:
              frame = game.environment.unwrapped._get_image()
              self.viewer.imshow(frame)
            except:
              frame = game.environment.render(mode='rgb_array')
            frames.append(frame)

            if self.config.sleep:
              time.sleep(self.config.sleep)

          if game.terminal or game.step >= self.config.max_steps:
            environment.was_real_done = True
            game.terminal = True
            break

      msg = "\033[92m[Game done]\033[0m --> "
      msg += "length: {:.1f}, return: {:.1f}, pred return: {:.1f}, pred value: {:.1f}, mcts value: {:.1f}"
      print(msg.format(game.step, np.sum(game.history.rewards), np.sum(game.pred_rewards),
                       np.mean(game.pred_values), np.mean(game.history.root_values)))

      if self.config.save_gif_as and frames:
        filename = self.config.save_gif_as + '.gif'
        self.save_frames_as_gif(frames, path_to_gif_folder, filename)
      return game

def get_label(state, config, idx):
    if config.detailed_label:
      label = 'path:{}, net:{}'.format(idx, state['step'])
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

def make_backwards_compatible(config):
  if not hasattr(config, 'avoid_repeat'):
    config.avoid_repeat = False

def state_generator(config):
  for idx, saves_dir in enumerate(config.saves_dir):
    for net in config.evaluate_nets:
      meta_state = torch.load(saves_dir+net, map_location=torch.device('cpu'))
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
                  if num_simulations is not None:
                    state['config'].num_simulations = num_simulations
                  state['config'].temperature = temperature
                  state['config'].only_value = only_value
                  state['config'].only_prior = only_prior
                  state['config'].use_exploration_noise = use_exploration_noise
                  state['config'].apply_mcts_steps = apply_mcts_steps
                  state['config'].render = config.render
                  state['config'].save_mcts = config.save_mcts
                  state['config'].save_mcts_after_step = config.save_mcts_after_step
                  state['config'].save_gif_as = config.save_gif_as
                  state['config'].sleep = config.sleep
                  state['config'].label = get_label(state, config, idx)

                  make_backwards_compatible(state['config'])

                  yield state

def run(evaluator, seed=None):
    environment = get_environment(evaluator.config)
    if seed is not None:
      environment.seed(seed)
      set_all_seeds(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
      game = evaluator.play_game(environment, device)

    game.history.observations = []
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

    import os
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    config = make_config()

    if config.parallel:
      ray.init()

    evaluators = []
    for state in state_generator(config):
      evaluators.append(Evaluator(state))

    info = (config.games_per_evaluation, len(evaluators))
    print("\n\033[92mStarting a {} episode evaluation of {} configurations\033[0m...".format(*info))
    
    if config.plot_summary:
      axes = None

    if config.seed[0] is not None:
      seeds = range(config.seed[0], config.games_per_evaluation + config.seed[0])
    else:
      seeds = [None] * config.games_per_evaluation
    
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
