from mcts import MCTS, Node
from utils import get_network, get_environment, set_all_seeds
from visualize_mcts import write_mcts_as_png
from matplotlib import animation
from copy import deepcopy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("TkAgg")
plt.style.use('dark_background')
import pyglet
from pyglet.gl import *
import numpy as np
import torch
import time
import ray
import random
import sys
import os


class ImageViewer(object):

  def __init__(self, display=None, scale=3):
    self.window = None
    self.isopen = False
    self.display = display
    self.scale = scale

  def imshow(self, arr):
    if self.window is None:
      height, width, _channels = arr.shape
      height, width = self.scale*height, self.scale*width
      self.window = pyglet.window.Window(width=width, height=height, 
                                         display=self.display, vsync=False,
                                         resizable=True)            
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
    image = pyglet.image.ImageData(arr.shape[1], arr.shape[0], 'RGB',
                                   arr.tobytes(), pitch=arr.shape[1]*-3)
    gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_NEAREST)
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
    print("Average length: {:.1f}({:.1f})".format(average_length, std_lengths))
    print("Average return: {:.1f}({:.1f})".format(average_return, std_return))
    print("Average predicted return: {:.1f}({:.1f})".format(average_pred_return, std_pred_return))
    print("Average predicted value: {:.1f}({:.1f})".format(average_pred_value, std_pred_value))
    print("Average mcts value: {:.1f}({:.1f})".format(average_root_value, std_root_value))
    print("Average search depth: {:.1f}({:.1f})\n".format(average_search_depth, std_search_depth))

  def get_quantiles(self, values, smooth=None):
      max_len = len(max(values, key=len))
      extended = list(zip(*[list(vals) + [vals[-1]] * (max_len-len(vals)) for vals in values]))
      center = np.quantile(extended, 0.5, axis=1)
      upper = np.quantile(extended, 0.75, axis=1)
      lower = np.quantile(extended, 0.25, axis=1)
      if smooth is not None:
        center = self.moving_average(center, size=smooth)
        upper = self.moving_average(upper, size=smooth)
        lower = self.moving_average(lower, size=smooth)
      quantiles = {'0.5': center, '0.75': upper, '0.25': lower}
      return quantiles

  def get_axes(self, args):
    labels = ['Return', 'Pred Return', 'Value',
              'Pred Value', 'MCTS Value', 'Search Depth']
    if args.include_policy:
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

  def plot_summary(self, games, args, axes=None):
    axes = self.get_axes(args) if axes is None else axes

    if args.clip_rewards:
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

    label = self.config.label
    datas = [[label, self.get_quantiles(returns, args.smooth)],
             [label, self.get_quantiles(pred_returns, args.smooth)],
             [label, self.get_quantiles(values, args.smooth)],
             [label, self.get_quantiles(pred_values, args.smooth)],
             [label, self.get_quantiles(mcts_values, args.smooth)],
             [label, self.get_quantiles(search_depths, args.smooth)]]

    if args.include_policy:
      policy = list(zip(*[zip(*game.history.child_visits) for game in games]))
      for i, prob_action in enumerate(policy):
        label = '{} - action: {}'.format(label, i)
        datas.append([label, self.get_quantiles(prob_action, args.smooth)])
        axes.append(axes[-1])

    for ax, [label, quantiles] in zip(axes, datas):
      ax.plot(quantiles['0.5'], linewidth=2, label=label)
      if args.include_bounds:
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

    if self.config.use_gpu:
      if torch.cuda.is_available():
        self.device = torch.device("cuda")
      else:
        raise RuntimeError("GPU was requested but torch.cuda.is_available() is False.")
    else:
      self.device = torch.device("cpu")

    self.mcts = MCTS(self.config)
    self.network = None

    if self.config.render:
      self.viewer = ImageViewer()

    if self.config.norm_obs:
      self.obs_min = np.array(self.config.obs_range[::2], dtype=np.float32)
      self.obs_max = np.array(self.config.obs_range[1::2], dtype=np.float32)
      self.obs_range = self.obs_max - self.obs_min

  def load_network(self):
    self.network = get_network(self.config, self.device)
    self.network.load_state_dict(state["weights"])
    self.network.to(self.device)
    self.network.eval()

  def play_game(self, environment):
    assert self.network is not None, ".load_network() needs to be called before playing."

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
      if self.config.norm_obs:
        current_observation = (current_observation - self.obs_min) / self.obs_range
      current_observation = torch.from_numpy(current_observation).to(self.device)

      initial_inference = self.network.initial_inference(current_observation.unsqueeze(0))
      
      legal_actions = game.environment.legal_actions()
      root.expand(initial_inference, game.to_play, legal_actions)

      if self.config.use_exploration_noise:
        root.add_exploration_noise(self.config.root_dirichlet_alpha, self.config.root_exploration_fraction)

      actions_to_apply, corresponding_rewards = [], []
      if self.config.only_prior:
        _, action = max([(child.prior, action) for action, child in root.children.items()])
        reward = self.network.recurrent_inference(root.hidden_state, [action]).reward.item()
        actions_to_apply.append(action)
        corresponding_rewards.append(reward)
        root.children[action].visit_count += 1
        game.search_depths.append([0])

      elif self.config.only_value:
        q_values = []
        max_q_val = -np.inf
        for action in root.children.keys():
          output = self.network.recurrent_inference(root.hidden_state, [action])
          if self.config.two_players:
            q_val = (output.reward - self.config.discount * output.value).item()
          else:
            q_val = (output.reward + self.config.discount * output.value).item()
          if q_val > max_q_val:
            max_q_val = q_val
            chosen_action = action
            reward = output.reward.item()
          root.children[action].visit_count += 1

        actions_to_apply.append(chosen_action)
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
        actions_applied = 0
        while node.expanded():
          action = self.config.select_action(root, temperature=self.config.temperature)
          reward = node.children[action].reward
          actions_to_apply.append(action)
          corresponding_rewards.append(reward)

          actions_applied += 1
          if actions_applied == self.config.apply_mcts_actions:
            break

          node = node.children[action]

      game.pred_values.append(initial_inference.value.item())
      game.store_search_statistics(root)
      
      for action, reward in zip(actions_to_apply, corresponding_rewards):
        game.pred_rewards.append(reward)
        if self.config.two_players:
          if game.to_play == self.config.random_opp:
            action = np.random.choice(legal_actions)
          elif game.to_play == self.config.human_opp:
            print("waiting for your input: {}".format(legal_actions))
            action = int(input())
            while action not in legal_actions:
              print("invalid action, choose again!")
              action = int(input())
          to_play = game.to_play

        game.apply(action)

        if self.config.verbose:
          prior_policy = [round(child.prior, 2) for child in root.children.values()]
          sum_visits = sum(child.visit_count for child in root.children.values())
          mcts_policy = [round(child.visit_count/sum_visits, 2) for child in root.children.values()]
          print("\nstep {}".format(game.step))
          print("   legal actions: {}".format(list(legal_actions)))
          print("   prior policy:  {}".format(prior_policy))
          print("   mcts policy:   {}".format(mcts_policy))
          print("   prior value:    {}".format(round(game.pred_values[-1], 2)))
          print("   mcts value:    {}".format(round(root.value(), 2)))

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
          if self.config.two_players:
            if to_play in [self.config.random_opp, self.config.human_opp]:
              game.history.rewards[-1] *= -1
          break

    msg = "\033[92m[Game done]\033[0m --> "
    msg += "length: {:.1f}, return: {:.1f}, pred return: {:.1f}, pred value: {:.1f}, mcts value: {:.1f}"
    print(msg.format(game.step, np.sum(game.history.rewards), np.sum(game.pred_rewards),
                     np.mean(game.pred_values), np.mean(game.history.root_values)))

    if self.config.save_gif_as and frames:
      filename = self.config.save_gif_as + '.gif'
      self.save_frames_as_gif(frames, path_to_gif_folder, filename)

    return game

def get_label(state, detailed=False, path_idx=None):
  label_parts = ['net:{}'.format(state['training_step'])]
  if detailed:
    if path_idx is not None:
      label_parts.append('path:{}'.format(path_idx))
    if state['config'].only_value:
      label_parts.append('only value')
    elif state['config'].only_prior:
      label_parts.append('only prior')
    else:
      label_parts.append('sims:{}'.format(state['config'].num_simulations))
      if state['config'].apply_mcts_actions > 1:
        label_parts.append('mcts-actions:{}'.format(state['config'].apply_mcts_actions))
      if state['config'].temperature:
        label_parts.append('temp:{}'.format(state['config'].temperature))
      if state['config'].use_exploration_noise:
        label_parts.append('with noise'.format(state['config'].temperature))
  return ', '.join(label_parts)

def state_generator(args):
  for path_idx, saves_dir in enumerate(args.saves_dir):
    for net in args.nets:
      meta_state = torch.load(saves_dir + net, map_location=torch.device('cpu'))
      for temperature in args.temperatures:
        for num_simulations in args.num_simulations:
          for only_prior in args.only_prior:
            for only_value in args.only_value:
              for use_exploration_noise in args.use_exploration_noise:
                for apply_mcts_actions in args.apply_mcts_actions:
                  if not (only_prior and only_value):
                    state = deepcopy(meta_state)
                    state['config'].saves_dir = saves_dir

                    if num_simulations is not None:
                      state['config'].num_simulations = num_simulations
                    
                    state['config'].temperature = temperature
                    state['config'].only_value = only_value
                    state['config'].only_prior = only_prior
                    state['config'].use_exploration_noise = use_exploration_noise
                    state['config'].apply_mcts_actions = apply_mcts_actions
                    state['config'].render = args.render
                    state['config'].save_mcts = args.save_mcts
                    state['config'].save_mcts_after_step = args.save_mcts_after_step
                    state['config'].save_gif_as = args.save_gif_as
                    state['config'].sleep = args.sleep
                    state['config'].random_opp = args.random_opp
                    state['config'].human_opp = args.human_opp
                    state['config'].label = get_label(state, args.detailed_label, path_idx)
                    state['config'].use_gpu = args.use_gpu
                    state['config'].verbose = args.verbose

                    ### Back-comp
                    #state['config'].norm_obs = state['config'].norm_states
                    #state['config'].obs_range = state['config'].state_range
                    #state['config'].stack_obs = state['config'].stack_frames

                    yield state

def run(evaluator, seed=None):
  environment = get_environment(evaluator.config)
  if seed is not None:
    environment.seed(seed)
    set_all_seeds(seed)

  with torch.inference_mode():
    game = evaluator.play_game(environment)

  game.history.observations = []
  game.environment = None
  return game

@ray.remote
def run_parallel(evaluator, seed=None):
  return run(evaluator, seed=seed)


if __name__ == '__main__':
  from config import get_evaluation_args
  args = get_evaluation_args()

  if args.parallel:
    ray.init()

  evaluators = []
  for state in state_generator(args):
    evaluators.append(Evaluator(state))

  info = (args.num_games, len(evaluators))
  print("\n\033[92mStarting a {} episode evaluation of {} configurations\033[0m...".format(*info))
  
  if args.plot_summary:
    axes = None

  if args.seed is not None:
    seeds = range(args.seed, args.num_games + args.seed)
  else:
    seeds = [None] * args.num_games
  
  for evaluator in evaluators:
    info = (evaluator.config.label, evaluator.device)
    print("\n\033[92mEvaluating... - label: ({}) on {}\033[0m".format(*info))

    evaluator.load_network()
    if args.parallel:
      games = ray.get([run_parallel.remote(evaluator, seed) for seed in seeds])
    else:
      games = [run(evaluator, seed) for seed in seeds]

    evaluator.print_summary(games)

    if args.plot_summary:
      axes = evaluator.plot_summary(games, args, axes)

  if args.plot_summary:
    wm = plt.get_current_fig_manager()
    wm.window.state('zoomed')
    plt.show() 

