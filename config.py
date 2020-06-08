import argparse
from game import Game
import numpy as np
import torch


class Config(object):

  def __init__(self, args):
      self.__dict__.update(args)

      self.value_support_min, self.value_support_max = self.value_support
      self.reward_support_min, self.reward_support_max = self.reward_support

      self.value_support_range = list(range(self.value_support_min, self.value_support_max + 1))
      self.value_support_size = len(self.value_support_range)

      self.reward_support_range = list(range(self.reward_support_min, self.reward_support_max + 1))
      self.reward_support_size = len(self.reward_support_range)

  def inverse_reward_transform(self, reward_logits):
      return self.inverse_transform(reward_logits, self.reward_support_range)

  def inverse_value_transform(self, value_logits):
      return self.inverse_transform(value_logits, self.value_support_range)

  def inverse_transform(self, logits, scalar_support_range):
      probabilities = torch.softmax(logits, dim=1)
      support = torch.tensor(scalar_support_range, dtype=torch.float, device=probabilities.device).expand(probabilities.shape)
      value = torch.sum(support * probabilities, dim=1, keepdim=True)
      if not self.no_target_transform:
        value = torch.sign(value) * (((torch.sqrt(1 + 4 * 0.001 * (torch.abs(value) + 1 + 0.001)) - 1) / (2 * 0.001)) ** 2 - 1)
      return value

  def value_phi(self, x):
      return self.scalar_to_support(x, self.value_support_min, self.value_support_max, self.value_support_size)

  def reward_phi(self, x):
      return self.scalar_to_support(x, self.reward_support_min, self.reward_support_max, self.reward_support_size)

  def visit_softmax_temperature(self, training_step):
      step1, step2 = self.visit_softmax_steps
      temp1, temp2, temp3 = self.visit_softmax_temperatures
      if training_step < step1:
        return temp1
      elif training_step < step2:
        return temp2
      else:
        return temp3

  def to_torch(self, x, device, scale=False):
    x = np.float32(x)
    if scale and self.scale_state is not None:
      x = (x - self.scale_state[0]) / (self.scale_state[1] - self.scale_state[0])
    return torch.from_numpy(x).to(device)

  @staticmethod
  def scalar_transform(x):
      output = torch.sign(x) * (torch.sqrt(torch.abs(x) + 1) - 1) + 0.001 * x
      return output

  @staticmethod
  def scalar_to_support(x, min, max, support_size):
      x.clamp_(min, max)
      x_low = x.floor()
      x_high = x.ceil()
      p_high = (x - x_low)
      p_low = 1 - p_high

      support = torch.zeros(x.shape[0], x.shape[1], support_size).to(x.device)
      x_high_idx, x_low_idx = x_high - min, x_low - min
      support.scatter_(2, x_high_idx.long().unsqueeze(-1), p_high.unsqueeze(-1))
      support.scatter_(2, x_low_idx.long().unsqueeze(-1), p_low.unsqueeze(-1))
      return support

  @staticmethod
  def to_numpy(x):
    return x.detach().cpu().numpy()

  @staticmethod
  def select_action(node, temperature=0):
      visit_counts = np.array([child.visit_count for child in node.children])
      if temperature:
        distribution = np.array([visit_count**(1 / temperature) for visit_count in visit_counts])
        distribution = distribution / sum(distribution)
        action = np.random.choice(len(visit_counts), p=distribution)
      else:
        action = np.random.choice(np.where(visit_counts == visit_counts.max())[0])
      return action

  def new_game(self, environment):
      return Game(environment, self)


def make_config():
  parser = argparse.ArgumentParser()

  ### Network
  network = parser.add_argument_group('network')
  network.add_argument('--architecture', type=str, default='TinierNetwork')
  network.add_argument('--value_support', nargs='+', type=int, default=[-25, 25])
  network.add_argument('--reward_support', nargs='+', type=int, default=[-5, 5])
  network.add_argument('--no_support', action='store_true')
  network.add_argument('--seed', nargs='+', type=int, default=[None])

  ## Myopic
  myopic = parser.add_argument_group('myopic')
  myopic.add_argument('--focal_size', nargs='+', type=int, default=[20, 20])
  myopic.add_argument('--focal_step', type=int, default=2)
  myopic.add_argument('--blurr', type=int, default=0)

  ### Environment
  environment = parser.add_argument_group('environment')
  environment.add_argument('--environment', type=str, default='BreakoutNoFrameskip-v4') # BreakoutNoFrameskip-v4 SpaceInvadersNoFrameskip-v4

  # Environment Modifications
  environment_modifications = parser.add_argument_group('general environment modifications')
  environment_modifications.add_argument('--scale_state', nargs='+', type=int, default=None)
  environment_modifications.add_argument('--clip_rewards', action='store_true')
  environment_modifications.add_argument('--stack_frames', type=int, default=1)

  atari = parser.add_argument_group('atari environment modifications')
  atari.add_argument('--wrap_atari', action='store_true')
  atari.add_argument('--episode_life', action='store_true')
  atari.add_argument('--stack_actions', action='store_true')
  atari.add_argument('--frame_size', nargs='+', type=int, default=[96, 96])
  atari.add_argument('--noop_max', type=int, default=30)
  atari.add_argument('--frame_skip', type=int, default=4)

  ### Self-Play
  self_play = parser.add_argument_group('self play')
  self_play.add_argument('--num_actors', nargs='+', type=int, default=[7])
  self_play.add_argument('--max_steps', type=int, default=27000)
  self_play.add_argument('--num_simulations', nargs='+', type=int, default=[30])
  self_play.add_argument('--max_sequence_length', type=int, default=200)
  self_play.add_argument('--visit_softmax_temperatures', nargs='+', type=float, default=[1.0, 0.5, 0.25])
  self_play.add_argument('--visit_softmax_steps', nargs='+', type=int, default=[50e3, 100e3])
  self_play.add_argument('--fixed_temperatures', nargs='+', type=float, default=[])

  # Root prior exploration noise.
  exploration = parser.add_argument_group('exploration')
  exploration.add_argument('--root_dirichlet_alpha', type=float, default=0.25)
  exploration.add_argument('--root_exploration_fraction', type=float, default=0.25)

  # UCB formula
  ucb = parser.add_argument_group('UCB formula')
  ucb.add_argument('--pb_c_base', type=int, default=19652)
  ucb.add_argument('--pb_c_init', type=float, default=1.25)

  ### Prioritized Replay Buffer
  per = parser.add_argument_group('prioritized experience replay')
  per.add_argument('--window_size', nargs='+', type=int, default=[50000])
  per.add_argument('--epsilon', type=float, default=0.01)
  per.add_argument('--alpha', type=float, default=1.)
  per.add_argument('--beta', type=float, default=1.)
  per.add_argument('--beta_increment_per_sampling', type=float, default=0.001)

  ### Training
  training = parser.add_argument_group('training')
  training.add_argument('--training_steps', type=int, default=1000000)
  training.add_argument('--policy_loss', type=str, default='CrossEntropyLoss')
  training.add_argument('--scalar_loss', type=str, default='MSE')
  training.add_argument('--num_unroll_steps', nargs='+', type=int, default=[5])
  training.add_argument('--checkpoint_frequency', type=int, default=100)
  training.add_argument('--td_steps', nargs='+', type=int, default=10)
  training.add_argument('--batch_size', nargs='+', type=int, default=[64])
  training.add_argument('--batches_per_fetch', type=int, default=15)
  training.add_argument('--stored_before_train', type=int, default=5000)
  training.add_argument('--clip_grad', type=int, default=0)
  training.add_argument('--no_target_transform', action='store_true')
  training.add_argument('--sampling_ratio', type=float, default=0.)
  training.add_argument('--discount', nargs='+', type=float, default=[0.997])

  # Optimizer
  training.add_argument('--optimizer', type=str, default='Adam')
  training.add_argument('--momentum', type=float, default=0.9)
  training.add_argument('--weight_decay', type=float, default=1e-4)

  # Learning rate scheduler
  training.add_argument('--lr_scheduler', type=str, default='')
  training.add_argument('--lr_init', nargs='+', type=float, default=[0.0005])
  training.add_argument('--lr_decay_rate', type=float, default=0.1)
  training.add_argument('--lr_decay_steps', type=int, default=350000)

  ### Saving and Loading
  load_and_save = parser.add_argument_group('saving and loading')
  load_and_save.add_argument('--save_state_frequency', type=int, default=1000)
  load_and_save.add_argument('--load_state', type=str, default='')
  load_and_save.add_argument('--override_loaded_config', action='store_true')

  ### Evalutation
  evaluation = parser.add_argument_group('evaluation')
  evaluation.add_argument('--games_per_evaluation', type=int, default=1)
  evaluation.add_argument('--eval_temperatures', nargs='+', type=float, default=[0])
  evaluation.add_argument('--only_prior', nargs='+', type=int, default=[0])
  evaluation.add_argument('--only_value', nargs='+', type=int, default=[0])
  evaluation.add_argument('--use_exploration_noise', nargs='+', type=int, default=[0])
  evaluation.add_argument('--saves_dir', nargs='+', type=str, default=[''])
  evaluation.add_argument('--evaluate_nets', nargs='+', type=str, default=[''])
  evaluation.add_argument('--plot_summary', action='store_true')
  evaluation.add_argument('--include_bounds', action='store_true')
  evaluation.add_argument('--include_policy', action='store_true')
  evaluation.add_argument('--detailed_eval_tag', action='store_true')
  evaluation.add_argument('--smooth', type=int, default=0)
  evaluation.add_argument('--plot_aggregate', action='store_true')
  evaluation.add_argument('--apply_mcts_steps', nargs='+', type=int, default=[1])

  ### Logging
  logging = parser.add_argument_group('logging')
  logging.add_argument('--run_tag', type=str, default=None)
  logging.add_argument('--group_tag', type=str, default='default')
  logging.add_argument('--log', nargs='+', type=str, default='')
  logging.add_argument('--actor_log_frequency', type=int, default=1)
  logging.add_argument('--learner_log_frequency', type=int, default=100)

  ### Debugging
  debug = parser.add_argument_group('debugging')
  debug.add_argument('--debug', action='store_true')
  debug.add_argument('--render', nargs='+', type=str, default='')
  debug.add_argument('--verbose', nargs='+', type=str, default='')
  debug.add_argument('--print_network_summary', action='store_true')
  debug.add_argument('--save_mcts_to_path', type=str, default='')

  args = vars(parser.parse_args())

  return Config(args=args)
