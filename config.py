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
      if training_step <= step1:
        return temp1
      elif training_step <= step2:
        return temp2
      else:
        return temp3

  @staticmethod
  def scalar_transform(x):
      output = torch.sign(x)*(torch.sqrt(torch.abs(x) + 1) - 1) + 0.001*x
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
  def select_action(node, temperature=0.):
      actions = list(node.children.keys())
      visit_counts = np.array([child.visit_count for child in node.children.values()])
      if temperature:
        distribution = visit_counts ** (1/temperature)
        distribution = distribution / distribution.sum()
        idx = np.random.choice(len(actions), p=distribution)
      else:
        idx = np.random.choice(np.where(visit_counts == visit_counts.max())[0])
      action = actions[idx]
      return action

  def new_game(self, environment):
      return Game(environment, self)


def make_config():
  parser = argparse.ArgumentParser()

  ### Network
  network = parser.add_argument_group('network')
  network.add_argument('--architecture', choices=['FCNetwork', 'MuZeroNetwork', 'TinyNetwork'], type=str, default='FCNetwork')
  network.add_argument('--value_support', nargs=2, type=int, default=[-15, 15])
  network.add_argument('--reward_support', nargs=2, type=int, default=[-15, 15])
  network.add_argument('--no_support', action='store_true')
  network.add_argument('--seed', nargs='+', type=int, default=[None])

  ### Environment
  environment = parser.add_argument_group('environment')
  environment.add_argument('--environment', type=str, default='LunarLander-v2')
  environment.add_argument('--two_players', action='store_true')

  # Environment Modifications
  environment_modifications = parser.add_argument_group('general environment modifications')
  environment_modifications.add_argument('--clip_rewards', action='store_true')
  environment_modifications.add_argument('--stack_frames', type=int, default=1)
  environment_modifications.add_argument('--state_range', nargs='+', type=float, default=None)
  environment_modifications.add_argument('--norm_states', action='store_true')
  environment_modifications.add_argument('--max_episode_steps', type=int, default=None)
  environment_modifications.add_argument('--sticky_actions', type=int, default=1)
  environment_modifications.add_argument('--episode_life', action='store_true')
  environment_modifications.add_argument('--fire_reset', action='store_true')
  environment_modifications.add_argument('--noop_reset', action='store_true')
  environment_modifications.add_argument('--noop_max', type=int, default=30)
  environment_modifications.add_argument('--avoid_repeat', action='store_true')

  atari = parser.add_argument_group('atari environment modifications')
  atari.add_argument('--wrap_atari', action='store_true')
  atari.add_argument('--stack_actions', action='store_true')
  atari.add_argument('--frame_size', nargs='+', type=int, default=[96, 96])
  atari.add_argument('--frame_skip', type=int, default=4)

  ### Self-Play
  self_play = parser.add_argument_group('self play')
  self_play.add_argument('--num_actors', nargs='+', type=int, default=[7])
  self_play.add_argument('--max_steps', type=int, default=40000)
  self_play.add_argument('--num_simulations', nargs='+', type=int, default=[30])
  self_play.add_argument('--max_history_length', type=int, default=500)
  self_play.add_argument('--visit_softmax_temperatures', nargs=2, type=float, default=[1.0, 0.5, 0.25])
  self_play.add_argument('--visit_softmax_steps', nargs=2, type=int, default=[15e3, 30e3])
  self_play.add_argument('--fixed_temperatures', nargs='+', type=float, default=[])

  # MCTS exploration terms
  exploration = parser.add_argument_group('exploration')
  exploration.add_argument('--root_dirichlet_alpha', type=float, default=0.25)
  exploration.add_argument('--root_exploration_fraction', type=float, default=0.25)
  exploration.add_argument('--init_value_score', type=float, default=0.0)
  exploration.add_argument('--known_bounds', nargs=2, type=float, default=[None, None])

  # UCB formula
  ucb = parser.add_argument_group('UCB formula')
  ucb.add_argument('--pb_c_base', type=int, default=19652)
  ucb.add_argument('--pb_c_init', type=float, default=1.25)

  ### Prioritized Replay Buffer
  per = parser.add_argument_group('prioritized experience replay')
  per.add_argument('--window_size', nargs='+', type=int, default=[100000])
  per.add_argument('--window_step', nargs='+', type=int, default=[None])
  per.add_argument('--epsilon', type=float, default=0.01)
  per.add_argument('--alpha', type=float, default=1.)
  per.add_argument('--beta', type=float, default=1.)
  per.add_argument('--beta_increment_per_sampling', type=float, default=0.001)

  ### Training
  training = parser.add_argument_group('training')
  training.add_argument('--training_steps', type=int, default=100000000)
  training.add_argument('--policy_loss', type=str, default='CrossEntropyLoss')
  training.add_argument('--scalar_loss', type=str, default='MSE')
  training.add_argument('--num_unroll_steps', nargs='+', type=int, default=[5])
  training.add_argument('--send_weights_frequency', type=int, default=500)
  training.add_argument('--weight_sync_frequency', type=int, default=1000)
  training.add_argument('--td_steps', nargs='+', type=int, default=[10])
  training.add_argument('--batch_size', nargs='+', type=int, default=[256])
  training.add_argument('--batches_per_fetch', type=int, default=15)
  training.add_argument('--stored_before_train', type=int, default=50000)
  training.add_argument('--clip_grad', type=int, default=0)
  training.add_argument('--no_target_transform', action='store_true')
  training.add_argument('--sampling_ratio', type=float, default=0.)
  training.add_argument('--discount', nargs='+', type=float, default=[0.997])
  training.add_argument('--revisit_frequency', type=float, default=np.float('inf'))
  training.add_argument('--revisit', action='store_true')
  training.add_argument('--use_q_max', action='store_true')

  # Optimizer
  training.add_argument('--optimizer', choices=['RMSprop', 'Adam', 'AdamW', 'SGD'], type=str, default='AdamW')
  training.add_argument('--momentum', type=float, default=0.9)
  training.add_argument('--weight_decay', type=float, default=1e-4)

  # Learning rate scheduler
  training.add_argument('--lr_scheduler', type=str, default='')
  training.add_argument('--lr_init', nargs='+', type=float, default=[0.0008])
  training.add_argument('--lr_decay_rate', type=float, default=0.1)
  training.add_argument('--lr_decay_steps', type=int, default=100000)

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
  evaluation.add_argument('--detailed_label', action='store_true')
  evaluation.add_argument('--smooth', type=int, default=0)
  evaluation.add_argument('--apply_mcts_steps', nargs='+', type=int, default=[1])
  evaluation.add_argument('--parallel', action='store_true')
  evaluation.add_argument('--save_gif_as', type=str, default='')
  evaluation.add_argument('--sleep', type=float, default=0)
  evaluation.add_argument('--save_mcts', action='store_true')
  evaluation.add_argument('--save_mcts_after_step', type=int, default=0)
  evaluation.add_argument('--render', action='store_true')
  evaluation.add_argument('--human_opp', type=int, choices=[0, 1], default=None)
  evaluation.add_argument('--random_opp', type=int, choices=[0, 1], default=None)

  ### Logging
  logging = parser.add_argument_group('logging')
  logging.add_argument('--run_tag', type=str, default=None)
  logging.add_argument('--group_tag', type=str, default='default')
  logging.add_argument('--log', nargs=2, choices=['actors', 'learner'], type=str, default='')
  logging.add_argument('--actor_log_frequency', type=int, default=1)
  logging.add_argument('--learner_log_frequency', type=int, default=100)

  ### Debugging
  debug = parser.add_argument_group('debugging')
  debug.add_argument('--debug', action='store_true')
  debug.add_argument('--verbose', nargs=2, choices=['actors', 'learner'], type=str, default='')

  args = parser.parse_args()

  if any(np.array(args.window_size) < args.stored_before_train):
    err_msg = '--window_size must be larger than --stored_before_train.'
    parser.error(err_msg)

  if args.fixed_temperatures:
    for num_actors in args.num_actors:
      if len(args.fixed_temperatures) != num_actors:
        err_msg = 'if fixed temperatures is used a temperature for each actor must be specified.'
        parser.error(err_msg)

  return Config(args=vars(args))
