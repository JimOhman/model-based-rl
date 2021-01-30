from utils import get_network, get_environment, set_all_seeds
from config import make_config
from mcts import MCTS, Node
from copy import deepcopy
from main import get_run_tag, config_generator
from game import Game
from logger import Logger
from datetime import datetime
import torch
import pytz
import time
import pickle
import os


class HumanActor():

  def __init__(self, config):
    self.config = config

    if config.load_buffer:
      self.load_buffer()
    else:
      self.replay_buffer = []

    self.environment = get_environment(config)

    if config.seed is not None:
      self.environment.seed(config.seed)

  def play_game(self):
    global human_agent_action, human_wants_restart, human_sets_pause

    ACTIONS = self.environment.action_space.n
    SKIP_CONTROL = 0

    human_agent_action = 0
    human_wants_restart = False
    human_sets_pause = False

    def key_press(key, mod):
      global human_agent_action, human_wants_restart, human_sets_pause
      if key==0xff0d: human_wants_restart = True
      if key==32: human_sets_pause = not human_sets_pause
      a = int( key - ord('0') )
      if a <= 0 or a >= ACTIONS: return
      human_agent_action = a

    def key_release(key, mod):
      global human_agent_action
      a = int( key - ord('0') )
      if a <= 0 or a >= ACTIONS: return
      if human_agent_action == a:
        human_agent_action = 0

    self.environment.reset()
    self.environment.render()
    self.environment.unwrapped.viewer.window.on_key_press = key_press
    self.environment.unwrapped.viewer.window.on_key_release = key_release

    histories = []
    play = True
    while play:
      game = Game(self.environment, self.config)

      initial_observation = self.environment.reset()
      game.history.observations.append(initial_observation)

      self.environment.render()
            
      skip = 0
      while not game.terminal:

        if not skip:
          action = human_agent_action
          skip = SKIP_CONTROL
        else:
          skip -= 1

        game.apply(action)

        game.history.child_visits.append(None)
        game.history.root_values.append(None)
        game.history.errors.append(None)

        self.environment.render()
        time.sleep(self.config.speed)

        if game.done:
          print("length: {}, reward: {:.2f}".format(game.step, game.sum_rewards))
          save_history = input('Game returned done, do you want to store the history? (Y/n): ')
          if save_history.capitalize() != 'N':
            collect_from = game.previous_collect_to
            history = game.get_history_sequence(collect_from)
            self.replay_buffer.append(history)

      keep_playing = input('Play another? (Y/n): ')
      if keep_playing.capitalize() != 'N':
        play = True
      else:
        play = False

    if self.config.save_buffer and self.replay_buffer:
      self.save_buffer()

  def save_buffer(self):
    base_path = os.path.join('runs', config.environment, 'stored_games')
    if self.config.tag is None:
      if self.config.load_buffer is None:
        date = datetime.now(tz=pytz.timezone('Europe/Stockholm'))
        date = date.strftime("%d-%b-%Y_%H-%M-%S")
        save_folder_path = os.path.join(base_path, date)
      else:
        save_folder_path = os.path.split(self.config.load_buffer)[0]
    else:
      save_folder_path = os.path.join(base_path, config.tag)

    os.makedirs(save_folder_path, exist_ok=True)
    save_path = os.path.join(save_folder_path, 'games')

    with open('{}.pkl'.format(save_path), 'wb') as output:
      pickle.dump(self.replay_buffer, output, pickle.HIGHEST_PROTOCOL)

    num_experiences = sum(len(x.rewards) for x in self.replay_buffer)
    print("\nThe replay buffer is saved with {} experiences.".format(num_experiences))

  def load_buffer(self):
    with open('{}.pkl'.format(self.config.load_buffer), 'rb') as input:
      self.replay_buffer = pickle.load(input)


if __name__ == '__main__':
  import argparse
  args = argparse.ArgumentParser()

  args.add_argument('--environment', type=str, default='')
  args.add_argument('--load_buffer', type=str, default=None)
  args.add_argument('--save_buffer', action='store_true')
  args.add_argument('--tag', type=str, default=None)
  args.add_argument('--seed', type=str, default=None)
  args.add_argument('--speed', type=float, default=0.02)

  args.add_argument('--wrap_atari', action='store_true')
  args.add_argument('--stack_frames', type=int, default=1)
  args.add_argument('--episode_life', action='store_true')
  args.add_argument('--clip_rewards', action='store_true')

  config = args.parse_args()

  human_actor = HumanActor(config)
  human_actor.play_game()
