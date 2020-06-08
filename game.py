from typing import NamedTuple
import numpy as np


class History(NamedTuple):
  observations: list
  child_visits: list
  root_values: list
  actions: list
  rewards: list
  errors: list
  dones: list


class Game(object):

  def __init__(self, environment, config):
    self.episode_life = config.episode_life
    self.clip_rewards = config.clip_rewards
    self.environment = environment
    self.render = True if 'actors' in config.render else False

    self.history = History([], [], [], [], [], [], [])

    self.terminal, self.done = False, False

    self.previous_collect_to = 0
    self.sum_rewards = 0
    self.step = 0

  def apply(self, action):
    observation, reward, done, info = self.environment.step(action)
    self.terminal = self.environment.was_real_done if self.episode_life else done
    self.done = done

    self.sum_rewards += reward
    self.step += 1

    self.history.actions.append(action)
    self.history.dones.append(done)
    if self.clip_rewards:
      reward = np.sign(reward)
    self.history.rewards.append(reward)

    if done:
      observation = self.environment.reset()
      self.history.observations.append(observation)
    else:
      self.history.observations.append(observation)

    if self.render:
      self.environment.render()

  def store_search_statistics(self, root):
    sum_visits = sum(child.visit_count for child in root.children)
    self.history.child_visits.append([child.visit_count / sum_visits for child in root.children])
    self.history.root_values.append(root.value())

  def get_observation(self, state_index):
    if not self.history.observations:
      observation = self.environment.reset()
      self.history.observations.append(observation)
    return self.history.observations[state_index]

  def get_history_sequence(self, collect_from):
    history = History(*[data[collect_from:] for data in self.history])
    self.previous_collect_to = self.step
    return history
