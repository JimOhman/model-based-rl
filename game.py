from typing import NamedTuple
import numpy as np


class HistorySlice(NamedTuple):
  observations: list
  child_visits: list
  root_values: list
  actions: list
  rewards: list
  errors: list
  dones: list


class History():

  def __init__(self, observations: list,
                     child_visits: list,
                     root_values: list,
                     actions: list,
                     rewards: list,
                     errors: list,
                     dones: list):
    self.observations = observations
    self.child_visits = child_visits
    self.root_values = root_values
    self.actions = actions
    self.rewards = rewards
    self.errors = errors
    self.dones = dones

  def get_slice(self, collect_from):
    return HistorySlice(self.observations[collect_from:],
                        self.child_visits[collect_from:],
                        self.root_values[collect_from:],
                        self.actions[collect_from:],
                        self.rewards[collect_from:],
                        self.errors[collect_from:],
                        self.dones[collect_from:])


class Game(object):

  def __init__(self, environment, config):
    self.episode_life = config.episode_life
    self.clip_rewards = config.clip_rewards
    self.sticky_actions = config.sticky_actions
    self.environment = environment

    self.history = History([], [], [], [], [], [], [])

    self.terminal, self.done = False, False

    self.previous_collect_to = 0
    self.sum_rewards = 0
    self.step = 0

  def apply(self, action):
    observation, reward, done, info = self.environment.step(action)

    if self.clip_rewards:
      self.sum_rewards += self.environment.original_reward
    else:
      self.sum_rewards += reward

    self.step += self.sticky_actions

    self.terminal = self.environment.was_real_done if self.episode_life else done

    if done:
      observation = self.environment.reset()

    self.history.observations.append(observation)

    if self.step >= self.environment._max_episode_steps:
      done = False

    self.history.actions.append(action)
    self.history.dones.append(done)
    self.history.rewards.append(reward)

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
    history = self.history.get_slice(collect_from)
    self.previous_collect_to = int(self.step / self.sticky_actions)
    return history
