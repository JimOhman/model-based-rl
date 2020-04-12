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
        self.render = config.render

        self.history = History([], [], [], [], [], [], [])

        self.terminal = False
        self.sum_rewards = 0
        self.step = 0

    def apply(self, action):
        observation, reward, done, info = self.environment.step(action)
        self.sum_rewards += reward
        self.history.observations.append(observation)
        self.history.actions.append(action)
        self.history.dones.append(done)
        if self.clip_rewards:
          reward = self.clip_reward(reward)
        self.history.rewards.append(reward)
        self.step += 1

        if done:
          if not self.episode_life:
            self.terminal = True
          else:
            self.terminal = self.environment.was_real_done
          observation = self.environment.reset()
        #   self.history.observations.append(observation)
        # else:
        #   self.history.observations.append(observation)

        if self.render:
          self.environment.render()

    def clip_reward(self, reward):
        return np.sign(reward)

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children)
        self.history.child_visits.append([child.visit_count / sum_visits for child in root.children])
        self.history.root_values.append(root.value())

    def get_observation(self, state_index):
        if not self.history.observations:
          observation = self.environment.reset()
          self.history.observations.append(observation)
        return self.history.observations[state_index]

    def get_history_sequence(self, length):
        history = [self.history[0][-length-1:]]
        for data in self.history[1:]:
          history.append(data[-length:])
        return History(*history)
