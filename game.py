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
  steps: list
  env_states: list


class History():

  def __init__(self, observations: list,
                     child_visits: list,
                     root_values: list,
                     actions: list,
                     rewards: list,
                     errors: list,
                     dones: list,
                     steps: list,
                     env_states: list):
    self.observations = observations
    self.child_visits = child_visits
    self.root_values = root_values
    self.actions = actions
    self.rewards = rewards
    self.errors = errors
    self.dones = dones
    self.steps = steps
    self.env_states = env_states

  def get_slice(self, collect_from):
    return HistorySlice(self.observations[collect_from:],
                        self.child_visits[collect_from:],
                        self.root_values[collect_from:],
                        self.actions[collect_from:],
                        self.rewards[collect_from:],
                        self.errors[collect_from:],
                        self.dones[collect_from:],
                        self.steps[collect_from:],
                        self.env_states[collect_from:])


class Game(object):

  def __init__(self, environment, config):
    self.episode_life = config.episode_life
    self.clip_rewards = config.clip_rewards
    self.sticky_actions = config.sticky_actions
    self.use_q_max = config.use_q_max
    self.revisit = config.revisit

    self.environment = environment

    self.history = History([], [], [], [], [], [], [], [], [])

    self.terminal, self.done = False, False
    self.previous_collect_to = 0
    self.history_idx = 0
    self.sum_rewards = 0
    self.sum_values = 0
    self.step = 0

    self.avoid_repeat = config.avoid_repeat
    self.non_zero_reward_step = 0

  def apply(self, action):

    self.history.steps.append(self.step)

    if self.revisit:
      env_state = self.environment.unwrapped.clone_state()
      self.history.env_states.append(env_state)

    if self.avoid_repeat:
      if (self.environment._elapsed_steps - self.non_zero_reward_step) > 500:
          if np.random.rand() < 0.05:
            action = np.random.randint(self.environment.action_space.n)

    observation, reward, done, info = self.environment.step(action)

    if reward != 0:
      self.non_zero_reward_step = self.environment._elapsed_steps

    if self.clip_rewards:
      self.sum_rewards += self.environment.last_reward
    else:
      self.sum_rewards += reward

    self.step = self.environment._elapsed_steps
    self.history_idx += 1

    self.terminal = self.environment.was_real_done if self.episode_life else done
    self.done = done

    if done:
      observation = self.environment.reset()

    self.history.observations.append(observation)
    self.history.actions.append(action)
    self.history.dones.append(done)
    self.history.rewards.append(reward)

  def store_search_statistics(self, root):
    sum_visits = sum(child.visit_count for child in root.children)
    self.history.child_visits.append([child.visit_count/sum_visits for child in root.children])
    if not self.use_q_max:
      value = root.value()
    else:
      value = max([child.reward + child.value() for child in root.children])
    self.history.root_values.append(value)
    self.sum_values += value

  def get_observation(self, state_index):
    if not self.history.observations:
      observation = self.environment.reset()
      self.history.observations.append(observation)
    return self.history.observations[state_index]

  def get_history_sequence(self, collect_from):
    history = self.history.get_slice(collect_from)
    self.previous_collect_to = self.history_idx
    return history
