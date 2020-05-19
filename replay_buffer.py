from collections import namedtuple
import numpy as np
import random
import ray


@ray.remote
class ReplayBuffer(object):

    def __init__(self, config):
        self.window_size = config.window_size
        self.batch_size = config.batch_size
        self.buffer = []

    def save_game(self, game):
        if len(self.buffer) > self.window_size:
          self.buffer.pop(0)
        self.buffer.append(game)

    def sample_batch(self, num_unroll_steps, td_steps):
        games = [self.sample_game() for _ in range(self.batch_size)]
        game_pos = [(g, self.sample_position(g)) for g in games]
        batch = []
        for g, i in game_pos:
          image = g.make_image(i)
          actions = g.action_history[i:i + num_unroll_steps]
          target = g.make_target(i, num_unroll_steps, td_steps)
          batch.append((image, actions, target))
        return batch

    def sample_game(self):
        return random.choice(self.buffer, 1)

    def sample_position(self, game):
        return random.choice(len(game.action_history), 1)

    def size(self):
        return len(self.buffer)


class SumTree(object):
    position = 0

    def __init__(self, capacity):
        self.tree = np.zeros(2 * capacity - 1)
        self.buffer = np.zeros(capacity, dtype=object)
        self.capacity = capacity
        self.num_memories = 0

    def add(self, priorities, history):
        for step, priority in enumerate(priorities):
          idx = self.position + self.capacity - 1
          self.buffer[self.position] = (step, history)
          self.update(idx, priority)

          self.position = (self.position + 1) % self.capacity

          if self.num_memories < self.capacity:
              self.num_memories += 1

    def update(self, idx, priority):
        change = priority - self.tree[idx]
        self.tree[idx] = priority

        while idx != 0:
            idx = (idx - 1) // 2
            self.tree[idx] += change

    def get_leaf(self, value):
        parent_index = 0

        while True:
            left_child_index = 2 * parent_index + 1
            right_child_index = left_child_index + 1

            if left_child_index >= len(self.tree):
                leaf_index = parent_index
                break
            else:
                if value <= self.tree[left_child_index]:
                    parent_index = left_child_index
                else:
                    value -= self.tree[left_child_index]
                    parent_index = right_child_index

        buffer_index = leaf_index - self.capacity + 1
        step, history = self.buffer[buffer_index]

        return leaf_index, self.tree[leaf_index], step, history

    @property
    def total_priority(self):
        return self.tree[0]

@ray.remote
class PrioritizedReplay():

    def __init__(self, config):
        self.window_size = config.window_size
        self.batch_size = config.batch_size

        self.beta_increment_per_sampling = config.beta_increment_per_sampling
        self.epsilon = config.epsilon
        self.alpha = config.alpha
        self.beta = config.beta

        self.num_unroll_steps = config.num_unroll_steps
        self.td_steps = config.td_steps
        self.discount = config.discount
        
        self.tree = SumTree(capacity=self.window_size)

        self.throughput = 0

    def get_priorities(self, errors):
        return np.power((np.abs(errors) + self.epsilon), self.alpha)

    def save_history(self, history, ignore=None):
        if ignore is not None:
          priorities = self.get_priorities(history.errors[:-ignore])
        else:
          priorities = self.get_priorities(history.errors)
        self.tree.add(priorities, history)

        self.throughput += len(priorities)

    def sample_batch(self):
        priorities = []
        batch = []
        batch_observations = []
        batch_actions = []
        batch_targets = []
        idxs = []
        steps = []

        if self.beta < 1:
            self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        priority_segment = self.tree.total_priority / self.batch_size
        for i in range(self.batch_size):
            s1, s2 = priority_segment * i, priority_segment * (i + 1)
            value = random.uniform(s1, s2)

            idx, priority, step, history = self.tree.get_leaf(value)

            priorities.append(priority)
            idxs.append(idx)
            steps.append(step)

            observation = history.observations[step]

            actions = history.actions[step:step + self.num_unroll_steps]
            num_actions = len(history.child_visits[0])
            while len(actions) < self.num_unroll_steps:
              actions.append(np.random.randint(num_actions))

            target = self.make_target(history, step, num_actions)
            
            batch_observations.append(observation)
            batch_actions.append(actions)
            batch_targets.append(target)

        batch = [batch_observations, batch_actions, batch_targets]

        sampling_probabilities = priorities / self.tree.total_priority
        is_weights = np.power(self.tree.num_memories * sampling_probabilities, -self.beta)
        is_weights = is_weights / is_weights.max()

        return batch, idxs, is_weights

    def make_target(self, history, step, num_actions):
        done_indexes = history.dones[step:step + self.num_unroll_steps + 1]

        end_index = len(history.root_values)

        target_rewards, target_values, target_policies = [], [], []
        for current_index in range(step, step + self.num_unroll_steps + 1):
            bootstrap_index = current_index + self.td_steps
            if bootstrap_index < end_index:
              value = history.root_values[bootstrap_index] * self.discount**self.td_steps
            else:
              value = 0

            for i, reward in enumerate(history.rewards[current_index:bootstrap_index]):
              value += reward * self.discount**i

            if current_index > 0 and current_index <= len(history.rewards):
                last_reward = history.rewards[current_index - 1]
            else:
                last_reward = 0

            if current_index < end_index:
              target_policies.append(history.child_visits[current_index])
              target_rewards.append(last_reward)
              target_values.append(value)
            else:
              dummy_policy = [0] * num_actions
              target_policies.append(dummy_policy)
              target_rewards.append(last_reward)
              target_values.append(0)

        return target_rewards, target_values, target_policies

    def update(self, idxs, errors):
      priorities = self.get_priorities(errors)
      for idx, priority in zip(idxs, priorities):
          self.tree.update(idx, priority)

    def size(self):
      return self.tree.num_memories

    def get_throughput(self):
      return self.throughput
