import numpy as np
import random
import ray


class SumTree(object):
    position = 0

    def __init__(self, capacity, capacity_step):
        self.tree = np.zeros(2 * capacity - 1)
        self.buffer = np.zeros(capacity, dtype=object)
        self.capacity = capacity
        self.capacity_step = capacity_step
        self.moving_capacity = capacity_step
        self.prev_moving_capacity = 0
        self.num_memories = 0

    def add(self, priorities, history):
        for step, priority in enumerate(priorities):
          idx = self.position + self.capacity - 1
          self.buffer[self.position] = (step, history)
          self.update(idx, priority)

          if self.position >= self.prev_moving_capacity:
            self.num_memories += 1

          self.position = (self.position + 1) % self.moving_capacity

          if self.position == 0:
            self.prev_moving_capacity = self.moving_capacity
            self.moving_capacity = min(self.capacity, (self.moving_capacity + self.capacity_step))

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
        self.window_step = config.window_step
        self.batch_size = config.batch_size

        self.beta_increment_per_sampling = config.beta_increment_per_sampling
        self.epsilon = config.epsilon
        self.alpha = config.alpha
        self.beta = config.beta

        self.num_unroll_steps = config.num_unroll_steps
        self.td_steps = config.td_steps
        self.discount = config.discount

        self.action_space = config.action_space
        self.obs_space = config.obs_space
        self.target_length = config.num_unroll_steps + 1

        self.absorbing_policy = np.zeros(self.action_space, dtype=np.float32)

        n_steps = self.num_unroll_steps + self.td_steps
        self.discounts = np.array([self.discount**n for n in range(n_steps)], dtype=np.float32)
        
        capacity = self.window_size
        if self.window_step is not None:
          capacity_step = self.window_step
        else:
          capacity_step = self.window_size

        self.tree = SumTree(capacity, capacity_step)

        self.throughput = {'frames': 0, 'games': 0}

        if config.seed is not None:
          np.random.seed(config.seed)
          random.seed(config.seed+1)

    def get_priorities(self, errors):
      return np.power((np.abs(errors) + self.epsilon), self.alpha)

    def save_history(self, history, ignore=None, terminal=False):
      if ignore is not None:
        errors = history.errors[:-ignore]
        priorities = self.get_priorities(errors) if errors else []
      else:
        priorities = self.get_priorities(history.errors)
      self.tree.add(priorities, history)

      self.throughput['frames'] += len(priorities)
      if terminal:
        self.throughput['games'] += 1

    def sample_batch(self):
        priorities = []
        batch_actions = []
        idxs = []

        batch_observations = np.zeros((self.batch_size, *self.obs_space), dtype=np.float32)
        target_policies = np.zeros((self.batch_size, self.target_length, self.action_space), dtype=np.float32)
        target_rewards = np.zeros((self.batch_size, self.target_length), dtype=np.float32)
        target_values = np.zeros((self.batch_size, self.target_length), dtype=np.float32)

        if self.beta < 1:
          self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        priority_segment = self.tree.total_priority / self.batch_size
        for batch_idx in range(self.batch_size):
          s1, s2 = priority_segment * batch_idx, priority_segment * (batch_idx + 1)
          value = random.uniform(s1, s2)

          idx, priority, step, history = self.tree.get_leaf(value)

          priorities.append(priority)
          idxs.append(idx)

          batch_observations[batch_idx, :] = np.float32(history.observations[step])

          actions = history.actions[step:step + self.num_unroll_steps]
          while len(actions) < self.num_unroll_steps:
            actions.append(np.random.randint(self.action_space))
          batch_actions.append(actions)

          self.insert_target(batch_idx, history, step, target_rewards,
                                                       target_values,
                                                       target_policies)

        batch = (batch_observations, batch_actions, (target_rewards, target_values, target_policies))

        sampling_probabilities = priorities / self.tree.total_priority
        is_weights = np.power(self.tree.num_memories*sampling_probabilities, -self.beta)
        is_weights /= is_weights.max()
        return batch, idxs, is_weights

    def get_max_history(self):
      fr = self.tree.capacity - 1
      to = fr + self.tree.num_memories
      tree_slice = self.tree.tree[fr:to]
      
      max_idx = np.argmax(tree_slice)

      leaf_index = max_idx + self.tree.capacity - 1
      priority = self.tree.tree[leaf_index]

      step, history = self.tree.buffer[max_idx]
      return leaf_index, priority, step, history

    def insert_target(self, batch_idx, history, step, target_rewards,
                                                      target_values,
                                                      target_policies):
        end_index = len(history.root_values)
        for idx, current_index in enumerate(range(step, step + self.num_unroll_steps + 1)):

          bootstrap_index = current_index + self.td_steps
          if bootstrap_index < end_index:
            value = history.root_values[bootstrap_index] * self.discount**self.td_steps
          else:
            value = 0

          rewards = history.rewards[current_index:bootstrap_index]
          if rewards:
            rewards = np.array(rewards, dtype=np.float32)
            value += np.dot(rewards, self.discounts[:len(rewards)])

          if current_index > 0 and current_index <= len(history.rewards):
            last_reward = history.rewards[current_index - 1]
          else:
            last_reward = 0

          if current_index < end_index:
            target_policies[batch_idx, idx, :] = history.child_visits[current_index]
            target_rewards[batch_idx, idx] = last_reward
            target_values[batch_idx, idx] = value
          else:
            target_policies[batch_idx, idx, :] = self.absorbing_policy
            target_rewards[batch_idx, idx] = last_reward
            target_values[batch_idx, idx] = 0

    def update(self, idxs, errors):
      priorities = self.get_priorities(errors)
      for idx, priority in zip(idxs, priorities):
        self.tree.update(idx, priority)

    def size(self):
      return self.tree.num_memories

    def get_throughput(self):
      return self.throughput
