import numpy as np
import torch
import math


class MinMaxStats(object):

  def __init__(self, maximum_bound=None, minimum_bound=None):
    self.maximum = -float('inf') if maximum_bound is None else maximum_bound
    self.minimum = float('inf')  if minimum_bound is None else minimum_bound

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value

  def reset(self, maximum_bound=None, minimum_bound=None):
    self.maximum = -float('inf') if maximum_bound is None else maximum_bound
    self.minimum = float('inf')  if minimum_bound is None else minimum_bound


class Node(object):

  def __init__(self, prior):
    self.hidden_state = None
    self.visit_count = 0
    self.value_sum = 0
    self.reward = 0
    self.children = []
    self.prior = prior

  def expanded(self):
    return len(self.children) > 0

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

  def expand(self, network_output):
    self.hidden_state = network_output.hidden_state
    if network_output.reward:
      self.reward = network_output.reward.item()
    policy = torch.softmax(network_output.policy_logits.squeeze(), dim=0)
    for action, p in enumerate(policy):
      self.children.append(Node(p.item()))

  def add_exploration_noise(self, dirichlet_alpha, exploration_fraction):
    noise = np.random.dirichlet([dirichlet_alpha] * len(self.children))
    for a, n in enumerate(noise):
      self.children[a].prior = self.children[a].prior*(1 - exploration_fraction) + n*exploration_fraction


class MCTS(object):

  def __init__(self, config):
    self.num_simulations = config.num_simulations
    self.discount = config.discount
    self.pb_c_base = config.pb_c_base
    self.pb_c_init = config.pb_c_init

    self.min_max_stats = MinMaxStats()

  def run(self, root, network):
    self.min_max_stats.reset()

    search_paths = []
    for _ in range(self.num_simulations):
      node = root
      search_path = [node]

      while node.expanded():
        action, node = self.select_child(node)
        search_path.append(node)

      parent = search_path[-2]
      last_action = [action]

      network_output = network.recurrent_inference(parent.hidden_state, last_action)
      node.expand(network_output)

      self.backpropagate(search_path, network_output.value.item())

      search_paths.append(search_path)
    return search_paths

  def select_child(self, node):
    if node.visit_count == 0:
      action = np.random.randint(len(node.children))
    else:
      action = np.argmax([self.ucb_score(node, child) for child in node.children])
    child = node.children[action]
    return action, child

  def ucb_score(self, parent, child):
    pb_c = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    if child.visit_count > 0:
      value_score = self.min_max_stats.normalize(child.reward + self.discount * child.value())
    else:
      value_score = 0.
    return prior_score + value_score

  def backpropagate(self, search_path, value):
    for idx, node in enumerate(reversed(search_path)):
      node.value_sum += value
      node.visit_count += 1
      self.min_max_stats.update(node.reward + self.discount * node.value())
      value = node.reward + self.discount * value
