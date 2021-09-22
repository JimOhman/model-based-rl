import numpy as np
import torch
import math


class MinMaxStats(object):

  def __init__(self, minimum_bound=None, maximum_bound=None):
    self.minimum = float('inf')  if minimum_bound is None else minimum_bound
    self.maximum = -float('inf') if maximum_bound is None else maximum_bound

  def update(self, value: float):
    self.minimum = min(self.minimum, value)
    self.maximum = max(self.maximum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      return (value - self.minimum) / (self.maximum - self.minimum)
    elif self.maximum == self.minimum:
      return 1.0
    return value

  def reset(self, minimum_bound=None, maximum_bound=None):
    self.minimum = float('inf')  if minimum_bound is None else minimum_bound
    self.maximum = -float('inf') if maximum_bound is None else maximum_bound


class Node(object):

  def __init__(self, prior):
    self.hidden_state = None
    self.visit_count = 0
    self.value_sum = 0
    self.reward = 0
    self.children = {}
    self.prior = prior
    self.to_play = 1

  def expanded(self):
    return len(self.children) > 0

  def value(self):
    if self.visit_count == 0:
      return 0
    return self.value_sum / self.visit_count

  def expand(self, network_output, to_play, actions):
    self.to_play = to_play
    self.hidden_state = network_output.hidden_state
    if network_output.reward:
      self.reward = network_output.reward.item()
    policy = {a: math.exp(network_output.policy_logits[0][a].item()) for a in actions}
    policy_sum = sum(policy.values())
    for action, p in policy.items():
      self.children[action] = Node(p / policy_sum)

  def add_exploration_noise(self, dirichlet_alpha, frac):
    actions = list(self.children.keys())
    noise = np.random.dirichlet([dirichlet_alpha] * len(actions))
    for a, n in zip(actions, noise):
      self.children[a].prior = self.children[a].prior*(1-frac) + n*frac


class MCTS(object):

  def __init__(self, config):
    self.num_simulations = config.num_simulations
    self.discount = config.discount
    self.pb_c_base = config.pb_c_base
    self.pb_c_init = config.pb_c_init
    self.init_value_score = config.init_value_score
    self.action_space = range(config.action_space)
    self.two_players = config.two_players
    self.known_bounds = config.known_bounds

    self.min_max_stats = MinMaxStats(*config.known_bounds)

  def run(self, root, network):
    self.min_max_stats.reset(*self.known_bounds)

    search_paths = []
    for _ in range(self.num_simulations):
      node = root
      search_path = [node]
      to_play = root.to_play

      while node.expanded():
        action, node = self.select_child(node)
        search_path.append(node)

        if self.two_players:
          to_play *= -1

      parent = search_path[-2]

      network_output = network.recurrent_inference(parent.hidden_state, [action])
      node.expand(network_output, to_play, self.action_space)

      self.backpropagate(search_path, network_output.value.item(), to_play)

      search_paths.append(search_path)
    return search_paths

  def select_child(self, node):
    if node.visit_count == 0:
      _, action, child = max(
          (child.prior, action, child)
          for action, child in node.children.items())
    else:
      _, action, child = max(
          (self.ucb_score(node, child), action, child)
          for action, child in node.children.items())
    return action, child

  def ucb_score(self, parent, child):
    pb_c = math.log((parent.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
    pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior
    if child.visit_count > 0:
      value = -child.value() if self.two_players else child.value()
      value_score = self.min_max_stats.normalize(child.reward + self.discount*value)
    else:
      value_score = self.init_value_score
    return prior_score + value_score

  def backpropagate(self, search_path, value, to_play):
    for idx, node in enumerate(reversed(search_path)):
      node.value_sum += value if node.to_play == to_play else -value
      node.visit_count += 1

      if self.two_players and node.to_play == to_play:
        reward = -node.reward
      else:
        reward = node.reward

      if idx < len(search_path) - 1:
        if self.two_players:
          new_q = node.reward - self.discount*node.value()
        else:
          new_q = node.reward + self.discount*node.value()
        self.min_max_stats.update(new_q)

      value = reward + self.discount*value


