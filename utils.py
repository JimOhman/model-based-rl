from atari_wrappers import make_atari, wrap_atari
from torch.optim.lr_scheduler import ExponentialLR
from torch.optim import RMSprop, Adam, SGD
from torch.nn import MSELoss, LogSoftmax
from networks import Network

def get_environment(config):
    env = make_atari(config)
    env = wrap_atari(env, config)
    return env

def get_networks(config):
    env = get_environment(config)
    network = Network(action_space=env.action_space.n)
    return network

def get_loss_functions(config):
    def cross_entropy_loss(policy_logits, target_policy):
        loss = (-target_policy * LogSoftmax(dim=1)(policy_logits)).sum(1)
        return loss
    if config.scalar_loss == 'MSELoss':
        scalar_loss = MSELoss()
    else:
        raise NotImplementedError
    if config.policy_loss == 'CrossEntropyLoss':
        policy_loss = cross_entropy_loss
    else:
        raise NotImplementedError
    return scalar_loss, policy_loss

from torch.optim import RMSprop, Adam, SGD

def get_optimizer(config, parameters):
    if config.optimizer == 'RMSprop':
        optimizer = RMSprop(parameters, lr=config.lr_init, momentum=config.momentum, eps=0.01)
    elif config.optimizer == 'Adam':
        optimizer = Adam(parameters, lr=config.lr_init, weight_decay=config.weight_decay)
    elif config.optimizer == 'SGD':
        optimizer = SGD(parameters, lr=config.lr_init, momentum=config.momentum, weight_decay=config.weight_decay)
    else:
        raise NotImplementedError
    return optimizer

def get_lr_scheduler(config, optimizer):
    if config.lr_scheduler:
        if config.lr_scheduler == 'ExponentialLR':
            lr_scheduler = ExponentialLR(optimizer, config.lr_decay_rate)
        else:
            raise NotImplementedError
    else:
        lr_scheduler = None
    return lr_scheduler


MAXIMUM_FLOAT_VALUE = float('inf')

class MinMaxStats(object):

  def __init__(self):
    self.maximum = -MAXIMUM_FLOAT_VALUE
    self.minimum = MAXIMUM_FLOAT_VALUE

  def update(self, value: float):
    self.maximum = max(self.maximum, value)
    self.minimum = min(self.minimum, value)

  def normalize(self, value: float) -> float:
    if self.maximum > self.minimum:
      return (value - self.minimum) / (self.maximum - self.minimum)
    return value


class ActionHistory(object):

    def __init__(self, history):
        self.history = list(history)

    def clone(self):
        return ActionHistory(self.history)

    def add_action(self, action):
        self.history.append(action)

    def last_action(self):
        return self.history[-1]


class Node(object):

    def __init__(self, prior):
        self.visit_count = 0
        self.prior = prior
        self.value_sum = 0
        self.children = {}
        self.hidden_state = None
        self.reward = 0

    def expanded(self):
        return len(self.children) > 0

    def value(self):
        if self.visit_count == 0:
          return 0
        return self.value_sum / self.visit_count


class Game(object):

    def __init__(self, environment, discount):
        self.environment = environment
        observation = self.environment.reset()
        self.observations = []
        self.observations.append(observation)
        self.action_history = []
        self.rewards = []
        self.life_loss = []
        self.child_visits = []
        self.root_values = []
        self.discount = discount

    def terminal(self):
        return True if self.environment.was_real_done else False

    def apply(self, action):
        observation, reward, done, info = self.environment.step(action)
        self.observations.append(observation)
        self.rewards.append(reward)
        self.action_history.append(action)
        self.life_loss.append(done)
        if done:
          self.environment.reset()

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append([child.visit_count / sum_visits for child in root.children.values()])
        self.root_values.append(root.value())

    def make_image(self, state_index, device):
        obs = self.observations[state_index]
        image = torch.tensor(np.float32(obs), dtype=torch.float32, device=device) / 255.0
        return image

    def make_target(self, state_index, num_unroll_steps, td_steps):

        life_loss_indexes = self.life_loss[state_index:state_index + num_unroll_steps + 1]
        life_index = life_loss_indexes.index(True)

        targets = []
        for current_index in range(state_index, state_index + num_unroll_steps + 1):
            bootstrap_index = current_index + td_steps

            for i, reward in enumerate(self.rewards[current_index:bootstrap_index]):
              value = reward * self.discount**i

            if bootstrap_index < life_index:
              value += self.root_values[bootstrap_index] * self.discount**td_steps

            if current_index > 0 and current_index <= len(self.rewards):
                last_reward = self.rewards[current_index - 1]
            else:
                last_reward = 0

            if current_index < life_index:
              targets.append((value, last_reward, self.child_visits[current_index]))
            else:
              targets.append((0, 0, []))
        return targets