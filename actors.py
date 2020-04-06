import ray
import torch
import math
from networks import Network
from utils import get_environment
import time


@ray.remote
class Actor(object):
    def __init__(self, actor_id, config, storage, replay_buffer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.id = actor_id
        self.storage = storage
        self.replay_buffer = replay_buffer
        self.config = config
        self.environment = get_environment(config)
        self.action_space = self.environment.action_space.n
        self.network = Network(self.config, self.device)

    def run_selfplay(self):

        while ray.get(self.storage.latest_weights.remote()) is None:
            time.sleep(1)

        while True:

            weights = ray.get(self.storage.latest_weights.remote())
            self.network.load_weights(weights)

            game = self.play_game()
            self.replay_buffer.save_game.remote(game)

    def play_game(self):
        game = self.config.new_game(self.environment)

        while not game.terminal() and len(game.action_history) < config.max_moves:

            current_observation = game.make_image(-1)
            initial_inference = self.network.initial_inference(current_observation)
            root = Node(0)
            self.expand_node(root, initial_inference)
            self.add_exploration_noise(root)

            self.run_mcts(root, game.action_history)
            action = self.select_action(root)
            game.apply(action)
            game.store_search_statistics(root)

        game.environment = None
        return game

    def run_mcts(self, root, action_history):
          min_max_stats = MinMaxStats(self.config.known_bounds)

          for _ in range(self.config.num_simulations):
            history = action_history.copy()
            node = root
            search_path = [node]

            while node.expanded():
              action, node = self.select_child(self.config, node, min_max_stats)
              history.add_action(action)
              search_path.append(node)

            # Inside the search tree we use the dynamics function to obtain the next
            # hidden state given an action and the previous hidden state.
            parent = search_path[-2]
            network_output = self.network.recurrent_inference(parent.hidden_state, history.last_action())
            self.expand_node(node, network_output)

            self.backpropagate(search_path, network_output.value, self.config.discount, min_max_stats)

    def expand_node(self, node, network_output):
          node.hidden_state = network_output.hidden_state
          node.reward = network_output.reward
          policy = {a: math.exp(network_output.policy_logits[a]) for a in range(self.action_space)}
          policy_sum = sum(policy.values())
          for action, p in policy.items():
            node.children[action] = Node(p / policy_sum)

    def backpropagate(search_path, value, discount, min_max_stats):
          for node in reversed(search_path):
            node.value_sum += value
            node.visit_count += 1
            min_max_stats.update(node.value())

            value = node.reward + discount * value

    def ucb_score(self, parent, child, min_max_stats) -> float:
          pb_c = math.log((parent.visit_count + self.config.pb_c_base + 1) / self.config.pb_c_base) + self.config.pb_c_init
          pb_c *= math.sqrt(parent.visit_count) / (child.visit_count + 1)

          prior_score = pb_c * child.prior
          value_score = min_max_stats.normalize(child.value())
          return prior_score + value_score

    def select_child(self, node, min_max_stats):
          _, action, child = max((self.ucb_score(node, child, min_max_stats), action, child) for action, child in node.children.items())
          return action, child

    def select_action(self, node):
          visit_counts, actions = list(zip(*[(child.visit_count, action) for child in node.children.items()]))
          temperature = self.config.visit_softmax_temperature_fn(training_steps=self.network.training_steps)
          action = self.action_sample(visit_counts, actions, temperature)
          return action

    def action_sample(visit_counts, actions, temperature):
        distribution = np.array([visit_count**(1 / temperature) for visit_count in visit_counts])
        distribution = distribution / sum(distribution)
        action = np.random.choice(actions, p=distribution)
        return action

    def add_exploration_noise(self, node):
        actions = list(node.children.keys())
        noise = np.random.dirichlet([self.config.root_dirichlet_alpha] * len(actions))
        frac = self.config.root_exploration_fraction
        for a, n in zip(actions, noise):
            node.children[a].prior = node.children[a].prior * (1 - frac) + n * frac

    def launch(self):
        print("Actor {} is online.".format(self.id))
        self.run_selfplay()