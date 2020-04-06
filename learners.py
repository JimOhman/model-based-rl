import ray
import torch
from networks import Network
from utils import get_optimizer, get_lr_scheduler, get_loss_functions, get_environment


@ray.remote
class Learner(object):
    def __init__(self, config, storage, replay_buffer):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.storage = storage
        self.replay_buffer = replay_buffer
        self.config = config
        env = get_environment(config)
        self.network = Network(env.action_space.n, self.device)
        self.optimizer = get_optimizer(config, self.network.parameters())
        self.lr_scheduler = get_lr_scheduler(config, self.optimizer)
        self.scalar_loss, self.policy_loss = get_loss_functions(config)

    def train_network(self, storage, replay_buffer):
      self.store_weights(0)

      while self.replay_buffer.get_num_games.remote() == 0:
        time.sleep(1)

      for i in range(self.config.training_steps):

        if i % self.config.checkpoint_interval == 0:
          self.store_weights(i)

        batch = self.replay_buffer.sample_batch.remote(self.config.num_unroll_steps, self.config.td_steps)
        self.update_weights(batch)
        
        self.network.training_steps += 1

      self.store_weights(self.config.training_steps)

    def store_weights(self, i):
        weights = self.network.get_weights()
        self.storage.store_weights.remote(i, weights)

    def lr_step(self):
        lr = self.config.lr_init * self.config.lr_decay_rate ** (self.network.training_step / self.config.lr_decay_steps)
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def update_weights(self, batch):
      loss = 0
      self.optimizer.zero_grad()
      for image, actions, targets in batch:
        value, reward, policy_logits, hidden_state = self.network.initial_inference(image)

        target_value, target_reward, target_policy = targets[0]

        value_loss = self.scalar_loss(value, target_value)
        value_loss.register_hook(lambda grad: grad * gradient_scale)

        policy_loss = self.policy_loss(policy_logits, target_policy)
        policy_loss.register_hook(lambda grad: grad * gradient_scale)

        predictions = []

        for action in actions:
          value, reward, policy_logits, hidden_state = self.network.recurrent_inference(hidden_state, action)
          predictions.append((1.0 / len(actions), value, reward, policy_logits))

          hidden_state.register_hook(lambda grad: grad * 0.5)

        for prediction, target in zip(predictions, targets[1:]):
          gradient_scale, value, reward, policy_logits = prediction
          target_value, target_reward, target_policy = target

          value_loss = self.scalar_loss(value, target_value)
          value_loss.register_hook(lambda grad: grad * gradient_scale)

          reward_loss = self.scalar_loss(reward, target_reward)
          reward_loss.register_hook(lambda grad: grad * gradient_scale)

          policy_loss = self.policy_loss(policy_logits, target_policy)
          policy_loss.register_hook(lambda grad: grad * gradient_scale)

          loss += (value_loss + reward_loss + policy_loss)

      loss.backward()
      self.optimizer.step()
      self.lr_step()

    def launch(self):
      print("Learner is online.")
      # self.train_network()
