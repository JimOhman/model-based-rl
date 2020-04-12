from utils import get_network, get_optimizer, get_lr_scheduler, get_loss_functions, to_torch, to_numpy
from evaluate import Evaluator
from logger import Logger
import numpy as np
import torch
import time
import ray


@ray.remote(num_cpus=2)
class Learner(Evaluator, Logger):

    def __init__(self, run_tag, config, storage, replay_buffer):
        self.run_tag = run_tag
        self.worker_id = 'learner'
        self.replay_buffer = replay_buffer
        self.storage = storage
        self.config = config

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.network = get_network(config, self.device)
        self.optimizer = get_optimizer(config, self.network.parameters())
        self.lr_scheduler = get_lr_scheduler(config, self.optimizer)
        self.scalar_loss_fn, self.policy_loss_fn = get_loss_functions(config)
        self.to_torch = to_torch
        self.to_numpy = to_numpy

        self.training_step = 0
        self.sum_reward_loss = 0
        self.sum_value_loss = 0
        self.sum_policy_loss = 0
        self.sum_full_weighted_loss = 0

        if 'learner' in self.config.verbose:
          self.verbose = True
        else:
          self.verbose = False

        if 'learner' in self.config.log:
          self.log = True
        else:
          self.log = False

        if self.log:
          Logger.__init__(self)

        Evaluator.__init__(self)

    def train_network(self):
      self.send_weights()

      while ray.get(self.replay_buffer.get_size.remote()) < self.config.stored_before_train:
        time.sleep(1)

      for n in range(1, self.config.training_steps + 1):

        batch = ray.get(self.replay_buffer.sample_batch.remote())
        self.update_weights(batch)

        if n % self.config.checkpoint_interval == 0:
          self.send_weights()

        if n % self.config.save_state_frequency == 0:
          self.dump_state()

        if n % self.config.evaluation_frequency == 0:
          self.evaluate_network()

        self.storage.set_stats.remote({"training step": n})

      self.send_weights()

    def update_weights(self, batch):
      batch, idxs, is_weights = batch
      observations, actions, targets = batch
      observations = self.to_torch(observations, self.device)

      target_rewards, target_values, target_policies = zip(*targets)
      target_values = self.to_torch(target_values, self.device)
      target_rewards = self.to_torch(target_rewards, self.device)
      target_policies = self.to_torch(target_policies, self.device)

      value, reward, policy_logits, hidden_state = self.network.initial_inference(observations)

      new_errors = self.to_numpy(value.squeeze() - target_values[:, 0])
      self.replay_buffer.update.remote(idxs, new_errors)

      reward_loss = self.scalar_loss_fn(reward.squeeze(), target_rewards[:, 0])
      value_loss = self.scalar_loss_fn(value.squeeze(), target_values[:, 0])
      policy_loss = self.policy_loss_fn(policy_logits, target_policies[:, 0])

      for i, action in enumerate(zip(*actions), 1):
        value, reward, policy_logits, hidden_state = self.network.recurrent_inference(hidden_state, action)
        hidden_state.register_hook(lambda grad: grad * 0.5)

        value_loss += self.scalar_loss_fn(value.squeeze(), target_values[:, i])
        reward_loss += self.scalar_loss_fn(reward.squeeze(), target_rewards[:, i])
        policy_loss += self.policy_loss_fn(policy_logits, target_policies[:, i])

      is_weights = self.to_torch(is_weights, self.device)
      full_weighted_loss = (is_weights * (reward_loss + value_loss + policy_loss)).mean()

      gradient_scale = (1 / self.config.num_unroll_steps)
      full_weighted_loss.register_hook(lambda grad: grad * gradient_scale)

      self.optimizer.zero_grad()
      full_weighted_loss.backward()
      self.optimizer.step()
      if self.lr_scheduler is not None:
        self.lr_scheduler.step()
      self.training_step += 1

      self.sum_reward_loss += reward_loss.mean().item()
      self.sum_value_loss += value_loss.mean().item()
      self.sum_policy_loss += policy_loss.mean().item()
      self.sum_full_weighted_loss += full_weighted_loss.item()

      if self.verbose:
        print("training step: {}".format(self.training_step))
        print("value loss: {}".format(value_loss.mean()))
        print("policy loss: {}".format(policy_loss.mean()))
        print("reward loss: {}".format(reward_loss.mean()))
        print("full weighted loss: {}\n".format(full_weighted_loss))

      if self.log and self.training_step % self.config.learner_log_frequency == 0:
        self.log_scalar(tag='losses/reward_loss', value=(self.sum_reward_loss/self.config.learner_log_frequency), i=self.training_step)
        self.log_scalar(tag='losses/value_loss', value=(self.sum_value_loss/self.config.learner_log_frequency), i=self.training_step)
        self.log_scalar(tag='losses/policy_loss', value=(self.sum_policy_loss/self.config.learner_log_frequency), i=self.training_step)
        self.log_scalar(tag='losses/full_weighted_loss', value=(self.sum_full_weighted_loss/self.config.learner_log_frequency), i=self.training_step)
        self.sum_reward_loss = 0
        self.sum_value_loss = 0
        self.sum_policy_loss = 0
        self.sum_full_weighted_loss = 0

    def send_weights(self):
        self.storage.store_weights.remote(self.network.get_weights())

    def get_network(self):
      return self.network

    def launch(self):
      print("Learner is online.")
      self.train_network()
