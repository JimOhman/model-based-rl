from utils import get_network, get_optimizer, get_lr_scheduler, get_loss_functions, set_all_seeds
from mcts import MCTS, Node
from logger import Logger
from copy import deepcopy
import datetime
import torch
import time
import pytz
import ray


@ray.remote(num_cpus=1)
class Learner(Logger):

  def __init__(self, config, storage, replay_buffer, state=None, run_tag=None, group_tag=None, date=None):
    set_all_seeds(config.seed)

    self.run_tag = run_tag
    self.group_tag = group_tag
    self.worker_id = 'learner'
    self.replay_buffer = replay_buffer
    self.storage = storage
    self.config = deepcopy(config)

    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.network = get_network(config, self.device)

    self.optimizer = get_optimizer(config, self.network.parameters())
    self.lr_scheduler = get_lr_scheduler(config, self.optimizer)
    self.scalar_loss_fn, self.policy_loss_fn = get_loss_functions(config)

    self.training_step = 0
    self.losses_to_log = {'reward': 0., 'value': 0., 'policy': 0.}

    self.verbose = True if 'learner' in self.config.verbose else False
    self.log = True if 'learner' in self.config.log else False

    self.sleep = 0.

    if state is not None:
      self.load_state(state, date)

    Logger.__init__(self)

  def load_state(self, state, date):
    self.network.load_state_dict(state['weights'])
    self.optimizer.load_state_dict(state['optimizer'])
    self.training_step = state['step']
    old_run_tag = state['dirs']['base'].split('{}'.format(self.config.group_tag))[1][1:]
    self.run_tag = old_run_tag + '_resumed_' + date

  def send_weights(self):
    self.storage.store_weights.remote(self.network.get_weights(), self.training_step)

  def calculate_fps_throughput(self):
    time_now = time.time()
    throughput_fps = {}
    throughput_fps['actors'] = ray.get(self.replay_buffer.get_throughput.remote()) / (time_now - self.time_at_start)
    throughput_fps['learner'] = self.config.batch_size * self.training_step / (time_now - self.time_at_start)
    return throughput_fps

  def learn(self):
    self.send_weights()

    while ray.get(self.replay_buffer.size.remote()) < self.config.stored_before_train:
      time.sleep(1)

    self.network.train()
    self.time_at_start = time.time()
    while self.training_step < self.config.training_steps:
      not_ready = [self.replay_buffer.sample_batch.remote() for _ in range(self.config.batches_per_fetch)]
      while len(not_ready) > 0:
        ready, not_ready = ray.wait(not_ready, num_returns=1)

        batch = ray.get(ready[0])

        self.update_weights(batch)
        self.training_step += 1

        if self.training_step % self.config.checkpoint_frequency == 0:
          self.send_weights()

        if self.training_step % self.config.save_state_frequency == 0:
          self.save_state()

        if self.verbose or self.log and self.training_step % self.config.learner_log_frequency == 0:
          reward_loss = self.losses_to_log['reward'] / self.config.learner_log_frequency
          value_loss = self.losses_to_log['value'] / self.config.learner_log_frequency
          policy_loss = self.losses_to_log['policy'] / self.config.learner_log_frequency
          throughput = self.calculate_fps_throughput()

          if self.verbose:
            print("\ntraining step: {}".format(self.training_step))
            print("reward loss: {}".format(reward_loss))
            print("value loss: {}".format(value_loss))
            print("policy loss: {}".format(policy_loss))
            print("throughput: learner={}, actors={}".format(round(throughput['learner']), round(throughput['actors'])))

          if self.log and self.training_step % self.config.learner_log_frequency == 0:
            self.log_scalar(tag='loss/reward', value=reward_loss, i=self.training_step)
            self.log_scalar(tag='loss/value', value=value_loss, i=self.training_step)
            self.log_scalar(tag='loss/policy', value=policy_loss, i=self.training_step)
            self.log_scalar(tag='throughput/actors', value=throughput['actors'], i=self.training_step)
            self.log_scalar(tag='throughput/learner', value=throughput['learner'], i=self.training_step)
            self.log_scalar(tag='throughput/ratio', value=(throughput['learner'] / throughput['actors']), i=self.training_step)

            if self.config.debug:
              for name, weights in self.network.named_parameters():
                self.log_histogram(weights.grad.data.cpu().numpy(), 'gradients' + '/' + name + '_grad', self.training_step)
                self.log_histogram(weights.data.cpu().numpy(), 'network_weights' + '/' + name, self.training_step)

          self.losses_to_log['reward'] = 0
          self.losses_to_log['value'] = 0
          self.losses_to_log['policy'] = 0

        if self.config.sampling_ratio:
          throughput = self.calculate_fps_throughput()
          ratio = (throughput['learner'] / throughput['actors'])
          if ratio > self.config.sampling_ratio:
            self.sleep += 0.0001
          elif ratio < self.config.sampling_ratio:
            self.sleep = max(0, self.sleep - 0.0001)
          time.sleep(self.sleep)

    self.send_weights()

  def update_weights(self, batch):
    batch, idxs, is_weights = batch
    observations, actions, targets = batch

    target_rewards, target_values, target_policies = zip(*targets)

    observations = self.config.to_torch(observations, self.device, norm=self.config.norm_states)

    target_policies = self.config.to_torch(target_policies, self.device)
    target_values = self.config.to_torch(target_values, self.device)
    target_rewards = self.config.to_torch(target_rewards, self.device)

    value, reward, policy_logits, hidden_state = self.network.initial_inference(observations)

    init_value = self.config.inverse_value_transform(value) if not self.config.no_support else value
    new_errors = self.config.to_numpy(init_value.squeeze() - target_values[:, 0])
    self.replay_buffer.update.remote(idxs, new_errors)

    if not self.config.no_target_transform:
      target_values = self.config.scalar_transform(target_values)
      target_rewards = self.config.scalar_transform(target_rewards)

    if not self.config.no_support:
      target_values = self.config.value_phi(target_values)
      target_rewards = self.config.reward_phi(target_rewards)

    reward_loss = 0
    value_loss = self.scalar_loss_fn(value.squeeze(), target_values[:, 0])
    policy_loss = self.policy_loss_fn(policy_logits.squeeze(), target_policies[:, 0])
    hidden_state.register_hook(lambda grad: grad * 0.5)

    for i, action in enumerate(zip(*actions), 1):
      value, reward, policy_logits, hidden_state = self.network.recurrent_inference(hidden_state, action)
      hidden_state.register_hook(lambda grad: grad * 0.5)

      reward_loss += self.scalar_loss_fn(reward.squeeze(), target_rewards[:, i])

      value_loss += self.scalar_loss_fn(value.squeeze(), target_values[:, i])

      policy_loss += self.policy_loss_fn(policy_logits.squeeze(), target_policies[:, i])

    is_weights = self.config.to_torch(is_weights, self.device)
    reward_loss = (is_weights * reward_loss).mean()
    value_loss = (is_weights * value_loss).mean()
    policy_loss = (is_weights * policy_loss).mean()

    full_weighted_loss = reward_loss + value_loss + policy_loss

    full_weighted_loss.register_hook(lambda grad: grad * (1 / self.config.num_unroll_steps))

    self.optimizer.zero_grad()

    full_weighted_loss.backward()

    if self.config.clip_grad:
      torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.config.clip_grad)

    self.optimizer.step()

    if self.lr_scheduler is not None:
      self.lr_scheduler.step()

    if self.log:
      self.losses_to_log['reward'] += reward_loss.detach().cpu().item()
      self.losses_to_log['value'] += value_loss.detach().cpu().item()
      self.losses_to_log['policy'] += policy_loss.detach().cpu().item()

  def launch(self):
    print("Learner is online.")
    self.learn()
