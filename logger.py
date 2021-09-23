from torch.utils.tensorboard import SummaryWriter
import torch
import pytz
import json
import os


class Logger(object):

  def __init__(self):

    self.dirs = self.make_dirs()
    self.save_config()

    self.writer = SummaryWriter(self.dirs['worker'])

  def log_scalar(self, value, tag, i):
    self.writer.add_scalar(tag, value, i)

  def log_scalars(self, value_dict, group_tag, i):
    self.writer.add_scalars(group_tag, value_dict, i)

  def log_histogram(self, values, tag, i):
    self.writer.add_histogram(tag, values, i)

  def log_image(self, image, tag):
    self.writer.add_image(tag, image)

  def save_config(self):
    path = os.path.join(self.dirs['config'], 'config.json')
    if not os.path.isfile(path):
      json.dump(self.config.__dict__, open(path, 'w'), indent=2)

  def make_dirs(self):
    base_dir = os.path.join('runs', self.config.environment)

    if self.group_tag is not None:
      base_dir = os.path.join(base_dir, self.group_tag)

    base_dir = os.path.join(base_dir, self.run_tag)

    dirs = {'base': base_dir,
            'worker': os.path.join(base_dir, self.worker_id),
            'saves': os.path.join(base_dir, 'saves'),
            'config': os.path.join(base_dir, 'config')}

    os.makedirs(dirs['saves'], exist_ok=True)
    os.makedirs(dirs['config'], exist_ok=True)
    os.makedirs(dirs['worker'], exist_ok=True)
    return dirs

