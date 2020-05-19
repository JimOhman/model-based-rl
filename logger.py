from torch.utils.tensorboard import SummaryWriter
import torch
import pytz
import datetime
import json
import os


class Logger(object):

    def __init__(self, dirs=None):
        if dirs is None:
            self.dirs = self.get_dirs()
            self.save_config()
        else:
            self.dirs = dirs
            self.dirs['worker'] = '/'.join(self.dirs['worker'].split('/')[:-1]) + self.worker_id

        self.writer = SummaryWriter(self.dirs['worker'])

    def log_scalar(self, value, tag, i):
        self.writer.add_scalar(tag, value, i)

    def log_scalars(self, value_dict, group_tag, i):
        self.writer.add_scalars(group_tag, value_dict, i)

    def log_histogram(self, values, tag, i):
        self.writer.add_histogram(tag, values, i)

    def save_state(self):
        state = {'dirs': self.dirs, 'step': self.training_step, 'config': self.config, 'weights': self.network.get_weights()}
        torch.save(state, os.path.join(self.dirs['saves'], str(self.training_step)))

    def save_config(self):
        path = os.path.join(self.dirs['config'], 'config.json')
        if not os.path.isfile(path):
            json.dump(self.config.__dict__, open(path, 'w'), indent=2)

    def get_dirs(self):
        env_dir = os.path.join('runs', self.config.environment)

        dirs = {}
        dirs['base'] = os.path.join(env_dir, self.run_tag)
        dirs['worker'] = os.path.join(dirs['base'], self.worker_id)
        dirs['saves'] = os.path.join(dirs['base'], 'saves')
        dirs['config'] = os.path.join(dirs['base'], 'config')

        os.makedirs(dirs['saves'], exist_ok=True)
        os.makedirs(dirs['config'], exist_ok=True)
        os.makedirs(dirs['worker'], exist_ok=True)

        return dirs
