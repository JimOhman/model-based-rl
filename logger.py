from torch.utils.tensorboard import SummaryWriter
import torch
import pytz
import datetime
import json
import os


class Logger(object):
    def __init__(self):
        self._init_directories()

        self.writer = SummaryWriter(self.directories['worker_id'])

        path = os.path.join(self.directories['config'], 'config.json')
        if not os.path.isfile(path):
            json.dump(self.config.__dict__, open(path, 'w'), indent=2)

    def log_scalar(self, value, tag, i):
        self.writer.add_scalar(tag, value, i)

    def dump_state(self):
        state = {'directories': self.directories, 'config': self.config, 'weights': self.network.get_weights()}
        torch.save(state, os.path.join(self.directories['saves'], str(self.training_step)))

    def _init_directories(self):
        run_directory = 'runs'
        if not os.path.exists(run_directory):
            os.mkdir(run_directory)

        environment_directory = os.path.join(run_directory, self.config.environment_id)
        if not os.path.exists(environment_directory):
            os.mkdir(environment_directory)

        if not self.run_tag:
            self.run_tag = datetime.datetime.now(tz=pytz.timezone('Europe/Stockholm')).strftime("%d-%b-%Y_%H:%M:%S")

        directories = {}
        directories['base'] = os.path.join(environment_directory, self.run_tag)
        directories['worker_id'] = os.path.join(directories['base'], self.worker_id)
        directories['saves'] = os.path.join(directories['base'], 'saves')
        directories['config'] = os.path.join(directories['base'], 'config')

        if not os.path.exists(directories['base']):
            os.mkdir(directories['base'])

        if not os.path.exists(directories['saves']):
            os.mkdir(directories['saves'])

        if not os.path.exists(directories['config']):
            os.mkdir(directories['config'])

        os.mkdir(directories['worker_id'])

        self.directories = directories
