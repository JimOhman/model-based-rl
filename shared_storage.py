import ray

@ray.remote
class SharedStorage(object):

    def __init__(self, num_actors):
        self.weights = None
        self.stats = {'training step': 0}

    def latest_weights(self):
        return self.weights, self.stats['training step']

    def store_weights(self, weights, step):
        self.weights = weights
        self.stats['training step'] = step

    def get_stats(self, tag=None):
      return self.stats if tag is None else self.stats[tag]

    def set_stats(self, stats):
      for key, value in stats.items():
        self.stats[key] = value
