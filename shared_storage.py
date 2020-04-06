import ray

@ray.remote
class SharedStorage(object):

    def __init__(self):
        self._weights = {}

    def latest_weights(self):
        if self._weights:
          return self._weights[max(self._weights.keys())]
        else:
          return None

    def store_weights(self, step, weights):
        self._weights[step] = weights