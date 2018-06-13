import numpy as np
import time

class RandomModel:

    def __init__(self, config,**kargs):
        if kargs["seed"] >= 0:
            self.seed = int(kargs["seed"])
        else:
            self.seed = int(time.time())
        np.random.seed(seed=self.seed)
        self.dataset = None

    def fit(self, x, Y, dataset = None ,mask = None):
        self.dataset = dataset

    def predict(self, query, leave_one_out = None):
        return np.random.uniform(low=0.0, high=1.0, size=len(self.dataset.ds.candidates))


