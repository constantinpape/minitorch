import numpy as np


class Compose(object):
    def __init__(self, *transforms):
        assert all(callable(trafo) for trafo in transforms)
        self.transforms = transforms

    def __call__(self, data, target):
        for trafo in self.transforms:
            data, target = trafo(data, target)
        return data, target
