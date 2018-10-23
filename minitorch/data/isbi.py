import os

import numpy as np
import h5py
import imageio

from ..util import pad, normalize
from .dataset import DataseWithTransform


# TODO implement download
# TODO support tif version of the dataset
class Isbi2012(DataseWithTransform):
    """
    """
    split_slice = 27
    xy_shape = (572, 572)

    def _preprocess_data(self, data):
        # first, normalize the data
        data = normalize(data)
        # then pad the data
        data = pad(data, (data.shape[0],) + self.xy_shape)
        # finally add channel dimension
        return data[:, None]

    def _preprocess_target(self, target):
        # first, map the target to [0, 1]
        target /= target.max()
        # then inver the target to have the membranes as foreground
        target = 1. - target
        # pad the target
        target = pad(target, (target.shape[0],) + self.xy_shape,
                     mode='constant')
        # add channel dimension
        return target[:, None]

    def __init__(self, root, train=True, transform=None):
        assert os.path.exists(root), root

        slice_ = slice(0, self.split_slice) if train else slice(self.split_slice, 30)

        with h5py.File(root, 'r') as f:
            data = f['volumes/raw'][slice_]
            target = f['volumes/labels/membranes'][slice_]
        super().__init__(self._preprocess_data(data),
                         self._preprocess_target(target),
                         transform)
