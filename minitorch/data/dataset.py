import numpy as np
import torch
import torch.utils.data as du


class DataseWithTransform(du.Dataset):
    """
    """

    def __init__(self, data, target, transform=None):
        assert isinstance(data, np.ndarray)
        assert isinstance(target, np.ndarray)
        assert len(data) == len(target)
        self.data = data
        self.target = target
        if transform is not None:
            assert callable(transform)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        data = self.data[index]
        target = self.target[index]
        if self.transform is not None:
            data, target = self.transform(data, target)
        return torch.from_numpy(data), torch.from_numpy(target)
