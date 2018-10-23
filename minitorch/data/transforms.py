import numpy as np


class Compose(object):
    """
    """
    def __init__(self, *transforms):
        assert all(callable(trafo) for trafo in transforms)
        self.transforms = transforms

    def __call__(self, data, target):
        for trafo in self.transforms:
            data, target = trafo(data, target)
        return data, target


class Transform(object):
    """
    """
    dim = None
    appy_to_data = True
    apply_to_target = True

    # dummy random state for transforms without
    # random state
    def get_random_state(self):
        return None

    def _apply_transform(self, input_, state):
        # determine the dimension of the input
        indim = input_.ndim

        # if self.dim is None, the transform dimension is not specified and
        # can be applied to any input
        tdim = indim if self.dim is None else self.dim
        assert indim >= tdim and indim <= 5

        # apply
        if indim == tdim:
            return self.transform(input_, state)
        # apply
        elif indim == tdim + 1:
            return np.stack([self.transform(inp, state) for inp in input_])
        # apply
        elif indim == tdim + 2:
            # TODO check for correctness
            return np.array([[self.transform(inp, state) for inp in tensor]
                             for tensor in input_])
        # apply
        elif indim == tdim + 3:
            return

    def __call__(self, data, target, state=None):
        # update the random state
        if state is None:
            state = self.get_random_state()

        # apply trafos to data and target
        if self.appy_to_data:
            data = self._apply_transform(data, state)
        if self.apply_to_target:
            target = self._apply_transform(target, state)
        return data, target


class Rotate2d(Transform):
    """
    """
    dim = 2

    # the random state corresponds to the number
    # of rotations by 90 degree
    def get_random_state(self):
        return np.random.randint(0, 4)

    def transform(self, input_, state):
        return np.rot90(input_, k=state) if state > 0 else input_


class Flip2d(Transform):
    """
    """
    dim = 2

    def __init__(self, use_lr_flips=True, use_ud_flips=True):
        self.use_lr_flips = use_lr_flips
        self.use_ud_flips = use_ud_flips

    # the random states corresponds to
    # applying ud flips and lr flips
    def get_random_state(self):
        use_lr = np.random.random() > .5 and self.use_lr_flips
        use_ud = np.random.random() > .5 and self.use_ud_flips
        return use_lr, use_ud

    def transform(self, input_, state):
        use_lr, use_ud = state
        if use_lr:
            input_ = np.fliplr(input_)
        if use_ud:
            input_ = np.flipud(input_)
        return input_


class Noise(Transform):
    apply_to_target = False

    # TODO implement more noise types
    def __init__(self, noise_type='gaussian', mean=0., std=0.1):
        assert noise_type in ('gaussian',), noise_type
        self.noise_type = noise_type
        self.mean = mean
        self.std = std

    def transform(self, input_, state=None):
        if self.noise_type == 'gaussian':
            noise = np.random.normal(loc=self.mean, scale=self.std,
                                     size=input_.shape).astype(input_.dtype)
        return input_ + noise


class ElasticDeformation(Transform):
    def __init__(self):
        pass

    def get_random_state(self):
        pass

    def transform(self):
        pass
