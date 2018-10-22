import numpy as np


# implement a crop function that crops a torch.tensor or np.ndarray
# to a given shape. we will need this for several applications
def crop_tensor(input_, shape_to_crop):
    input_shape = input_.shape
    assert all(ish >= csh for ish, csh in zip(input_shape, shape_to_crop)),\
        "Input shape must be larger equal crop shape"
    # get the difference between the shapes
    shape_diff = tuple((ish - csh) // 2
                       for ish, csh in zip(input_shape, shape_to_crop))
    # calculate the crop
    crop = tuple(slice(sd, sh - sd)
                 for sd, sh in zip(shape_diff, input_shape))
    return input_[crop]


# TODO this might fail for odd shapes
def pad(input_, out_shape, mode='reflect'):
    shape = input_.shape
    assert len(shape) == len(out_shape)
    shape_diff = [(osh - sh) // 2 for sh, osh in zip(shape, out_shape)]
    pad_width = [2 * [sd] for sd in shape_diff]
    return np.pad(input_, pad_width, mode=mode)


def normalize(input_, mode='zero_mean_unit_variance'):
    if mode == 'zero_mean_unit_variance':
        mean = input_.mean()
        std = input_.std()
        input_ -= mean
        input_ /= std
    # TODO use percentile normalization here
    elif mode == "zero_to_one":
        input_ -= input_.min()
        input_ /= input_.max()
    return input_
