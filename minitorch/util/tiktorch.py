import os
import inspect
from shutil import copyfile

import yaml
import torch


def _to_dynamic_shape(minimal_increment):
    assert isinstance(minimal_increment, tuple)
    if len(minimal_increment) == 2:
        dynamic_shape = '(%i * (nH + 1), %i * (nW + 1))' % minimal_increment
    elif len(minimal_increment) == 3:
        dynamic_shape = '(%i * (nD + 1), %i * (nH + 1), %i * (nW + 1))' % minimal_increment
    else:
        raise ValueError("Invald length %i for minimal increment" % len(minimal_increment))
    return dynamic_shape


def checkpoint_to_tiktorch(model, model_kwargs,
                           checkpoint_folder, output_folder,
                           input_shape, minimal_increment,
                           load_best=True, description=None,
                           data_source=None):
    """ Save checkpoint in tiktorch format:
    TODO link

    Arguments:
        model:
        model_kwargs:
        checkpoint_folder:
        output_folder:
        input_shape:
        minimal_increment:
        load_best:
        description:
        data_source:
    """
    os.makedirs(output_folder, exists_ok=True)

    # get the path to code and class name
    code_path = inspect.getfile(model)
    cls_name = model.__name__

    # build the model, check the input and get output shape
    model = model(**model_kwargs)
    weight_path = os.path.join(checkpoint_folder,
                               'best_weights.torch' if load_best else 'weights.torch')
    assert os.path.exists(weight_path), weight_path
    model.load_state_dict(torch.load(weight_path))

    input_ = torch.zeros(*input_shape, dtype=torch.float32)
    out = model(input_)
    output_shape = tuple(out.shape)

    # build the config
    config = {'input_shape': input_shape,
              'output_shape': output_shape,
              'dynamic_input_shape': _to_dynamic_shape(minimal_increment),
              'model_class_name': cls_name,
              'model_init_kwargs': model_kwargs,
              'torch_version': torch.__version__}
    if description is not None:
        assert isinstance(description, str)
        config['description'] = description
    if data_source is not None:
        assert isinstance(data_source, str)
        config['data_source'] = data_source

    # serialize config
    config_file = os.path.join(checkpoint_folder, 'tiktorch_config.yml')
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    # copy the state-dict and the code path
    copyfile(weight_path, os.path.join(output_folder, 'state.nn'))
    copyfile(code_path, os.path.join(output_folder, 'model.py'))
