import tensorboardX as tb
from ..util import crop_tensor


# TODO type annotations
# we adapt the tensorboard logger s.t. it can also log images
class TensorBoard(object):
    """ Tensorboard logger to use in `minitorch.util.main`.

    To modify the logging behaviour, overload `log_train` and `log_val`.
    """
    def __init__(self, log_dir, log_image_interval=20):
        self.log_dir = log_dir
        # we don't wan't to log images every iteration,
        # which is expensive, so we can specify `log_image_interval`
        self.log_image_interval = log_image_interval
        self.writer = tb.SummaryWriter(self.log_dir)

    def log_scalar(self, tag, value, step):
        self.writer.add_scalar(tag=tag, scalar_value=value, global_step=step)

    def log_single_image(self, tag, image, step):
        # change the image normalization for the tensorboard logger
        image -= image.min()
        max_val = image.max()
        if max_val > 0:
            image /= max_val
        self.writer.add_image(tag, img_tensor=image, global_step=step)

    def log_image(self, tag, image, step):
        # number of dimensions; need to substract batch and channel dim
        ndim = len(image.shape) - 2
        assert ndim in (2, 3), "can only log 2d or 3d images"
        n_channels = image.size(1)

        # get image(s) to cpu
        if ndim == 2:
            image = image[0, :].cpu().numpy()
        else:
            z = image.size(2) // 2
            image = image[0, :, z].cpu().numpy()

        # log image channels
        if n_channels == 1:
            self.log_single_image(tag, image, step)
        else:
            for c, im in enumerate(image):
                ctag = tag + '/channel%i' % c
                self.log_single_image(ctag, im, step)


    def log_train(self, step, loss, x, y, prediction):
        self.log_scalar(tag='train-loss', value=loss.item(),
                             step=step)
        # check if we log images in this iteration
        log_image_interval = self.log_image_interval
        if step % log_image_interval == 0:
            pshape = prediction.shape
            self.log_image(tag='train-input',
                           image=crop_tensor(x, pshape), step=step)
            self.log_image(tag='train-target',
                           image=crop_tensor(y, pshape), step=step)
            self.log_image(tag='train-prediction',
                           image=prediction.detach(), step=step)

    def log_val(self, step, loss, x, y, prediction, metric=None):
        self.log_scalar(tag='val-loss', value=loss, step=step)
        if metric is not None:
            self.log_scalar(tag='val-metric', value=metric, step=step)
        # we always log the last validation images
        pshape = prediction.shape
        self.log_image(tag='val-input',
                       image=crop_tensor(x, pshape), step=step)
        self.log_image(tag='val-target',
                       image=crop_tensor(y, pshape), step=step)
        self.log_image(tag='val-prediction',
                       image=prediction, step=step)
