import torch.nn as nn
from ..util import crop_tensor


# implement a wrapper around a loss function with pixel weights
# also crops target to prediction if specified
class WeightedLoss(nn.Module):
    """ Weighed loss
    """
    def __init__(self, loss_function,
                 weight_function=None, crop_target=True,
                 reduction='elementwise mean'):
        super().__init__()

        # TODO make sure that the reducntion of the loss
        # functioon is comparible with settings for weight
        # function and reduction
        self.loss_function = loss_function

        # the weighting function is optional and will not
        # be used if it is `None`
        # if a weighting function is given, it must
        # take the target tensor as input and return
        # a weight tensor with the same shape
        self.weight_function = weight_function

        # crop target
        self.crop_target = crop_target
        assert reduction in (None, 'none', 'sum', 'elementwise mean')
        self.reduction = reduction

    # to implement a loss function, we only need to
    # overload the forward pass.
    # the backward pass will be performed by torch automatically
    def forward(self, input, target):
        ishape = input.shape
        tshape = target.shape

        # make sure that the batches and channels target and input agree
        assert ishape[:2] == tshape[:2]
        assert ishape[1] == 1, "Only support a single channel for now"

        # crop the target to fit the input
        target = crop_tensor(target, ishape)

        # check if we have a weighting function and if so apply it
        if self.weight_function is not None:
            # apply the weight function
            weight = self.weight_function(target)
            # compute the loss WITHOUT reduction, which means that
            # the los will have the same shape as input and target
            # TODO TODO
            loss = self.loss_function(input, target, reduction='none')

            # multiply the loss by the weight and
            # reduce it via element-wise mean
            assert weight.shape == loss.shape, "Loss and weight must have the same shape"
            loss = torch.mean(loss * weight)

        # if we don't have a weighting function, just apply the loss
        else:
            loss = self.loss_function(input, target)

        # TODO reduction
        return loss
