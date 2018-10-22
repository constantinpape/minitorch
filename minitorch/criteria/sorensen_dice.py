import torch.nn as nn


# sorensen dice coefficient implemented in torch
# the coefficient takes values in [0, 1], where 0 is
# the worst score, 1 is the best score
class SorensenDice(nn.Module):
    def __init__(self, eps=1e-6, use_as_loss=False):
        super().__init__()
        self.eps = eps
        self.use_as_loss = use_as_loss

    # the dice coefficient of two sets represented as vectors a, b ca be
    # computed as (2 *|a b| / (a^2 + b^2))
    def forward(self, prediction, target):
        intersection = (prediction * target).sum()
        denominator = (prediction * prediction).sum() + (target * target).sum()
        score = (2 * intersection / denominator.clamp(min=self.eps))
        if self.use_as_loss:
            return 1. - score
        else:
            return score


# TODO generalized dice and tversky
