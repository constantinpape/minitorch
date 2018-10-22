import torch
import torch.nn as nn
from ..util import crop_tensor


class UNetBase(nn.Module):
    """ UNet Base class implementation

    Deriving classes must implement
    - _conv_block(in_channels, out_channels, level, part)
    - _upsampler(in_channels, out_channels, level)
    - _pooler(level)
    - _out_conv(in_channels, out_channels)

    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      final_activation: activation applied to the network output
    """

    def __init__(self, in_channels=1, out_channels=1,
                 initial_features=64, gain=2, depth=4,
                 final_activation=None):
        super().__init__()

        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels

        # the final activation must either be None or a Module
        if final_activation is not None:
            assert isinstance(final_activation, nn.Module), "Activation must be torch module"

        # modules of the encoder path
        n_features = [in_channels] + [initial_features * gain ** level
                                      for level in range(self.depth)]
        # print("encoder:")
        # print(n_features)
        self.encoder = nn.ModuleList([self._conv_block(n_features[level], n_features[level + 1],
                                                       level, part='encoder')
                                      for level in range(self.depth)])

        # the base convolution block
        self.base = self._conv_block(n_features[-1], gain * n_features[-1], part='base', level=0)

        # modules of the decoder path
        n_features = [initial_features * gain ** level
                      for level in range(self.depth + 1)]
        n_features = n_features[::-1]
        # print("decoder:")
        # print(n_features)
        self.decoder = nn.ModuleList([self._conv_block(n_features[level], n_features[level + 1],
                                                       level, part='decoder')
                                      for level in range(self.depth)])

        # the pooling layers; we use 2x2 MaxPooling
        self.poolers = nn.ModuleList([self._pooler(level) for level in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList([self._upsampler(n_features[level],
                                                         n_features[level + 1], level)
                                         for level in range(self.depth)])
        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        self.out_conv = self._out_conv(n_features[-1], out_channels)
        self.activation = final_activation

    # crop the `from_encoder` tensor and concatenate both
    def _crop_and_concat(self, from_decoder, from_encoder):
        cropped = crop_tensor(from_encoder, from_decoder.shape)
        return torch.cat((cropped, from_decoder), dim=1)

    def forward(self, input):
        x = input
        # apply encoder path
        encoder_out = []
        for level in range(self.depth):
            x = self.encoder[level](x)
            encoder_out.append(x)
            x = self.poolers[level](x)

        # apply base
        x = self.base(x)

        # apply decoder path
        encoder_out = encoder_out[::-1]
        for level in range(self.depth):
            x = self.upsamplers[level](x)
            x = self.decoder[level](self._crop_and_concat(x,
                                                          encoder_out[level]))

        # apply output conv and activation (if given)
        x = self.out_conv(x)
        if self.activation is not None:
            x = self.activation(x)
        return x


class Unet2d(UNetBase):
    # Convolutional block for single layer of the decoder / encoder
    # we apply to 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels, level, part):
        return nn.Sequential(nn.Conv2d(in_channels, out_channels, kernel_size=3),
                             nn.ReLU(),
                             nn.Conv2d(out_channels, out_channels, kernel_size=3),
                             nn.ReLU())


    # upsampling via transposed 2d convolutions
    def _upsampler(self, in_channels, out_channels, level):
        return nn.ConvTranspose2d(in_channels, out_channels,
                                  kernel_size=2, stride=2)

    # pooling via maxpool2d
    def _pooler(self, level):
        return nn.MaxPool2d(2)

    def _out_conv(self, in_channels, out_channels):
        return nn.Conv2d(in_channels, out_channels, 1)
