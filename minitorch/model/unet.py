import torch
import torch.nn as nn


class UNetBase(nn.Module):
    """ UNet Base class implementation

    Deriving classes must implement
    - _conv_block(in_channels, out_channels, level, part)
        return conv block for a U-Net level
    - _pooler(level)
        return pooling operation used for downsampling in-between encoders
    - _upsampler(in_channels, out_channels, level)
        return upsampling operation used for upsampling in-between decoders
    - _out_conv(in_channels, out_channels)
        return output conv layer

    Arguments:
      in_channels: number of input channels
      out_channels: number of output channels
      initial_features: number of features after first convolution
      gain: growth factor of features
      depth: depth of the u-net
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
        self.encoder = nn.ModuleList([self._conv_block(n_features[level], n_features[level + 1],
                                                       level, part='encoder')
                                      for level in range(self.depth)])

        # the base convolution block
        self.base = self._conv_block(n_features[-1], gain * n_features[-1], part='base', level=0)

        # modules of the decoder path
        n_features = [initial_features * gain ** level
                      for level in range(self.depth + 1)]
        n_features = n_features[::-1]
        self.decoder = nn.ModuleList([self._conv_block(n_features[level], n_features[level + 1],
                                                       self.depth - level - 1, part='decoder')
                                      for level in range(self.depth)])

        # the pooling layers;
        self.poolers = nn.ModuleList([self._pooler(level) for level in range(self.depth)])
        # the upsampling layers
        self.upsamplers = nn.ModuleList([self._upsampler(n_features[level],
                                                         n_features[level + 1],
                                                         self.depth - level - 1)
                                         for level in range(self.depth)])
        # output conv and activation
        # the output conv is not followed by a non-linearity, because we apply
        # activation afterwards
        self.out_conv = self._out_conv(n_features[-1], out_channels)
        self.activation = final_activation

    # NOTE we duplicate this from `minitorch.utils.data` so that we can provide
    # this file as a standalone header
    @staticmethod
    def _crop_tensor(input_, shape_to_crop):
        input_shape = input_.shape
        # get the difference between the shapes
        shape_diff = tuple((ish - csh) // 2
                           for ish, csh in zip(input_shape, shape_to_crop))
        # calculate the crop
        crop = tuple(slice(sd, sh - sd)
                     for sd, sh in zip(shape_diff, input_shape))
        return input_[crop]

    # crop the `from_encoder` tensor and concatenate both
    def _crop_and_concat(self, from_decoder, from_encoder):
        cropped = self._crop_tensor(from_encoder, from_decoder.shape)
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


#
# 2D U-Net implementations
#


class UNet2d(UNetBase):
    """ 2d U-Net for segmentation as described in
    https://arxiv.org/abs/1505.04597
    """
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


class UNet2dGN(UNet2d):
    """ 2d U-Net with GroupNorm
    """
    # Convolutional block for single layer of the decoder / encoder
    # we apply to 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels, level, part):
        num_groups1 = min(in_channels, 32)
        num_groups2 = min(out_channels, 32)
        return nn.Sequential(nn.GroupNorm(num_groups1, in_channels),
                             nn.Conv2d(in_channels, out_channels, kernel_size=3),
                             nn.ReLU(),
                             nn.GroupNorm(num_groups2, out_channels),
                             nn.Conv2d(out_channels, out_channels, kernel_size=3),
                             nn.ReLU())


def unet_2d(pretrained=None, **kwargs):
    net = UNet2dGN(**kwargs)
    if pretrained is not None:
        assert pretrained in ('isbi',)
        # TODO implement download
    return net


#
# 3D U-Net implementations
#


class UNet3d(UNetBase):
    """ 3d U-Net for segmentation as described in
    https://arxiv.org/abs/1606.06650
    """
    # Convolutional block for single layer of the decoder / encoder
    # we apply to 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels, level, part):
        return nn.Sequential(nn.Conv3d(in_channels, out_channels, kernel_size=3),
                             nn.ReLU(),
                             nn.Conv3d(out_channels, out_channels, kernel_size=3),
                             nn.ReLU())

    # upsampling via transposed 3d convolutions
    def _upsampler(self, in_channels, out_channels, level):
        return nn.ConvTranspose3d(in_channels, out_channels,
                                  kernel_size=2, stride=2)

    # pooling via maxpool3d
    def _pooler(self, level):
        return nn.MaxPool3d(2)

    def _out_conv(self, in_channels, out_channels):
        return nn.Conv3d(in_channels, out_channels, 1)


class UNet3dGN(UNet3d):
    """ 3d U-Net with GroupNorm
    """
    # Convolutional block for single layer of the decoder / encoder
    # we apply to 2d convolutions with relu activation
    def _conv_block(self, in_channels, out_channels, level, part):
        num_groups1 = min(in_channels, 32)
        num_groups2 = min(out_channels, 32)
        return nn.Sequential(nn.GroupNorm(num_groups1, in_channels),
                             nn.Conv3d(in_channels, out_channels, kernel_size=3),
                             nn.ReLU(),
                             nn.GroupNorm(num_groups2, out_channels),
                             nn.Conv3d(out_channels, out_channels, kernel_size=3),
                             nn.ReLU())


class AnisotropicUNet(UNet3dGN):
    """ 3D GroupNorm U-Net with anisotropic scaling

    Arguments:
      scale_factors: list of scale factors
      in_channels: number of input channels
      out_channels: number of output channels
      initial_features: number of features after first convolution
      gain: growth factor of features
      final_activation: activation applied to the network output
    """
    @staticmethod
    def _validate_scale_factors(scale_factors):
        assert isinstance(scale_factors, (list, tuple))
        for sf in scale_factors:
            assert isinstance(sf, (int, tuple))
            if not isinstance(sf, int):
                assert len(sf) == 3
                assert all(isinstance(sff, int) for sff in sf)

    def __init__(self, scale_factors, in_channels=1,
                 out_channels=1, initial_features=64,
                 gain=2, final_activation=None):
        self._validate_scale_factors(scale_factors)
        self.scale_factors = scale_factors
        super().__init__(in_channels=in_channels, out_channels=out_channels,
                         initial_features=initial_features, gain=gain,
                         detph=len(self.scale_factors), final_activation=final_activation)

    def _upsampler(self, in_channels, out_channels, level):
        scale_factor = self.scale_factors[level]
        return nn.ConvTranspose3d(in_channels, out_channels,
                                  kernel_size=scale_factor,
                                  stride=scale_factor)

    def _pooler(self, level):
        return nn.MaxPool3d(self.scale_factors[level])
