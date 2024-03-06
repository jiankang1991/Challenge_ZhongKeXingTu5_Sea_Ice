
from typing import Optional, Union, List
from segmentation_models_pytorch.base import SegmentationModel, SegmentationHead, ClassificationHead
# from segmentation_models_pytorch.unet.decoder import UnetDecoder
from segmentation_models_pytorch.encoders import get_encoder
from segmentation_models_pytorch.base import modules as md
from torchvision import transforms
import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if torch.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta, percentage):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)

class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = md.Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class DWSWConv(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):

        dwconv_1 = nn.Conv2d(in_channels, in_channels, kernel_size=(kernel_size, 1), padding=(kernel_size//2, 0), groups=in_channels)
        dwconv_2 = nn.Conv2d(in_channels, in_channels, kernel_size=(1, kernel_size), padding=(0, kernel_size//2), groups=in_channels)
        swconv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        bn = nn.BatchNorm2d(out_channels)
        relu = nn.ReLU()
        super().__init__(dwconv_1, dwconv_2, swconv, bn, relu)

class LightDecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            use_batchnorm=True,
            attention_type=None,
    ):
        super().__init__()
        self.conv1 = DWSWConv(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention1 = md.Attention(attention_type, in_channels=in_channels + skip_channels)
        self.conv2 = DWSWConv(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.attention2 = md.Attention(attention_type, in_channels=out_channels)

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
            x = self.attention1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.attention2(x)
        return x

class CenterBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, use_batchnorm=True):
        conv1 = md.Conv2dReLU(
            in_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        conv2 = md.Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        super().__init__(conv1, conv2)

class DepthWiseConv2d(nn.Conv2d):
    "Depth-wise convolution operation"
    def __init__(self, channels, kernel_size=3, stride=1):
        super().__init__(channels, channels, kernel_size, stride=stride, padding=kernel_size//2, groups=channels)

class PointWiseConv2d(nn.Conv2d):
    "Point-wise (1x1) convolution operation"
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=1, stride=1)

class Conv2d(nn.Conv2d):
    "https://github.com/joe-siyuan-qiao/WeightStandardization"
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True):
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size, stride,
                 padding, dilation, groups, bias)

    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2,
                                  keepdim=True).mean(dim=3, keepdim=True)
        weight = weight - weight_mean
        std = weight.view(weight.size(0), -1).std(dim=1).view(-1, 1, 1, 1) + 1e-5
        weight = weight / std.expand_as(weight)
        return F.conv2d(x, weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

class DWConv2d(Conv2d):
    def __init__(self, channels, kernel_size=3, stride=1):
        super().__init__(channels, channels, kernel_size, stride=stride, padding=kernel_size//2, groups=channels)

class PWConv2d(Conv2d):
    def __init__(self, in_channels, out_channels):
        super().__init__(in_channels, out_channels, kernel_size=1, stride=1)


class InvertedResidual(nn.Module):
    """
    Inverted bottleneck residual block with an scSE block embedded into the residual layer, after the 
    depthwise convolution. By default, uses batch normalization and Hardswish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, expansion_ratio = 1, squeeze_ratio = 1, \
        activation = nn.Hardswish(True), normalization = nn.BatchNorm2d):
        super().__init__()
        self.same_shape = in_channels == out_channels
        self.mid_channels = expansion_ratio*in_channels
        self.block = nn.Sequential(
            PointWiseConv2d(in_channels, self.mid_channels),
            normalization(self.mid_channels),
            activation,
            DepthWiseConv2d(self.mid_channels, kernel_size=kernel_size, stride=stride),
            normalization(self.mid_channels),
            activation,
            #md.sSEModule(self.mid_channels),
            md.SCSEModule(self.mid_channels, reduction = squeeze_ratio),
            #md.SEModule(self.mid_channels, reduction = squeeze_ratio),
            PointWiseConv2d(self.mid_channels, out_channels),
            normalization(out_channels)
        )
        
        if not self.same_shape:
            # 1x1 convolution used to match the number of channels in the skip feature maps with that 
            # of the residual feature maps
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                normalization(out_channels)
            )
          
    def forward(self, x):
        residual = self.block(x)
        
        if not self.same_shape:
            x = self.skip_conv(x)
        return x + residual

class InvertedResidual_WS(nn.Module):
    """
    Inverted bottleneck residual block with an scSE block embedded into the residual layer, after the 
    depthwise convolution. By default, uses batch normalization and Hardswish activation.
    """
    def __init__(self, in_channels, out_channels, kernel_size = 3, stride = 1, expansion_ratio = 1, squeeze_ratio = 1, \
        activation = nn.Hardswish(True), normalization = nn.BatchNorm2d):
        super().__init__()
        self.same_shape = in_channels == out_channels
        self.mid_channels = expansion_ratio*in_channels
        self.block = nn.Sequential(
            PWConv2d(in_channels, self.mid_channels),
            normalization(self.mid_channels),
            activation,
            DWConv2d(self.mid_channels, kernel_size=kernel_size, stride=stride),
            normalization(self.mid_channels),
            activation,
            #md.sSEModule(self.mid_channels),
            md.SCSEModule(self.mid_channels, reduction = squeeze_ratio),
            #md.SEModule(self.mid_channels, reduction = squeeze_ratio),
            PWConv2d(self.mid_channels, out_channels),
            normalization(out_channels)
        )
        
        if not self.same_shape:
            # 1x1 convolution used to match the number of channels in the skip feature maps with that 
            # of the residual feature maps
            self.skip_conv = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1),
                normalization(out_channels)
            )
          
    def forward(self, x):
        residual = self.block(x)
        
        if not self.same_shape:
            x = self.skip_conv(x)
        return x + residual

class UnetDecoder(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock(in_ch, skip_ch, out_ch, **kwargs)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        ouputs = [x]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            ouputs.append(x)
        return list(reversed(ouputs))

class DecoderBlock_v2(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            squeeze_ratio=1,
            expansion_ratio=1
    ):
        super().__init__()

        # Inverted Residual block convolutions
        self.conv1 = InvertedResidual(
            in_channels=in_channels+skip_channels, 
            out_channels=out_channels, 
            kernel_size=5, 
            stride=1, 
            expansion_ratio=expansion_ratio, 
            squeeze_ratio=squeeze_ratio
        )
        self.conv2 = InvertedResidual(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=5, 
            stride=1, 
            expansion_ratio=expansion_ratio, 
            squeeze_ratio=squeeze_ratio
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class DecoderBlock_v2_WS(nn.Module):
    def __init__(
            self,
            in_channels,
            skip_channels,
            out_channels,
            squeeze_ratio=1,
            expansion_ratio=1
    ):
        super().__init__()

        # Inverted Residual block convolutions
        self.conv1 = InvertedResidual_WS(
            in_channels=in_channels+skip_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            expansion_ratio=expansion_ratio, 
            squeeze_ratio=squeeze_ratio
        )
        self.conv2 = InvertedResidual_WS(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=3, 
            stride=1, 
            expansion_ratio=expansion_ratio, 
            squeeze_ratio=squeeze_ratio
        )

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")

        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class UnetDecoder_v2(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        # kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock_v2(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        ouputs = [x]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            ouputs.append(x)
        return list(reversed(ouputs))

class UnetDecoder_v2_WS(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        # kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            DecoderBlock_v2_WS(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        ouputs = [x]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            ouputs.append(x)
        return list(reversed(ouputs))

class UnetDecoder_v3(nn.Module):
    def __init__(
            self,
            encoder_channels,
            decoder_channels,
            n_blocks=5,
            use_batchnorm=True,
            attention_type=None,
            center=False,
    ):
        super().__init__()

        if n_blocks != len(decoder_channels):
            raise ValueError(
                "Model depth is {}, but you provide `decoder_channels` for {} blocks.".format(
                    n_blocks, len(decoder_channels)
                )
            )

        encoder_channels = encoder_channels[1:]  # remove first skip with same spatial resolution
        encoder_channels = encoder_channels[::-1]  # reverse channels to start from head of encoder

        # computing blocks input and output channels
        head_channels = encoder_channels[0]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        skip_channels = list(encoder_channels[1:]) + [0]
        out_channels = decoder_channels

        if center:
            self.center = CenterBlock(
                head_channels, head_channels, use_batchnorm=use_batchnorm
            )
        else:
            self.center = nn.Identity()

        # combine decoder keyword arguments
        # kwargs = dict(use_batchnorm=use_batchnorm, attention_type=attention_type)
        blocks = [
            LightDecoderBlock(in_ch, skip_ch, out_ch)
            for in_ch, skip_ch, out_ch in zip(in_channels, skip_channels, out_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, *features):

        features = features[1:]    # remove first skip with same spatial resolution
        features = features[::-1]  # reverse channels to start from head of encoder

        head = features[0]
        skips = features[1:]

        x = self.center(head)
        ouputs = [x]
        for i, decoder_block in enumerate(self.blocks):
            skip = skips[i] if i < len(skips) else None
            x = decoder_block(x, skip)
            ouputs.append(x)
        return list(reversed(ouputs))

class UnetAux(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.segmentation_head_1 = SegmentationHead(
            in_channels=decoder_channels[-2],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.segmentation_head_2 = SegmentationHead(
            in_channels=decoder_channels[-3],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        
        features = self.encoder(x)
        # print(list(map(lambda x: x.shape, features)))
        decoder_outputs = self.decoder(*features)
        # print(list(map(lambda x: x.shape, decoder_outputs)))
        seg0 = self.segmentation_head(decoder_outputs[0])
        seg1 = self.segmentation_head_1(decoder_outputs[1])
        seg2 = self.segmentation_head_2(decoder_outputs[2])

        return seg0, seg1, seg2

class UnetAux_v2(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder_v2(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.segmentation_head_1 = SegmentationHead(
            in_channels=decoder_channels[-2],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.segmentation_head_2 = SegmentationHead(
            in_channels=decoder_channels[-3],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        
        features = self.encoder(x)
        # print(list(map(lambda x: x.shape, features)))
        decoder_outputs = self.decoder(*features)
        # print(list(map(lambda x: x.shape, decoder_outputs)))
        seg0 = self.segmentation_head(decoder_outputs[0])
        seg1 = self.segmentation_head_1(decoder_outputs[1])
        seg2 = self.segmentation_head_2(decoder_outputs[2])

        return seg0, seg1, seg2
        # return seg0
        # return seg0, seg1, seg2, features[-1]

class UnetAux_v2_WS(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder_v2_WS(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.segmentation_head_1 = SegmentationHead(
            in_channels=decoder_channels[-2],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.segmentation_head_2 = SegmentationHead(
            in_channels=decoder_channels[-3],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        
        features = self.encoder(x)
        # print(list(map(lambda x: x.shape, features)))
        decoder_outputs = self.decoder(*features)
        # print(list(map(lambda x: x.shape, decoder_outputs)))
        seg0 = self.segmentation_head(decoder_outputs[0])
        seg1 = self.segmentation_head_1(decoder_outputs[1])
        seg2 = self.segmentation_head_2(decoder_outputs[2])

        return seg0, seg1, seg2

class UnetAux_v3(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.segmentation_head_1 = SegmentationHead(
            in_channels=decoder_channels[-2],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        self.segmentation_head_2 = SegmentationHead(
            in_channels=decoder_channels[-3],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )

        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        
        features = self.encoder(x)
        # print(list(map(lambda x: x.shape, features)))
        decoder_outputs = self.decoder(*features)
        # print(list(map(lambda x: x.shape, decoder_outputs)))
        seg0 = self.segmentation_head(decoder_outputs[0])
        seg1 = self.segmentation_head_1(decoder_outputs[1])
        seg2 = self.segmentation_head_2(decoder_outputs[2])

        return seg0, seg1, seg2

class UnetAux_v4(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet34",
        encoder_depth: int = 5,
        encoder_weights: Optional[str] = "imagenet",
        decoder_use_batchnorm: bool = True,
        decoder_channels: List[int] = (256, 128, 64, 32, 16),
        decoder_attention_type: Optional[str] = None,
        in_channels: int = 3,
        classes: int = 1,
        activation: Optional[Union[str, callable]] = None,
        aux_params: Optional[dict] = None,
    ):
        super().__init__()

        self.encoder = get_encoder(
            encoder_name,
            in_channels=in_channels,
            depth=encoder_depth,
            weights=encoder_weights,
        )

        self.decoder = UnetDecoder_v2(
            encoder_channels=self.encoder.out_channels,
            decoder_channels=decoder_channels,
            n_blocks=encoder_depth,
            use_batchnorm=decoder_use_batchnorm,
            center=True if encoder_name.startswith("vgg") else False,
            attention_type=decoder_attention_type,
        )

        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        self.seg0_drop = nn.Dropout2d(p=0.2)
        self.segmentation_head_1 = SegmentationHead(
            in_channels=decoder_channels[-2],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        self.seg1_drop = nn.Dropout2d(p=0.2)
        self.segmentation_head_2 = SegmentationHead(
            in_channels=decoder_channels[-3],
            out_channels=classes,
            activation=activation,
            kernel_size=3,
        )
        self.seg2_drop = nn.Dropout2d(p=0.2)
        if aux_params is not None:
            self.classification_head = ClassificationHead(
                in_channels=self.encoder.out_channels[-1], **aux_params
            )
        else:
            self.classification_head = None

        self.name = "u-{}".format(encoder_name)
        self.initialize()

    def forward(self, x):
        
        features = self.encoder(x)
        # print(list(map(lambda x: x.shape, features)))
        decoder_outputs = self.decoder(*features)
        # print(list(map(lambda x: x.shape, decoder_outputs)))
        seg0 = self.segmentation_head(self.seg0_drop(decoder_outputs[0]))
        seg1 = self.segmentation_head_1(self.seg1_drop(decoder_outputs[1]))
        seg2 = self.segmentation_head_2(self.seg2_drop(decoder_outputs[2]))

        return seg0, seg1, seg2

import segmentation_models_pytorch as smp
class compare_model(nn.Module):
    """
    Args:
    in_ch : input_channels
    ou_ch : num_classes
    model_name:[unet, Linknet, pspnet, fpn, deeplabv3, deeplabv3+]
                  default: 'resnet18'

    """
    def __init__(self,
                 in_ch:int = 3,
                 out_ch:int = 2,
                 model_name:str = 'unet'):
        super(compare_model, self).__init__()

        self.model_name = model_name
        if self.model_name == 'unet':
            self.model = smp.Unet('vgg16', encoder_depth=4, decoder_channels =(128,64,32,16),encoder_weights='imagenet',in_channels=in_ch,classes=out_ch)
        if self.model_name == 'Linknet':
            self.model = smp.Linknet('resnet18', encoder_depth=4, encoder_weights='imagenet',in_channels=in_ch,classes=out_ch)
        if self.model_name == 'pspnet':
            self.model = smp.PSPNet('mobilenet_v2', encoder_depth=3, encoder_weights='imagenet',in_channels=in_ch,classes=out_ch)
        if self.model_name == 'fpn':
            self.model = smp.FPN('resnet18', encoder_depth=5, encoder_weights='imagenet',in_channels=in_ch,classes=out_ch)
        if self.model_name == 'deeplabv3':
            self.model = smp.DeepLabV3('resnet18', encoder_depth=4, encoder_weights='imagenet',in_channels=in_ch,classes=out_ch)
        if self.model_name == 'deeplabv3+':
            self.model = smp.DeepLabV3Plus('resnet18', encoder_depth=5, encoder_weights='imagenet',in_channels=in_ch,classes=out_ch)


    def forward(self,x):
        out = self.model(x)
        return out 
    
if __name__ == '__main__':
    from thop import profile
    x = torch.randn((8,3,512,512))
    # model = UnetAux(encoder_name='efficientnet-b5', classes=2, encoder_depth=4, decoder_channels=(128, 64, 32, 16), decoder_attention_type='scse')
    model = UnetAux_v2(encoder_name='timm-tf_efficientnet_lite3', classes=2, encoder_depth=5, decoder_channels=(256, 128, 64, 32, 16), decoder_attention_type='scse')
    # model = UnetAux_v3(encoder_name='efficientnet-b5', classes=2, encoder_depth=4, decoder_channels=(128, 64, 32, 16), decoder_attention_type='scse')
    # numParams = count_parameters(model)
    # print(f"{numParams:.2E}")
    # model = compare_model(model_name='deeplabv3+')
    outputs = model(x)
    # print(list(map(lambda x: x.shape, outputs)))
    macs, params = profile(model,inputs=(x,))   ##verbose=False
    print('The number of MACs is %s'%(macs/1e9))   ##### MB
    print('The number of params is %s'%(params/1e6))   ##### MB
