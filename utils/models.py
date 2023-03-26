import torch
from torch import nn
from torch.cuda.amp import autocast
from typing import Optional, Union, List
import segmentation_models_pytorch as smp

# from .model_MedT import MedT
# from .models_MSAttUNet import MSAttUNetPlusPlus
# from .model_MidConUNetpp import MidCon_UNetplusplus
# from .model_PraNet import PraNet
# from .model_effiUNetpp import EfficientUnetPlusPlus
# from .model_segtran import Segtran2d
# from .model_UNetACB import UnetPlusPlus_ACB
# from .model_effiUNetpp_CBAM import EfficientUnetPlusPlus as EfficientUnetPlusPlusCBAM
# from .model_UNetppAux import EfficientUnetPlusPlus_Aux

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

class UNetplusplus_ReCo(smp.UnetPlusPlus):
    def __init__(self, encoder_name: str = "resnet34",
                encoder_depth: int = 5,
                encoder_weights: Optional[str] = "imagenet",
                decoder_use_batchnorm: bool = True,
                decoder_channels: List[int] = (256, 128, 64, 32, 16),
                decoder_attention_type: Optional[str] = None,
                in_channels: int = 3,
                classes: int = 1,
                output_dim: int = 128,
                activation: Optional[Union[str, callable]] = None,
                aux_params: Optional[dict] = None,
                pretrain_pth: Optional[str] = None) -> None:
        super().__init__(encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                decoder_use_batchnorm=decoder_use_batchnorm,
                decoder_channels=decoder_channels,
                decoder_attention_type=decoder_attention_type,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
                aux_params=aux_params)

        self.representation = nn.Sequential(
            nn.Conv2d(decoder_channels[-1], 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, output_dim, 1)
        )
        # load pretrained model based on optical images
        if pretrain_pth:
            loaded = torch.load(pretrain_pth)
            state = self.state_dict()
            pretrained_dict = {k: v for k, v in loaded['state_dict'].items()}
            state.update(pretrained_dict)
            self.load_state_dict(state)
            print(f"pretrained model {pretrain_pth} loading finished")

    def forward(self, x):
        features = self.encoder(x)
        # print(list(map(lambda x: x.shape, features)))
        decoder_output = self.decoder(*features)
        # print(decoder_output.shape)
        masks = self.segmentation_head(decoder_output)
        rep = self.representation(decoder_output)

        return masks, rep


class UNetplusplus_mpt(smp.UnetPlusPlus):
    """
    mixed precision training version of unetplusplus
    https://pytorch.org/docs/stable/notes/amp_examples.html
    """
    def __init__(self, encoder_name: str = "resnet34",
                encoder_depth: int = 5,
                encoder_weights: Optional[str] = "imagenet",
                decoder_use_batchnorm: bool = True,
                decoder_channels: List[int] = (256, 128, 64, 32, 16),
                decoder_attention_type: Optional[str] = None,
                in_channels: int = 3,
                classes: int = 1,
                activation: Optional[Union[str, callable]] = None,
                aux_params: Optional[dict] = None) -> None:
        super().__init__(encoder_name=encoder_name,
                encoder_depth=encoder_depth,
                encoder_weights=encoder_weights,
                decoder_use_batchnorm=decoder_use_batchnorm,
                decoder_channels=decoder_channels,
                decoder_attention_type=decoder_attention_type,
                in_channels=in_channels,
                classes=classes,
                activation=activation,
                aux_params=aux_params)
    @autocast()
    def forward(self, x):
        features = self.encoder(x)
        decoder_output = self.decoder(*features)
        masks = self.segmentation_head(decoder_output)
        return masks


model_dict = {
    'deeplabv3plus':smp.DeepLabV3Plus,
    # 'deeplabv3plus_reco':DeepLabV3Plus_ReCo,
    'unet':smp.Unet,
    'unetplusplus':smp.UnetPlusPlus,
    'unetplusplus_reco':UNetplusplus_ReCo,
    # 'medt':MedT,
    # 'msattunetplusplus':MSAttUNetPlusPlus,
    # 'pa_mpt_unetplusplus':UNetplusplus_mpt,
    # 'midconunetplusplus':MidCon_UNetplusplus,
    # 'pranet':PraNet,
    # 'effunetplusplus':EfficientUnetPlusPlus,
    # 'segtran2d':Segtran2d,
    # 'unetplusplusacb':UnetPlusPlus_ACB,
    # 'effunetpluspluscbam': EfficientUnetPlusPlusCBAM,
    # 'effunetplutplusaux': EfficientUnetPlusPlus_Aux,
    # 'manet': smp.MAnet
}


def _load_model_weights(model, path):
    """Backend for loading the model."""
    if torch.cuda.is_available():
        try:
            loaded = torch.load(path)
        except FileNotFoundError:
            raise FileNotFoundError("{} doesn't exist.".format(path))
    else:
        try:
            loaded = torch.load(path, map_location='cpu')
        except FileNotFoundError:
            raise FileNotFoundError("{} doesn't exist.".format(path))

    # if isinstance(loaded, torch.nn.Module):  # if it's a full model already
    #     model.load_state_dict(loaded.state_dict())
    # else:
    #     model.load_state_dict(loaded)
    model.load_state_dict(loaded['state_dict'])
    print(f"model {path} loading finished")
    return model

def _load_model_weights_v2(model, path):
    """Backend for loading the model."""
    if torch.cuda.is_available():
        try:
            loaded = torch.load(path)
        except FileNotFoundError:
            raise FileNotFoundError("{} doesn't exist.".format(path))
    else:
        try:
            loaded = torch.load(path, map_location='cpu')
        except FileNotFoundError:
            raise FileNotFoundError("{} doesn't exist.".format(path))
    
    state = model.state_dict()
    pretrained_dict = {k: v for k, v in loaded['state_dict'].items() if 'segmentation_head' not in k}
    state.update(pretrained_dict)
    model.load_state_dict(state)
    print(f"pretrained model {path} loading finished")
    return model



