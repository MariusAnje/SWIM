import torch
import torch.nn as nn
from typing import Union, List, Dict, Any, cast
from Functions import SCrossEntropyLossFunction
from modules import SReLU, SModel, SMaxpool2D, SBatchNorm2d, SAdaptiveAvgPool2d
from qmodules import QSLinear, QSConv2d



__all__ = [
    'VGG', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn',
    'vgg19_bn', 'vgg19',
]


model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-8a719046.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-19584684.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth',
}


class VGG(SModel):

    def __init__(
        self,
        features,
        num_classes: int = 1000,
        init_weights: bool = True,
        N = 6
    ) -> None:
        super(VGG, self).__init__()
        self.features = features
        self.avgpool = SAdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            QSLinear(N, 512 * 7 * 7, 4096),
            SReLU(),
            # nn.Dropout(),
            QSLinear(N, 4096, 4096),
            SReLU(),
            # nn.Dropout(),
            QSLinear(N, 4096, num_classes),
        )
        if init_weights:
            self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.first_only:
            xS = torch.zeros_like(x)
            x = (x, xS)
        x = self.features(x)
        x = self.avgpool(x)
        x = self.unpack_flattern(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, QSConv2d):
                nn.init.kaiming_normal_(m.op.weight, mode='fan_out', nonlinearity='relu')
                if m.op.bias is not None:
                    nn.init.constant_(m.op.bias, 0)
            elif isinstance(m, SBatchNorm2d):
                nn.init.constant_(m.op.weight, 1)
                nn.init.constant_(m.op.bias, 0)
            elif isinstance(m, QSLinear):
                nn.init.normal_(m.op.weight, 0, 0.01)
                nn.init.constant_(m.op.bias, 0)


def make_layers(N, cfg: List[Union[str, int]], batch_norm: bool = False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [SMaxpool2D(kernel_size=2, stride=2)]
        else:
            v = cast(int, v)
            conv2d = QSConv2d(N, in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, SBatchNorm2d(v), SReLU()]
            else:
                layers += [conv2d, SReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfgs: Dict[str, List[Union[str, int]]] = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def _vgg(arch: str, cfg: str, batch_norm: bool, pretrained: bool, progress: bool, **kwargs: Any) -> VGG:
    # if pretrained:
    #     kwargs['init_weights'] = False
    N = 16
    model = VGG(make_layers(N, cfgs[cfg], batch_norm=batch_norm), N=N, **kwargs)
    # if pretrained:
    #     state_dict = load_state_dict_from_url(model_urls[arch],
    #                                           progress=progress)
    #     model.load_state_dict(state_dict)
    return model


def vgg11(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG 11-layer model (configuration "A") from
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11', 'A', False, pretrained, progress, **kwargs)




def vgg11_bn(pretrained: bool = False, progress: bool = True, **kwargs: Any) -> VGG:
    """VGG 11-layer model (configuration "A") with batch normalization
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_.
    The required minimum input size of the model is 32x32.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _vgg('vgg11_bn', 'A', True, pretrained, progress, **kwargs)

