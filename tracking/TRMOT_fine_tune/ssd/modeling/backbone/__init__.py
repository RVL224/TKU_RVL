from ssd.modeling import registry
from .vgg import VGG
from .mobilenet import MobileNetV2
from .efficient_net import EfficientNet
from .mobilenet_fpn_test import MobileNetV2_fpn_test
from .vgg_M2det import VGG_M2det
from .mobilenet_fpn7_cfe import MobileNetV2_fpn7_cfe
from .mobilenet_fpn7 import mobilenet_v2_fpn7

__all__ = ['build_backbone', 'VGG', 'MobileNetV2', 'EfficientNet', 'MobileNetV2_fpn_test']


def build_backbone(cfg):
    print('-------build_backbone----------', cfg.MODEL.BACKBONE.NAME)
    print('-------build_backbone.PRETRAINED----------', cfg.MODEL.BACKBONE.PRETRAINED)
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
