from ssd.modeling import registry
from .vgg import VGG
from .mobilenet import MobileNetV2
from .mobilenet_7features import MobileNetV2_7features
from .mobilenet_fpn import MobileNetV2_fpn
from .mobilenet_fpn_test import MobileNetV2_fpn_test
from .mobilenet_se_fpn import MobileNetV2_se_fpn
from .mobilenet_fpn2up import MobileNetV2_fpn2up
from .mobilenet_res import MobileNetV2_res
from .mobilenet_layer7 import MobileNetV2_layer7
from .efficient_net import EfficientNet
from .cfenet import CFENET
from .cfenet_v2 import CFENET_v2
from .mobile_cfenet import Mobile_CFENET
from .mobilenet_fpn_r2 import MobileNetV2_fpn_r2
from .mobilenet_fpn_r3 import MobileNetV2_fpn_r3
from .mobilenet_fpn7_cfe import MobileNetV2_fpn7_cfe
from .mobilenet_fpn7 import MobileNetV2_fpn7
from .mobilenet_fpn6_mixconv import MobileNetV2_fpn6_mixconv

__all__ = [
    'build_backbone',\
    'VGG',\
    'MobileNetV2',\
    'MobileNetV2_res',\
    'MobileNetV2_fpn',\
    'MobileNetV2_test',\
    'MobileNetV2_fpn2up',\
    'MobileNetV2_se_fpn',\
    'MobileNetV2_layer7' ,\
    'MobileNetV2_7features',\
    'EfficientNet',\
    'CFENET',\
    'CFENET_v2',\
    'Mobile_CFENET',\
    'MobileNetV2_fpn_r2',\
    'MobileNetV2_fpn_r3',\
    'MobileNetV2_fpn7_cfe',\
    'MobileNetV2_fpn7',\
    'mobilenet_v2_fpn6_mixconv']


def build_backbone(cfg):
    return registry.BACKBONES[cfg.MODEL.BACKBONE.NAME](cfg, cfg.MODEL.BACKBONE.PRETRAINED)
