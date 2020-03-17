from ssd.modeling import registry
from .box_head import SSDBoxHead

__all__ = ['build_box_head', 'SSDBoxHead']


def build_box_head(cfg, nID):
    print('-------build_box_head----------', cfg.MODEL.BOX_HEAD.NAME)
    return registry.BOX_HEADS[cfg.MODEL.BOX_HEAD.NAME](cfg, nID)
