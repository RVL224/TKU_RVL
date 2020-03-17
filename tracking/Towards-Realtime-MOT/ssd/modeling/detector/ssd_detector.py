from torch import nn

from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head


class SSDDetector(nn.Module):
    def __init__(self, cfg, nID):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = build_box_head(cfg, nID)

    def forward(self, images, targets=None, targets_len=None, test_emb=False):
        features = self.backbone(images) #回傳6個features_(vgg)
        loss, losses = self.box_head(features, targets, test_emb)
        if self.training:
            return loss, losses
        return loss
