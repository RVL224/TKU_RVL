from torch import nn

from ssd.modeling.backbone import build_backbone
from ssd.modeling.box_head import build_box_head

import numpy as np

class SSDDetector(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.backbone = build_backbone(cfg)
        self.box_head = build_box_head(cfg)

    def forward(self, images, targets=None):
        features = self.backbone(images)
        # features_t = features[0].permute(0, 2, 3, 1)
        # print(features_t.size())
        # import sys
        # sys.exit()

        detections, detector_losses = self.box_head(features, targets)
        if self.training:
            return detector_losses
        return detections
