import torch
from torch import nn

from ssd.layers import SeparableConv2d
from ssd.modeling import registry


class BoxPredictor(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.cls_headers = nn.ModuleList()
        self.reg_headers = nn.ModuleList()
        self.id_headers = nn.ModuleList()

        for level, (boxes_per_location, out_channels) in enumerate(zip(cfg.MODEL.PRIORS.BOXES_PER_LOCATION, cfg.MODEL.BACKBONE.OUT_CHANNELS)):
            self.cls_headers.append(self.cls_block(level, out_channels, boxes_per_location))
            self.reg_headers.append(self.reg_block(level, out_channels, boxes_per_location))
            self.id_headers.append(self.id_block(level, out_channels, boxes_per_location))
            #這邊引用的self.cls_block & self.reg_block是底下SSDBoxPredictor & SSDLiteBoxPredictor的而不是原本BoxPredictor中定義的
        self.reset_parameters()

    def cls_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def reg_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError

    def id_block(self, level, out_channels, boxes_per_location):
        raise NotImplementedError
    #把 def cls_block & def reg_block 兩個function都刪掉也能跑,在SSDBoxPredictor & SSDLiteBoxPredictor繼承時會重新def

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, features):
        cls_logits = []
        bbox_pred = []
        id_pred = []
        
        #print('start')
        for feature, cls_header, reg_header, id_header in zip(features, self.cls_headers, self.reg_headers, self.id_headers):
            cls_logits.append(cls_header(feature).permute(0, 2, 3, 1).contiguous())
            #print(cls_header(feature).permute(0, 2, 3, 1).contiguous().shape)

            bbox_pred.append(reg_header(feature).permute(0, 2, 3, 1).contiguous())
            #print(reg_header(feature).permute(0, 2, 3, 1).contiguous().shape)

            id_pred.append(id_header(feature).permute(0, 2, 3, 1).contiguous())
            #print(id_header(feature).permute(0, 2, 3, 1).contiguous().shape)
        #print('end')

        batch_size = features[0].shape[0]
        cls_logits = torch.cat([c.view(c.shape[0], -1) for c in cls_logits], dim=1).view(batch_size, -1, self.cfg.MODEL.NUM_CLASSES)
        #print(cls_logits.shape)

        bbox_pred = torch.cat([l.view(l.shape[0], -1) for l in bbox_pred], dim=1).view(batch_size, -1, 4)
        #print(bbox_pred.shape)

        id_pred = torch.cat([i.view(i.shape[0], -1) for i in id_pred], dim=1).view(batch_size, -1, 512)
        
        return cls_logits, bbox_pred, id_pred


@registry.BOX_PREDICTORS.register('SSDBoxPredictor')
class SSDBoxPredictor(BoxPredictor):
    def cls_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    def reg_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)

    def id_block(self, level, out_channels, boxes_per_location):
        return nn.Conv2d(out_channels, boxes_per_location * 512, kernel_size=3, stride=1, padding=1)


@registry.BOX_PREDICTORS.register('SSDLiteBoxPredictor')
class SSDLiteBoxPredictor(BoxPredictor):
    def cls_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        if level == num_levels - 1:
            return nn.Conv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=1)
        return SeparableConv2d(out_channels, boxes_per_location * self.cfg.MODEL.NUM_CLASSES, kernel_size=3, stride=1, padding=1)

    def reg_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        if level == num_levels - 1:
            return nn.Conv2d(out_channels, boxes_per_location * 4, kernel_size=1)
        return SeparableConv2d(out_channels, boxes_per_location * 4, kernel_size=3, stride=1, padding=1)

    def id_block(self, level, out_channels, boxes_per_location):
        num_levels = len(self.cfg.MODEL.BACKBONE.OUT_CHANNELS)
        if level == num_levels - 1:
            return nn.Conv2d(out_channels, boxes_per_location * 512, kernel_size=1)
        return SeparableConv2d(out_channels, boxes_per_location * 512, kernel_size=3, stride=1, padding=1)



def make_box_predictor(cfg):
    print('-------make_box_predictor----------', cfg.MODEL.BOX_HEAD.PREDICTOR)
    return registry.BOX_PREDICTORS[cfg.MODEL.BOX_HEAD.PREDICTOR](cfg)
