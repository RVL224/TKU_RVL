from torch import nn
import torch
import torch.nn.functional as F

from collections import OrderedDict

from ssd.modeling import registry
from ssd.modeling.anchors.prior_box import PriorBox
from ssd.modeling.box_head.box_predictor import make_box_predictor
from ssd.utils import box_utils
from .inference import PostProcessor
from .loss import MultiBoxLoss
from .test_emb_block import Test_emb


@registry.BOX_HEADS.register('SSDBoxHead')
class SSDBoxHead(nn.Module):
    def __init__(self, cfg, nID):
        super().__init__()
        self.cfg = cfg
        self.predictor = make_box_predictor(cfg)
        self.loss_evaluator = MultiBoxLoss(cfg, nID = nID)
        self.test_emb = Test_emb(cfg, nID = nID)
        self.post_processor = PostProcessor(cfg)
        self.priors = None

        self.loss_names = ['loss', 'nT']
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0

    def forward(self, features, targets=None, test_emb=False):
        self.losses = OrderedDict()
        for ln in self.loss_names:
            self.losses[ln] = 0
        cls_logits, bbox_pred, id_pred = self.predictor(features)
        if test_emb:
            return self._forward_test_emb(cls_logits, bbox_pred, id_pred, targets)
        elif self.training:
            return self._forward_train(cls_logits, bbox_pred, id_pred, targets)
        else:
            return self._forward_test(cls_logits, bbox_pred, id_pred)

    def _forward_train(self, cls_logits, bbox_pred, id_pred, targets):
        gt_boxes, gt_labels, gt_ids, img_path = targets['boxes'], targets['labels'], targets['ids'], targets['img_path'] 
        loss, losses= self.loss_evaluator(img_path, cls_logits, bbox_pred, id_pred, gt_labels, gt_boxes, gt_ids)
        for name, l in zip(self.loss_names, losses):
            self.losses[name] += l
        return loss, torch.Tensor(list(self.losses.values())).cuda()

    def _forward_test(self, cls_logits, bbox_pred, id_pred):
        if self.priors is None:
            self.priors = PriorBox(self.cfg)().to(bbox_pred.device)
        scores = F.softmax(cls_logits, dim=2)
        boxes = box_utils.convert_locations_to_boxes(
            bbox_pred, self.priors, self.cfg.MODEL.CENTER_VARIANCE, self.cfg.MODEL.SIZE_VARIANCE
        )
        boxes = box_utils.center_form_to_corner_form(boxes)
        detections = (scores, boxes, id_pred)
        detections = self.post_processor(detections)
        return detections, {}

    def _forward_test_emb(self, cls_logits, bbox_pred, id_pred, targets):
        gt_boxes, gt_labels, gt_ids, img_path = targets['boxes'], targets['labels'], targets['ids'], targets['img_path']
        emb_and_gt = self.test_emb(img_path, cls_logits, bbox_pred, id_pred, gt_labels, gt_boxes, gt_ids)
        return emb_and_gt, 0

