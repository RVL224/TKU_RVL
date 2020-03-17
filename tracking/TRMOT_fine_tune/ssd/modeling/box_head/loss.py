import torch.nn as nn
import torch.nn.functional as F
import torch

from ssd.utils import box_utils, mask_splitter
import math

class MultiBoxLoss(nn.Module):
    def __init__(self, cfg, nID):
        """Implement SSD MultiBox Loss.

        Basically, MultiBox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(MultiBoxLoss, self).__init__()
        self.cfg = cfg
        self.neg_pos_ratio = self.cfg.MODEL.NEG_POS_RATIO
        self.emb_scale = math.sqrt(2) * math.log(nID-1)
        self.classifier = nn.Linear(cfg.MODEL.EMB_DIM, nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)
        
        self.mask_splitter = mask_splitter.splitter(cfg)

    def forward(self, img_path, confidence, predicted_locations, predicted_ids, labels, gt_locations, gt_ids):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """

        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            emb_mask = box_utils.hard_negative_mining(loss, labels, 15)

        ####embeding####
        gt_id_mask = self.mask_splitter.con_split(emb_mask)
        gid_list = []
        logit_lsit = []
        gt_ids = self.mask_splitter.split(gt_ids)
        for pid, gid, mask in zip(predicted_ids, gt_ids, gt_id_mask):
            # pid = pid.view(pid.shape[0], pid.shape[1], pid.shape[2], -1, 512)[mask].contiguous()
            mask,_ = mask.max(3)
            pid = pid[mask].contiguous()
            pid = self.emb_scale * F.normalize(pid)
            logit = self.classifier(pid).contiguous()
            logit_lsit.append(logit)

            gid = gid
            gid,_ = gid.max(3)
            gid = gid[mask]
            gid_list.append(gid)

        gt_ids = torch.cat([i for i in gid_list], dim=0)
        logits = torch.cat([i for i in logit_lsit], dim=0)

        loss = F.cross_entropy(logits, gt_ids, ignore_index=-1, reduction='mean')

        #loss

        nT = gt_locations.size(0)
        loss = loss.squeeze()

        return loss, [loss.item(), nT]

