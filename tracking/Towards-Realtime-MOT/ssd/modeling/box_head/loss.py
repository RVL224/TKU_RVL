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
        self.emb_neg_pos_ratio = self.cfg.MODEL.EMB_NEG_POS_RATIO
        self.emb_scale = math.sqrt(2) * math.log(nID-1)
        self.classifier = nn.Linear(cfg.MODEL.EMB_DIM, nID)
        self.IDLoss = nn.CrossEntropyLoss(ignore_index=-1)

        self.s_c = nn.Parameter(-4.15*torch.ones(1))  # -4.15 #k nn.Parameter將某個tensor轉成可訓練的類型parameter並將這個參數加入到module中,net.parameter()中就會有這個parameter,所以在優化時也會一起優化
        self.s_r = nn.Parameter(-4.85*torch.ones(1))  # -4.85
        self.s_id = nn.Parameter(-2.3*torch.ones(1))  # -2.3
        self.mask_splitter = mask_splitter.splitter(cfg)

    def forward(self, img_path, confidence, predicted_locations, predicted_ids, labels, gt_locations, gt_ids):
        """Compute classification loss and smooth l1 loss.

        Args:
            confidence (batch_size, num_priors, num_classes): class predictions.
            predicted_locations (batch_size, num_priors, 4): predicted locations.
            labels (batch_size, num_priors): real labels of all the priors.
            gt_locations (batch_size, num_priors, 4): real boxes corresponding all the priors.
        """
        num_classes = confidence.size(2)
        with torch.no_grad():
            # derived from cross_entropy=sum(log(p))
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = box_utils.hard_negative_mining(loss, labels, self.neg_pos_ratio)
            emb_mask = box_utils.hard_negative_mining(loss, labels, self.emb_neg_pos_ratio)

        confidence = confidence[mask, :] #將遮罩覆蓋到每一個cls上,由於遮罩內的值為bool,False的會直接刪除
        #print(confidence.shape) #每次的數量都不同將依據遮罩內的1為準,confidence中正負樣本比為1:self.neg_pos_ratio

        classification_loss = F.cross_entropy(confidence.view(-1, num_classes), labels[mask], reduction='sum')
        #print(confidence.view(-1, num_classes).shape, labels[mask].shape) #[len(mask), 7] [len(mask)]

        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].view(-1, 4)
        gt_locations = gt_locations[pos_mask, :].view(-1, 4)
        smooth_l1_loss = F.smooth_l1_loss(predicted_locations, gt_locations, reduction='sum')
        num_pos = gt_locations.size(0)
        #print(num_pos)#物件數量

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

        id_loss = F.cross_entropy(logits, gt_ids, ignore_index=-1, reduction='sum')

        #loss
        lbox = smooth_l1_loss / num_pos
        lconf = classification_loss / num_pos
        lid = id_loss / num_pos
        ####





        loss = torch.exp(-self.s_r)*lbox + torch.exp(-self.s_c)*lconf + torch.exp(-self.s_id)*lid + (self.s_r + self.s_c + self.s_id)
        loss *= 0.5
        nT = num_pos
        loss = loss.squeeze()

        with open('./%s/_check.txt' %self.cfg.OUTPUT_DIR, 'a') as output_log:
            s = 'lbox = %.8f'%lbox.item() + ' lconf = %.8f'%lconf.item() + ' lid = %.8f'%lid.item() + ' loss = %.8f'%loss.item() + ' s_id = %.8f'%self.s_id.item() + ' s_r = %.8f'%self.s_r.item() + ' s_c = %.8f'%self.s_c.item() + '\n'
            output_log.write(s)
        return loss, [loss.item(), lbox.item(), lconf.item(), lid.item(), nT]

