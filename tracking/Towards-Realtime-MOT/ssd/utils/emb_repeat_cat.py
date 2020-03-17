import torch

class emb_repeat_cat():
    def __init__(self, cfg):
        self.anchor_size = cfg.MODEL.PRIORS.BOXES_PER_LOCATION
        self.cfg = cfg

    def repeat_cat(self, id_preds):
        id_pred = []
        for idpred, anchor in zip(id_preds, self.anchor_size):
            batch_size = idpred.shape[0]
            idpred = idpred.repeat(1, 1, 1, anchor, 1).contiguous()
            id_pred.append(idpred)
        id_pred = torch.cat([i.view(i.shape[0], -1) for i in id_pred], dim=1).view(batch_size, -1, self.cfg.MODEL.EMB_DIM)

        return id_pred






