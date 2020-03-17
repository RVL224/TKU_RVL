
class splitter():
    def __init__(self, cfg):
        self.anchor_size = cfg.MODEL.PRIORS.BOXES_PER_LOCATION
        self.feature_size = cfg.MODEL.PRIORS.FEATURE_MAPS

    def con_split(self, mask):
        mask = mask.view(mask.shape[0],-1)
        k = 0
        mask_list = []
        for i, j in zip(self.feature_size,self.anchor_size):
            small_mask = mask[:, k:k+i**2*j].view(mask.shape[0], i, i, j)
            mask_list.append(small_mask)
            k += i**2*j
        return(mask_list)

    def split(self, mask):
        mask = mask.view(mask.shape[0],-1)
        k = 0
        mask_list = []
        for i, j in zip(self.feature_size,self.anchor_size):
            small_mask = mask[:, k:k+i**2*j].view(mask.shape[0], i, i, j)
            mask_list.append(small_mask)
            k += i**2*j
        return(mask_list)




