import torch
import torch.nn as nn
import torch.nn.functional as F

from ssd.layers import L2Norm
from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url

model_urls = {
    'vgg': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
}

class CFE(nn.Module):
    def __init__(self,inp):
        super(CFE, self).__init__()
        half_inp = int(inp/2)
        self.conv_l1 =  nn.Conv2d(inp,half_inp, kernel_size=(1,1), stride=1, padding=1, bias=False)
        self.conv_l2 =  nn.Conv2d(half_inp, half_inp, kernel_size=(7,1), stride=1, padding=1,groups=8, bias=False)
        self.conv_l3 =  nn.Conv2d(half_inp, half_inp, kernel_size=(1,7), stride=1, padding=1,groups=8, bias=False)
        self.conv_l4 =  nn.Conv2d(half_inp, half_inp, kernel_size=(1,1), stride=1, padding=0, bias=False)
        
        self.conv_r1 =  nn.Conv2d(inp,half_inp, kernel_size=(1,1), stride=1, padding=1, bias=False)
        self.conv_r2 =  nn.Conv2d(half_inp, half_inp, kernel_size=(1,7), stride=1, padding=1,groups=8, bias=False)
        self.conv_r3 =  nn.Conv2d(half_inp, half_inp, kernel_size=(7,1), stride=1, padding=1,groups=8, bias=False)
        self.conv_r4 =  nn.Conv2d(half_inp, half_inp, kernel_size=(1,1), stride=1, padding=0, bias=False)
        
        self.bn_l1 = nn.BatchNorm2d(half_inp)
        self.bn_l2 = nn.BatchNorm2d(half_inp)
        self.bn_l3 = nn.BatchNorm2d(half_inp)
        self.bn_l4 = nn.BatchNorm2d(half_inp)
        
        self.bn_r1 = nn.BatchNorm2d(half_inp)
        self.bn_r2 = nn.BatchNorm2d(half_inp)
        self.bn_r3 = nn.BatchNorm2d(half_inp)
        self.bn_r4 = nn.BatchNorm2d(half_inp)
        
        self.conv5 =  nn.Conv2d(inp, inp, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn5 = nn.BatchNorm2d(inp)
        
    def forward(self,x):
        
        x1 = F.relu(self.bn_l1(self.conv_l1(x)))
        x1 = F.relu(self.bn_l2(self.conv_l2(x1)))
        x1 = F.relu(self.bn_l3(self.conv_l3(x1)))
        x1 = F.relu(self.bn_l4(self.conv_l4(x1)))
        
        x2 = F.relu(self.bn_r1(self.conv_r1(x)))
        x2 = F.relu(self.bn_r2(self.conv_r2(x2)))
        x2 = F.relu(self.bn_r3(self.conv_r3(x2)))
        x2 = F.relu(self.bn_r4(self.conv_r4(x2)))
        
        x_cat = torch.cat([x1,x2],1)
        
        x_cat = self.bn5(self.conv5(x_cat))
        
        x = F.relu(sum(x,x_cat))
       
        return x

# borrowed from https://github.com/amdegroot/ssd.pytorch/blob/master/ssd.py
def add_vgg(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'C':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
    conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
    conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
    layers += [pool5, conv6,
               nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
    return layers


def add_extras(cfg, i, size=300):
    # Extra layers added to VGG for feature scaling
    layers = []
    in_channels = i
    flag = False
    for k, v in enumerate(cfg):
        if in_channels != 'S':
            if v == 'S':
                layers += [nn.Conv2d(in_channels, cfg[k + 1], kernel_size=(1, 3)[flag], stride=2, padding=1)]
            else:
                layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
            flag = not flag
        in_channels = v
    if size == 512:
        layers.append(nn.Conv2d(in_channels, 128, kernel_size=1, stride=1))
        layers.append(nn.Conv2d(128, 256, kernel_size=4, stride=1, padding=1))
    return layers


vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}
extras_base = {
    '300': [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    '512': [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256],
}


class VGG(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        size = cfg.INPUT.IMAGE_SIZE
        vgg_config = vgg_base[str(size)]
        extras_config = extras_base[str(size)]

        self.vgg = nn.ModuleList(add_vgg(vgg_config))
        self.extras = nn.ModuleList(add_extras(extras_config, i=1024, size=size))
        self.l2_norm = L2Norm(512, scale=20)
        self.cfe1 = CFE(512)
        self.cfe2 = CFE(1024)
        self.reset_parameters()
   
    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.cfe1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
        for m in self.cfe2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)

    def init_from_pretrain(self, state_dict):
        self.vgg.load_state_dict(state_dict)

    def forward(self, x):
        features = []
        for i in range(23):
            x = self.vgg[i](x)
        s = self.l2_norm(x)  # Conv4_3 L2 normalization
        features.append(s)
        
        x = self.cfe1(x)

        # apply vgg up to fc7
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        features.append(x)
        
        x = self.cfe2(x)

        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                features.append(x)

        return tuple(features)


@registry.BACKBONES.register('vgg')
def vgg(cfg, pretrained=True):
    model = VGG(cfg)
    if pretrained:
        model.init_from_pretrain(load_state_dict_from_url(model_urls['vgg']))
    return model
