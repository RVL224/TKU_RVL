import torch
import torch.nn as nn
import torch.nn.functional as F

from .nn_utils import *
from ssd.layers import L2Norm
from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url

model_urls = {
    'vgg': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
}


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

vgg_base = {
    '300': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
    '512': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
            512, 512, 512],
}

class VGG_M2det(nn.Module):
    def __init__(self,cfg):
        super().__init__()
        size = 512
        vgg_config = vgg_base[str(size)]

        self.vgg = nn.ModuleList(add_vgg(vgg_config))
        self.l2_norm = L2Norm(2048, scale=20)
        self.num_levels = 8
        self.num_scales = 6
        self.smooth = True
        self.reduce= BasicConv(512, 256, kernel_size=3, stride=1, padding=1)
        self.up_reduce= BasicConv(1024, 512, kernel_size=1, stride=1)
        self.sfam_module = SFAM(256, self.num_levels, self.num_scales, compress_ratio=16)
        self.leach = nn.ModuleList([BasicConv(
                    768,
                    128,
                    kernel_size=(1,1),stride=(1,1))]*self.num_levels)
        self.reset_parameters()
        self.num_levels = 8
        self.num_scales = 6
        self.smooth = True
        for i in range(self.num_levels):
            if i == 0:
                setattr(self,
                        'unet{}'.format(i+1),
                        TUM(first_level=True, 
                            input_planes=128, 
                            is_smooth=self.smooth,
                            scales=self.num_scales,
                            side_channel=512)) #side channel isn't fixed.
            else:
                setattr(self,
                        'unet{}'.format(i+1),
                        TUM(first_level=False, 
                            input_planes=128, 
                            is_smooth=self.smooth, 
                            scales=self.num_scales,
                            side_channel=256))
        self.reset_parameters()
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)
                
    def init_from_pretrain(self, state_dict):
        self.vgg.load_state_dict(state_dict)

    def forward(self, x):
        features = []
        for i in range(23):
            x = self.vgg[i](x)
        s = x  # Conv4_3 L2 normalization
        features.append(s)

        # apply vgg up to fc7
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
        features.append(x)
        x = torch.cat((self.reduce(features[0]), F.interpolate(self.up_reduce(features[1]),scale_factor=2,mode='nearest')),1)
        tum_outs = [getattr(self, 'unet{}'.format(1))(self.leach[0](x), 'none')]
        for i in range(1,self.num_levels,1):
            tum_outs.append(
                    getattr(self, 'unet{}'.format(i+1))(
                        self.leach[i](x), tum_outs[i-1][-1]
                        )
                    )
        sources = [torch.cat([_fx[i-1] for _fx in tum_outs],1) for i in range(self.num_scales, 0, -1)]
        
        sources = self.sfam_module(sources)
        sources[0] = self.l2_norm(sources[0])
        
        return sources
    
@registry.BACKBONES.register('vgg_M2det')
def vgg_M2det(cfg, pretrained=True):
    model = VGG_M2det(cfg)
    if pretrained:
        model.init_from_pretrain(load_state_dict_from_url(model_urls['vgg']))
    return model