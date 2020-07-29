import torch
import torch.nn as nn
import torch.nn.functional as F

from ssd.layers import L2Norm
from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url

model_urls = {
    'vgg': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
}

class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=True) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x
class CFEM(nn.Module):
    def __init__(self, in_planes, out_planes, stride = 1, scale = 0.1, groups=8, thinning=2, k = 7, dilation=1):
        super(CFEM, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        second_in_planes = in_planes // thinning

        p = (k-1)//2
        self.cfem_a = list()
        self.cfem_a += [BasicConv(in_planes, in_planes, kernel_size = (1,k), stride = 1, padding = (0,p), groups = groups, relu = False)]
        self.cfem_a += [BasicConv(in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_a += [BasicConv(second_in_planes, second_in_planes, kernel_size=3, stride=stride, padding=dilation, groups = 4, dilation=dilation)]
        self.cfem_a += [BasicConv(second_in_planes, second_in_planes, kernel_size = (k, 1), stride = 1, padding = (p, 0), groups = groups, relu = False)]
        self.cfem_a += [BasicConv(second_in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_a = nn.ModuleList(self.cfem_a)

        self.cfem_b = list()
        self.cfem_b += [BasicConv(in_planes, in_planes, kernel_size = (k,1), stride = 1, padding = (p,0), groups = groups, relu = False)]
        self.cfem_b += [BasicConv(in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_b += [BasicConv(second_in_planes, second_in_planes, kernel_size = 3, stride=stride, padding=dilation,groups =4,dilation=dilation)]
        self.cfem_b += [BasicConv(second_in_planes, second_in_planes, kernel_size = (1, k), stride = 1, padding = (0, p), groups = groups, relu = False)]
        self.cfem_b += [BasicConv(second_in_planes, second_in_planes, kernel_size = 1, stride = 1)]
        self.cfem_b = nn.ModuleList(self.cfem_b)


        self.ConvLinear = BasicConv(2 * second_in_planes, out_planes, kernel_size = 1, stride = 1, relu = False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size = 1, stride = stride, relu = False)
        self.relu = nn.ReLU(inplace = False)

    def forward(self,x):

        x1 = self.cfem_a[0](x)
        x1 = self.cfem_a[1](x1)
        x1 = self.cfem_a[2](x1)
        x1 = self.cfem_a[3](x1)
        x1 = self.cfem_a[4](x1)

        x2 = self.cfem_b[0](x)
        x2 = self.cfem_b[1](x2)
        x2 = self.cfem_b[2](x2)
        x2 = self.cfem_b[3](x2)
        x2 = self.cfem_b[4](x2)

        out = torch.cat([x1, x2], 1)

        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out

def get_CFEM(cfe_type='large', in_planes=512, out_planes=512, stride=1, scale=1, groups=8, dilation=1):
    assert cfe_type in ['large', 'normal', 'light'], 'no that type of CFEM'
    if cfe_type == 'large':
        return CFEM(in_planes, out_planes, stride=stride, scale=scale, groups=groups, dilation=dilation, thinning=2)
    elif cfe_type == 'normal':
        return CFEM(in_planes, out_planes, stride=stride, scale=scale, groups=groups, dilation=dilation, thinning=4)
    else:
        return CFEM(in_planes, out_planes, stride=stride, scale=scale, groups=groups, dilation=dilation, thinning=8)
    

      

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


class CFENET_v2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        size = cfg.INPUT.IMAGE_SIZE
        vgg_config = vgg_base[str(size)]
        extras_config = extras_base[str(size)]

        self.vgg = nn.ModuleList(add_vgg(vgg_config))
        self.extras = nn.ModuleList(add_extras(extras_config, i=1024, size=size))
        self.l2_norm1 = L2Norm(512, scale=20)
        
        self.laterial1 = get_CFEM(cfe_type='large', in_planes=512, out_planes=512, stride=1, scale=1, groups=8, dilation=2)
        self.laterial2 = get_CFEM(cfe_type='large', in_planes=1024, out_planes=1024, stride=1, scale=1, groups=8, dilation=2)
        
        self.arterial1 = get_CFEM(cfe_type='normal',in_planes=512, out_planes=512, stride=1, scale=1, groups=8)
        self.arterial2 = get_CFEM(cfe_type='normal',in_planes=1024, out_planes=1024, stride=1, scale=1, groups=8)
        
        self.reduce1 = BasicConv(512, 256, kernel_size = 3, stride = 1, padding = 1)
        self.up_reduce1 = BasicConv(1024, 256, kernel_size =1)
        self.reduce2 = BasicConv(1024, 512, kernel_size =1)
        self.up_reduce2 = BasicConv(256, 512, kernel_size =1)
                        
        self.reset_parameters()
   
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
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
        
        # Conv4_3 L2 normalization
        s1 = x
        x = self.arterial1(x)
        
        # apply vgg up to fc7
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
            
        s2 = x  
        x = self.arterial2(x)
        
        x = self.extras[0](x)
        s3 = x

   
        t1 = self.reduce1(s1)
        t2 = self.reduce2(s2)
        
        up1 = F.upsample(self.up_reduce1(s2),scale_factor=2,mode='bilinear')
        up2 = self.up_reduce2(s3)
        
#         print(up1.shape)
#         print(up2.shape)
#         print(t1.shape)
#         print(t2.shape)
        
        feature_1 = torch.cat((t1,up1), 1)
        feature_2 = torch.cat((t2,up2), 1)

        feature_1 = self.laterial1(feature_1)
        feature_2 = self.laterial2(feature_2)

        features.append(feature_1)
        features.append(feature_2)

         
        
        for i in range(1,len(self.extras)):
            x=F.relu(self.extras[i](x))
            if i % 2 == 1:
                features.append(x)

        return tuple(features)


@registry.BACKBONES.register('cfenet_v2')
def cfenet_v2(cfg, pretrained=True):
    model = CFENET_v2(cfg)
    if pretrained:
        model.init_from_pretrain(load_state_dict_from_url(model_urls['vgg']))
    return model