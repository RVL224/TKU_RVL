import torch
import torch.nn as nn
import torch.nn.functional as F

from ssd.layers import L2Norm
from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url

model_urls = {
    'vgg': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
}

class FFB1(nn.Module):
    def __init__(self):
        super(FFB1, self).__init__()
        self.conv1 =  nn.Conv2d(512,512, kernel_size=(1,1), stride=1, padding=0)
        self.conv2 =  nn.Conv2d(1024,512, kernel_size=(1,1), stride=1, padding=0)
        self.upsample = nn.UpsamplingNearest2d(size=(38,38))
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x1,x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x2 = self.upsample(x2)
        x  = x1 + x2
        out  = self.relu(x)  
        return out
    
class FFB2(nn.Module):
    def __init__(self):
        super(FFB2, self).__init__()
        self.conv1 =  nn.Conv2d(1024,1024, kernel_size=(1,1), stride=1, padding=0)
        self.conv2 =  nn.Conv2d(256,1024, kernel_size=(1,1), stride=1, padding=0)
        self.upsample = nn.UpsamplingNearest2d(size=(19,19))
        self.relu = nn.ReLU(inplace=True)
     
    def forward(self,x1,x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x2 = self.upsample(x2)
        x  = x1 + x2
        out  = self.relu(x)  
        return out
class FFB3(nn.Module):
    def __init__(self):
        super(FFB3, self).__init__()
        self.conv1 =  nn.Conv2d(512,512, kernel_size=(1,1), stride=1, padding=0)
        self.conv2 =  nn.Conv2d(1024,512, kernel_size=(1,1), stride=1, padding=0)
        self.upsample = nn.UpsamplingNearest2d(size=(64,64))
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x1,x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x2 = self.upsample(x2)
        x  = x1 + x2
        out  = self.relu(x)  
        return out
    
class FFB4(nn.Module):
    def __init__(self):
        super(FFB4, self).__init__()
        self.conv1 =  nn.Conv2d(1024,1024, kernel_size=(1,1), stride=1, padding=0)
        self.conv2 =  nn.Conv2d(256,1024, kernel_size=(1,1), stride=1, padding=0)
        self.upsample = nn.UpsamplingNearest2d(size=(32,32))
        self.relu = nn.ReLU(inplace=True)
     
    def forward(self,x1,x2):
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x2 = self.upsample(x2)
        x  = x1 + x2
        out  = self.relu(x)  
        return out
    
class CFE(nn.Module):
    def __init__(self,inp):
        super(CFE, self).__init__()
        half_inp = int(inp/2)
        groups = int(inp/128)
        
#         self.conv1 =  nn.Conv2d(inp,half_inp, kernel_size=(1,1), stride=1, padding=0)
#         self.conv2 =  nn.Conv2d(half_inp, half_inp, kernel_size=(7,7), stride=1, padding=3,groups=groups)
#         self.conv3 =  nn.Conv2d(half_inp, inp, kernel_size=(1,1), stride=1, padding=0)

        self.conv_l1 =  nn.Conv2d(inp,half_inp, kernel_size=(1,1), stride=1, padding=0)
        self.conv_l2 =  nn.Conv2d(half_inp, half_inp, kernel_size=(7,1), stride=1, padding=(3,0),groups=groups)
        self.conv_l3 =  nn.Conv2d(half_inp, half_inp, kernel_size=(1,7), stride=1, padding=(0,3),groups=groups)
        self.conv_l4 =  nn.Conv2d(half_inp, half_inp, kernel_size=(1,1), stride=1, padding=0)
        
        self.conv_r1 =  nn.Conv2d(inp,half_inp, kernel_size=(1,1), stride=1, padding=0)
        self.conv_r2 =  nn.Conv2d(half_inp, half_inp, kernel_size=(1,7), stride=1, padding=(0,3),groups=groups)
        self.conv_r3 =  nn.Conv2d(half_inp, half_inp, kernel_size=(7,1), stride=1, padding=(3,0),groups=groups)
        self.conv_r4 =  nn.Conv2d(half_inp, half_inp, kernel_size=(1,1), stride=1, padding=0)
        
        self.conv5 = nn.Conv2d(inp, inp, kernel_size=(1,1), stride=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
 
       
    def forward(self,x):
        identity = x

        out_l = self.relu(self.conv_l1(x))
        out_l = self.relu(self.conv_l2(out_l))
        out_l = self.relu(self.conv_l3(out_l))
        out_l = self.relu(self.conv_l4(out_l))
        
        out_r = self.relu(self.conv_r1(x))
        out_r = self.relu(self.conv_r2(out_r))
        out_r = self.relu(self.conv_r3(out_r))
        out_r = self.relu(self.conv_r4(out_r))
        
        out = torch.cat([out_l,out_r],1)
        out = self.conv5(out)
        
        out += identity
        out = self.relu(out)
        
        return out
#         out = self.relu(self.conv1(x))
#         out = self.relu(self.conv2(out))
#         out = self.conv3(out)
        
#         out += x 
#         out = self.relu(out)
#         return out
        
      

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


class CFENET(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        size = cfg.INPUT.IMAGE_SIZE
        vgg_config = vgg_base[str(size)]
        extras_config = extras_base[str(size)]

        self.vgg = nn.ModuleList(add_vgg(vgg_config))
        self.extras = nn.ModuleList(add_extras(extras_config, i=1024, size=size))
        self.l2_norm1 = L2Norm(512, scale=20)
        self.l2_norm2 = L2Norm(1024, scale=20)
        self.cfe1 = CFE(512)
        self.cfe2 = CFE(1024)
        self.cfe3 = CFE(512)
        self.cfe4 = CFE(1024)
        if size == 300:
            self.ffb1 = FFB1()
            self.ffb2 = FFB2()
        if size == 512:
            self.ffb1 = FFB3()
            self.ffb2 = FFB4()
        
        # self.conv1 =  nn.Conv2d(512,512, kernel_size=(1,1), stride=1, padding=0)
        # self.conv2 =  nn.Conv2d(1024,512, kernel_size=(1,1), stride=1, padding=0)
        # self.conv3 =  nn.Conv2d(1024,1024, kernel_size=(1,1), stride=1, padding=0)
        # self.conv4 =  nn.Conv2d(256,1024, kernel_size=(1,1), stride=1, padding=0)
        # self.upsample1 = nn.UpsamplingNearest2d(size=(38,38))
        # self.upsample2 = nn.UpsamplingNearest2d(size=(19,19))
        self.reset_parameters()
   
    def reset_parameters(self):
        for m in self.extras.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.cfe1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.cfe2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.cfe3.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.cfe4.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.ffb1.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
        for m in self.ffb2.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
               

    def init_from_pretrain(self, state_dict):
        self.vgg.load_state_dict(state_dict)

    def forward(self, x):
        features = []
        for i in range(23):
            x = self.vgg[i](x)
        
        # Conv4_3 L2 normalization
        s1 = x
        x = self.cfe1(x)
        
        # apply vgg up to fc7
        for i in range(23, len(self.vgg)):
            x = self.vgg[i](x)
            
        s2 = x  
        x = self.cfe2(x)
        
        x = self.extras[0](x)
        s3 = x

   
        feature_1 = self.ffb1(s1,s2)
        feature_2 = self.ffb2(s2,s3)
  

        feature_1 = self.cfe3(feature_1)
        feature_2 = self.cfe4(feature_2)

        features.append(feature_1)
        features.append(feature_2)

         
        
        for i in range(1,len(self.extras)):
            x=F.relu(self.extras[i](x))
            if i % 2 == 1:
                features.append(x)

        return tuple(features)


@registry.BACKBONES.register('cfenet')
def cfenet(cfg, pretrained=True):
    model = CFENET(cfg)
    if pretrained:
        model.init_from_pretrain(load_state_dict_from_url(model_urls['vgg']))
    return model