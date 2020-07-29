from torch import nn
import torch
from ssd.modeling import registry
from ssd.utils.model_zoo import load_state_dict_from_url

model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth',
}
class FFB1(nn.Module):
    def __init__(self):
        super(FFB1, self).__init__()
        self.conv1 =  nn.Conv2d(96,96, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.conv2 =  nn.Conv2d(1280,96, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(96)
        self.bn2 = nn.BatchNorm2d(96)
        self.upsample = nn.UpsamplingNearest2d(size=(32,32))
        self.relu = nn.ReLU(inplace=True)

    def forward(self,x1,x2):
        x1 = self.bn1(self.conv1(x1))
        x2 = self.bn2(self.conv2(x2))
        x2 = self.upsample(x2)
        x  = x1 + x2
        out  = self.relu(x)  
        return out
    
class FFB2(nn.Module):
    def __init__(self):
        super(FFB2, self).__init__()
        self.conv1 =  nn.Conv2d(1280,1280, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.conv2 =  nn.Conv2d(512,1280, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(1280)
        self.bn2 = nn.BatchNorm2d(1280)
        self.upsample = nn.UpsamplingNearest2d(size=(16,16))
        self.relu = nn.ReLU(inplace=True)
     
    def forward(self,x1,x2):
        x1 = self.bn1(self.conv1(x1))
        x2 = self.bn2(self.conv2(x2))
        x2 = self.upsample(x2)
        x  = x1 + x2
        out  = self.relu(x)  
        return out
    
class CFE(nn.Module):
    def __init__(self,inp):
        super(CFE, self).__init__()
        half_inp = int(inp/2)

        self.conv_l1 =  nn.Conv2d(inp,half_inp, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.conv_l2 =  nn.Conv2d(half_inp, half_inp, kernel_size=(7,1), stride=1, padding=(3,0),groups=8, bias=False)
        self.conv_l3 =  nn.Conv2d(half_inp, half_inp, kernel_size=(1,7), stride=1, padding=(0,3),groups=8, bias=False)
        self.conv_l4 =  nn.Conv2d(half_inp, half_inp, kernel_size=(1,1), stride=1, padding=0, bias=False)
        
        self.conv_r1 =  nn.Conv2d(inp,half_inp, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.conv_r2 =  nn.Conv2d(half_inp, half_inp, kernel_size=(1,7), stride=1, padding=(0,3),groups=8, bias=False)
        self.conv_r3 =  nn.Conv2d(half_inp, half_inp, kernel_size=(7,1), stride=1, padding=(3,0),groups=8, bias=False)
        self.conv_r4 =  nn.Conv2d(half_inp, half_inp, kernel_size=(1,1), stride=1, padding=0, bias=False)
        
        self.conv5 = nn.Conv2d(inp, inp, kernel_size=(1,1), stride=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        
        self.bn_l1 = nn.BatchNorm2d(half_inp)
        self.bn_l2 = nn.BatchNorm2d(half_inp)
        self.bn_l3 = nn.BatchNorm2d(half_inp)
        self.bn_l4 = nn.BatchNorm2d(half_inp)
        
        self.bn_r1 = nn.BatchNorm2d(half_inp)
        self.bn_r2 = nn.BatchNorm2d(half_inp)
        self.bn_r3 = nn.BatchNorm2d(half_inp)
        self.bn_r4 = nn.BatchNorm2d(half_inp)
        
        self.bn_5 = nn.BatchNorm2d(inp)

    def forward(self,x):
        identity = x

        out_l = self.relu(self.bn_l1(self.conv_l1(x)))
        out_l = self.relu(self.bn_l2(self.conv_l2(out_l)))
        out_l = self.relu(self.bn_l3(self.conv_l3(out_l)))
        out_l = self.relu(self.bn_l4(self.conv_l4(out_l)))
        
        out_r = self.relu(self.bn_r1(self.conv_r1(x)))
        out_r = self.relu(self.bn_r2(self.conv_r2(out_r)))
        out_r = self.relu(self.bn_r3(self.conv_r3(out_r)))
        out_r = self.relu(self.bn_r4(self.conv_r4(out_r)))
        
        out = torch.cat([out_l,out_r],1)
        out = self.bn_5(self.conv5(out))
        
        out += identity
        out = self.relu(out)
        
        return out

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, groups=1):
        padding = (kernel_size - 1) // 2
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        hidden_dim = int(round(inp * expand_ratio))
        self.use_res_connect = self.stride == 1 and inp == oup

        layers = []
        if expand_ratio != 1:
            # pw
            layers.append(ConvBNReLU(inp, hidden_dim, kernel_size=1))
        layers.extend([
            # dw
            ConvBNReLU(hidden_dim, hidden_dim, stride=stride, groups=hidden_dim),
            # pw-linear
            nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class Mobile_CFENET(nn.Module):
    def __init__(self, width_mult=1.0, inverted_residual_setting=None):
        super(Mobile_CFENET, self).__init__()
        block = InvertedResidual
        input_channel = 32
        last_channel = 1280
        self.cfe1 = CFE(96)
        self.cfe2 = CFE(1280)
        self.cfe3 = CFE(96)
        self.cfe4 = CFE(1280)
        self.ffb1 = FFB1()
        self.ffb2 = FFB2()
        
        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1],
            ]

        # only check the first element, assuming user knows t,c,n,s are required
        if len(inverted_residual_setting) == 0 or len(inverted_residual_setting[0]) != 4:
            raise ValueError("inverted_residual_setting should be non-empty "
                             "or a 4-element list, got {}".format(inverted_residual_setting))

        # building first layer
        input_channel = int(input_channel * width_mult)
        self.last_channel = int(last_channel * max(1.0, width_mult))
        features = [ConvBNReLU(3, input_channel, stride=2)]
        # building inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            output_channel = int(c * width_mult)
            for i in range(n):
                stride = s if i == 0 else 1
                features.append(block(input_channel, output_channel, stride, expand_ratio=t))
                input_channel = output_channel
        # building last several layers
        features.append(ConvBNReLU(input_channel, self.last_channel, kernel_size=1))
        # make it nn.Sequential
        self.features = nn.Sequential(*features)
        self.extras = nn.ModuleList([
            InvertedResidual(1280, 512, 2, 0.2),
            InvertedResidual(512, 256, 2, 0.25),
            InvertedResidual(256, 256, 2, 0.5),
            InvertedResidual(256, 64, 2, 0.25)
        ])

        self.reset_parameters()

    def reset_parameters(self):
        # weight initialization
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

    def forward(self, x):
        features = []
        for i in range(14):
            x = self.features[i](x)
            
        s1 = x
        x = self.cfe1(x)
        
        for i in range(14, len(self.features)):
            x = self.features[i](x)
            
        s2 = x
        x = self.cfe2(x)
        
        x = self.extras[0](x)
        s3 = x
        
        feature1 = self.ffb1(s1,s2)
        feature2 = self.ffb2(s2,s3)
        
        feature1 = self.cfe3(feature1)
        feature2 = self.cfe4(feature2)
        
        features.append(feature1)
        features.append(feature2)
        features.append(x)
        for i in range(1,len(self.extras)):
            x = self.extras[i](x)
            features.append(x)

        return tuple(features)


@registry.BACKBONES.register('mobile_cfenet')
def mobile_cfenet(cfg, pretrained=False):
    model = Mobile_CFENET()
    if pretrained:
        model.load_state_dict(load_state_dict_from_url(model_urls['mobilenet_v2']), strict=False)
    return model
