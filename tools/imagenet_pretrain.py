import torch
import torch.nn as nn
import torch.nn.functional as F
from . import model_utils

__all__ = ['pidnet_s']

BatchNorm2d = nn.BatchNorm2d
bn_mom = 0.1
algc = False

    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=False):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual

        if self.no_relu:
            return out
        else:
            return self.relu(out)

class Bottleneck(nn.Module):
    expansion = 2

    def __init__(self, inplanes, planes, stride=1, downsample=None, no_relu=True):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = BatchNorm2d(planes, momentum=bn_mom)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = BatchNorm2d(planes * self.expansion, momentum=bn_mom)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.no_relu = no_relu

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        if self.no_relu:
            return out
        else:
            return self.relu(out)


class PIDNet_S_Imgnet(nn.Module):

    def __init__(self, block=BasicBlock, layers=[2,2,3,3], num_classes=1000, planes=32, last_planes=1024):
        super(PIDNet_S_Imgnet, self).__init__()

        highres_planes = planes * 2
        self.last_planes = last_planes
        self.conv1 =  nn.Sequential(
                          nn.Conv2d(3,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                          nn.Conv2d(planes,planes,kernel_size=3, stride=2, padding=1),
                          BatchNorm2d(planes, momentum=bn_mom),
                          nn.ReLU(inplace=True),
                      )

        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, planes, planes, layers[0])
        self.layer2 = self._make_layer(block, planes, planes * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, planes * 2, planes * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, planes * 4, planes * 8, layers[3], stride=2)

        self.compression3 = nn.Sequential(
                                          nn.Conv2d(planes * 4, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )
        self.pag3 = model_utils.PagFM(highres_planes, planes)

        self.compression4 = nn.Sequential(
                                          nn.Conv2d(planes * 8, highres_planes, kernel_size=1, bias=False),
                                          BatchNorm2d(highres_planes, momentum=bn_mom),
                                          )
        self.pag4 = model_utils.PagFM(highres_planes, planes)
        

        
        

        self.layer3_ = self._make_layer(block, planes * 2, highres_planes, 2)
        #self.layer3_d = self._make_single_layer(block, planes * 2, highres_planes)

        self.layer4_ = self._make_layer(block, highres_planes, highres_planes, 2)
        #self.layer4_d = self._make_single_layer(block, highres_planes, highres_planes)

        self.layer5_ = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)
        #self.layer5_d = self._make_layer(Bottleneck, highres_planes, highres_planes, 1)

        self.layer5 =  self._make_layer(Bottleneck, planes * 8, planes * 8, 2)

        self.down5 = nn.Sequential(
                                     nn.Conv2d(planes * 4, planes * 8, kernel_size=3, stride=2, padding=1, bias=False),
                                     BatchNorm2d(planes * 8, momentum=bn_mom),
                                     nn.ReLU(inplace=True),
                                     nn.Conv2d(planes * 8, planes * 16, kernel_size=3, stride=2, padding=1, bias=False),
                                     BatchNorm2d(planes * 16, momentum=bn_mom),
                                     )        

        self.last_layer = nn.Sequential(
                                        nn.Conv2d(planes * 16, last_planes, kernel_size=1, stride=1, padding=0, bias=False),
                                        BatchNorm2d(last_planes, momentum=bn_mom),
                                        nn.ReLU(inplace=True),
                                        nn.AdaptiveAvgPool2d((1, 1)),
                                        )

        self.linear = nn.Linear(last_planes, num_classes)

        #self.bag = model_utils.BagFM(planes * 4, planes * 2, planes * 4)
        #self.dfm = model_utils.DFM(planes * 4, planes * 4)



        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)


    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        inplanes = planes * block.expansion
        for i in range(1, blocks):
            if i == (blocks-1):
                layers.append(block(inplanes, planes, stride=1, no_relu=True))
            else:
                layers.append(block(inplanes, planes, stride=1, no_relu=False))

        return nn.Sequential(*layers)
    
    def _make_single_layer(self, block, inplanes, planes, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, momentum=bn_mom),
            )

        layer = block(inplanes, planes, stride, downsample, no_relu=True)
        
        return layer

    def forward(self, x):

        x = self.conv1(x)
        x = self.layer1(x)
        x = self.relu(self.layer2(self.relu(x)))
        x_ = self.layer3_(x)
        
        x = self.relu(self.layer3(x))
        x_ = self.pag3(x_, self.compression3(x))
        
        x = self.relu(self.layer4(x))
        x_ = self.layer4_(self.relu(x_))
        x_ = self.pag4(x_, self.compression4(x))

            
        x_ = self.layer5_(self.relu(x_))
        x = self.layer5(x)
        
        x = x + self.down5(self.relu(x_))
        
        x = self.last_layer(self.relu(x))
        x = x.view(-1, self.last_planes)
        x = self.linear(x)

        return x

def pidnet_s():
    model = PIDNet_S_Imgnet()
    return model

if __name__ == '__main__':

    x = torch.randn(2, 3, 224, 224).to("cuda:0")
    net = pidnet_s().to("cuda:0")
    num_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    net.eval()
    a = net(x)