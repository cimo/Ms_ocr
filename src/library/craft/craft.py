import torch
import torch.nn as torchNN
import torch.nn.functional as torchNNfunctional

# Source
from vgg16bn import Vgg16Bn

class _DoubleConvolution(torchNN.Module):
    def __init__(self, channelIn, channelMid, channelOut):
        super(_DoubleConvolution, self).__init__()
        
        self.conv = torchNN.Sequential(
            torchNN.Conv2d(channelIn + channelMid, channelMid, kernel_size=1),
            torchNN.BatchNorm2d(channelMid),
            torchNN.ReLU(inplace=True),
            
            torchNN.Conv2d(channelMid, channelOut, kernel_size=3, padding=1),
            torchNN.BatchNorm2d(channelOut),
            torchNN.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class Craft(torchNN.Module):
    def _upsampleAndConcatenate(self, x, source):
        x = torchNNfunctional.interpolate(x, size=source.size()[2:], mode="bilinear", align_corners=False)

        return torch.cat([x, source], dim=1)

    def __init__(self, pretrained=False, freeze=False):
        super(Craft, self).__init__()

        self.basenet = Vgg16Bn(pretrained, freeze)

        self.upconv1 = _DoubleConvolution(1024, 512, 256)
        self.upconv2 = _DoubleConvolution(512, 256, 128)
        self.upconv3 = _DoubleConvolution(256, 128, 64)
        self.upconv4 = _DoubleConvolution(128, 64, 32)

        classNumber = 2

        self.conv_cls = torchNN.Sequential(
            torchNN.Conv2d(32, 32, kernel_size=3, padding=1), torchNN.ReLU(inplace=True),
            torchNN.Conv2d(32, 32, kernel_size=3, padding=1), torchNN.ReLU(inplace=True),
            torchNN.Conv2d(32, 16, kernel_size=3, padding=1), torchNN.ReLU(inplace=True),
            torchNN.Conv2d(16, 16, kernel_size=1), torchNN.ReLU(inplace=True),
            torchNN.Conv2d(16, classNumber, kernel_size=1),
        )

        for convolution in [self.upconv1, self.upconv2, self.upconv3, self.upconv4, self.conv_cls]:
            Vgg16Bn.init_weights(convolution.modules())

    def forward(self, x):
        basenetList = self.basenet(x)

        y = torch.cat([basenetList[0], basenetList[1]], dim=1)
        y = self.upconv1(y)

        y = self._upsampleAndConcatenate(y, basenetList[2])
        y = self.upconv2(y)

        y = self._upsampleAndConcatenate(y, basenetList[3])
        y = self.upconv3(y)

        y = self._upsampleAndConcatenate(y, basenetList[4])
        feature = self.upconv4(y)

        y = self.conv_cls(feature)

        return y.permute(0, 2, 3, 1), feature
