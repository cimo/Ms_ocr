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
    def _upsampleAndConcatenate(self, scoreMap, basenet):
        scoreMap = torchNNfunctional.interpolate(scoreMap, size=basenet.size()[2:], mode="bilinear", align_corners=False)

        return torch.cat([scoreMap, basenet], dim=1)

    def __init__(self, pretrained, freeze):
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
            Vgg16Bn.weight(convolution.modules())

    def forward(self, modelInput):
        basenetList = self.basenet(modelInput)

        scoreMap = torch.cat([basenetList[0], basenetList[1]], dim=1)
        scoreMap = self.upconv1(scoreMap)

        scoreMap = self._upsampleAndConcatenate(scoreMap, basenetList[2])
        scoreMap = self.upconv2(scoreMap)

        scoreMap = self._upsampleAndConcatenate(scoreMap, basenetList[3])
        scoreMap = self.upconv3(scoreMap)

        scoreMap = self._upsampleAndConcatenate(scoreMap, basenetList[4])
        feature = self.upconv4(scoreMap)

        scoreMap = self.conv_cls(feature)

        return scoreMap.permute(0, 2, 3, 1), feature
