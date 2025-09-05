import torch
import torch.nn as torchNN

# Source
from vgg16bn import Vgg16Bn

class RefineNet(torchNN.Module):
    def __init__(self):
        super(RefineNet, self).__init__()

        self.last_conv = torchNN.Sequential(
            torchNN.Conv2d(34, 64, kernel_size=3, padding=1), torchNN.BatchNorm2d(64), torchNN.ReLU(inplace=True),
            torchNN.Conv2d(64, 64, kernel_size=3, padding=1), torchNN.BatchNorm2d(64), torchNN.ReLU(inplace=True),
            torchNN.Conv2d(64, 64, kernel_size=3, padding=1), torchNN.BatchNorm2d(64), torchNN.ReLU(inplace=True)
        )

        self.aspp1 = torchNN.Sequential(
            torchNN.Conv2d(64, 128, kernel_size=3, dilation=6, padding=6), torchNN.BatchNorm2d(128), torchNN.ReLU(inplace=True),
            torchNN.Conv2d(128, 128, kernel_size=1), torchNN.BatchNorm2d(128), torchNN.ReLU(inplace=True),
            torchNN.Conv2d(128, 1, kernel_size=1)
        )

        self.aspp2 = torchNN.Sequential(
            torchNN.Conv2d(64, 128, kernel_size=3, dilation=12, padding=12), torchNN.BatchNorm2d(128), torchNN.ReLU(inplace=True),
            torchNN.Conv2d(128, 128, kernel_size=1), torchNN.BatchNorm2d(128), torchNN.ReLU(inplace=True),
            torchNN.Conv2d(128, 1, kernel_size=1)
        )

        self.aspp3 = torchNN.Sequential(
            torchNN.Conv2d(64, 128, kernel_size=3, dilation=18, padding=18), torchNN.BatchNorm2d(128), torchNN.ReLU(inplace=True),
            torchNN.Conv2d(128, 128, kernel_size=1), torchNN.BatchNorm2d(128), torchNN.ReLU(inplace=True),
            torchNN.Conv2d(128, 1, kernel_size=1)
        )

        self.aspp4 = torchNN.Sequential(
            torchNN.Conv2d(64, 128, kernel_size=3, dilation=24, padding=24), torchNN.BatchNorm2d(128), torchNN.ReLU(inplace=True),
            torchNN.Conv2d(128, 128, kernel_size=1), torchNN.BatchNorm2d(128), torchNN.ReLU(inplace=True),
            torchNN.Conv2d(128, 1, kernel_size=1)
        )

        Vgg16Bn.weightInit(self.last_conv.modules())
        Vgg16Bn.weightInit(self.aspp1.modules())
        Vgg16Bn.weightInit(self.aspp2.modules())
        Vgg16Bn.weightInit(self.aspp3.modules())
        Vgg16Bn.weightInit(self.aspp4.modules())

    def forward(self, scoreMap, feature):
        refine = torch.cat([scoreMap.permute(0,3,1,2), feature], dim=1)
        refine = self.last_conv(refine)

        aspp1 = self.aspp1(refine)
        aspp2 = self.aspp2(refine)
        aspp3 = self.aspp3(refine)
        aspp4 = self.aspp4(refine)

        out = aspp1 + aspp2 + aspp3 + aspp4

        return out.permute(0, 2, 3, 1)