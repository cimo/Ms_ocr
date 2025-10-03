import torch
import torch.nn as torchNN
import torch.nn.init as torchNNinit
from collections import namedtuple as collectionNamedTuple
from torchvision import models as torchvisionModel

class Vgg16Bn(torchNN.Module):
    @staticmethod
    def weightInit(moduleList):
        for module in moduleList:
            if isinstance(module, torchNN.Conv2d):
                torchNNinit.xavier_uniform_(module.weight.data)

                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, torchNN.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()
            elif isinstance(module, torchNN.Linear):
                module.weight.data.normal_(0, 0.01)
                module.bias.data.zero_()
                
    def __init__(self, pretrained, freeze):
        super(Vgg16Bn, self).__init__()

        if pretrained:
            vgg = torchvisionModel.vgg16_bn()
            
            modelHub = torch.hub.load_state_dict_from_url("http://download.pytorch.org/models/vgg16_bn-6c64b313.pth", progress=True)
            
            vgg.load_state_dict(modelHub)
            
            vggFeature = vgg.features
        else:
            vggFeature = torchvisionModel.vgg16_bn(weights=None).features

        self.slice1 = torchNN.Sequential()
        self.slice2 = torchNN.Sequential()
        self.slice3 = torchNN.Sequential()
        self.slice4 = torchNN.Sequential()
        self.slice5 = torchNN.Sequential()

        for a in range(12):
            self.slice1.add_module(str(a), vggFeature[a])

        for a in range(12, 19):
            self.slice2.add_module(str(a), vggFeature[a])

        for a in range(19, 29):
            self.slice3.add_module(str(a), vggFeature[a])

        for a in range(29, 39):
            self.slice4.add_module(str(a), vggFeature[a])

        self.slice5 = torchNN.Sequential(
                torchNN.MaxPool2d(kernel_size=3, stride=1, padding=1),
                torchNN.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6),
                torchNN.Conv2d(1024, 1024, kernel_size=1)
        )

        if not pretrained:
            Vgg16Bn.weightInit(self.slice1.modules())
            Vgg16Bn.weightInit(self.slice2.modules())
            Vgg16Bn.weightInit(self.slice3.modules())
            Vgg16Bn.weightInit(self.slice4.modules())

        Vgg16Bn.weightInit(self.slice5.modules())

        if freeze:
            for parameter in self.slice1.parameters():
                parameter.requires_grad= False

    def forward(self, modelInput):
        scoreMap = self.slice1(modelInput)
        hRelu2_2 = scoreMap

        scoreMap = self.slice2(scoreMap)
        hRelu3_2 = scoreMap

        scoreMap = self.slice3(scoreMap)
        hRelu4_3 = scoreMap

        scoreMap = self.slice4(scoreMap)
        hRelu5_3 = scoreMap

        scoreMap = self.slice5(scoreMap)
        hFc7 = scoreMap

        vgg_outputs = collectionNamedTuple("VggOutputs", ["fc7", "relu5_3", "relu4_3", "relu3_2", "relu2_2"])

        return vgg_outputs(hFc7, hRelu5_3, hRelu4_3, hRelu3_2, hRelu2_2)