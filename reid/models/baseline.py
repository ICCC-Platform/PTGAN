import torch
import torch.nn as nn
from torchvision.models.resnet import resnet50


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        resnet = resnet50(pretrained=True)
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4,
            resnet.avgpool,
        )
        self.bottleneck = nn.BatchNorm1d(2048)
        self.bottleneck.bias.requires_grad_(False)
        self.classifier = nn.Linear(in_features=2048, out_features=575, bias=False)

    def forward(self, x):
        feature = self.backbone(x).view(x.size(0), -1)
        return feature, self.classifier(self.bottleneck(feature))
