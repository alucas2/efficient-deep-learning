import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_filters=[64, 128, 256, 512], num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = num_filters[0]

        self.conv1 = nn.Conv2d(3, num_filters[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(num_filters[0])
        
        self.final_image_size = 32
        conv_layers = []
        for i in range(len(num_blocks)):
            stride = 1 if i == 0 else 2
            conv_layers.append(self._make_layer(block, num_filters[i], num_blocks[i], stride=stride))
            self.final_image_size //= stride

        self.conv_layers = nn.Sequential(*conv_layers)
        self.linear = nn.Linear(num_filters[-1] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.conv_layers(out)
        out = F.avg_pool2d(out, self.final_image_size)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def make_resnet18(num_classes):
    return ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_filters=[64, 128, 256, 512], num_classes=num_classes)

def make_thinresnet18(num_classes):
    return ResNet(BasicBlock, num_blocks=[2, 2, 2, 2], num_filters=[32, 64, 128, 256], num_classes=num_classes)

def make_resnet20(num_classes):
    return ResNet(BasicBlock, num_blocks=[3, 3, 3], num_filters=[64, 128, 256], num_classes=num_classes)