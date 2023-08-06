import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义基本的ResNet块
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1,drop_prob = 0.0):
        super(BasicBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels)
        )
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(drop_prob)
        # 如果输入通道数和输出通道数不一样，使用1x1卷积改变维度
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = self.conv(x)
        out += self.shortcut(x)
        out = self.relu(out)
        out = self.dropout(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1, drop_prob=0.0):
        super(Bottleneck, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(drop_prob),   
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )
        self.relu = nn.ReLU(inplace = True)
        self.dropout = nn.Dropout(drop_prob)
    def forward(self, x):
        residual = x
        out = self.conv(x)
        out += self.shortcut(residual)
        out = self.relu(out)
        out = self.dropout(out)
        return out
    
# 定义ResNet-18模型
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000,drop_prob=0.0):
        super(ResNet18, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self.make_layer(64, 2, stride=1,drop_prob = drop_prob)
        self.layer2 = self.make_layer(128, 2, stride=2,drop_prob = drop_prob)
        self.layer3 = self.make_layer(256, 2, stride=2,drop_prob = drop_prob)
        self.layer4 = self.make_layer(512, 2, stride=2,drop_prob = drop_prob)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)
        
    def make_layer(self, out_channels, num_blocks, stride,drop_prob = 0.0):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, drop_prob))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, drop_prob = drop_prob))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResNet152(nn.Module):
    def __init__(self, num_classes=1000, drop_prob=0.0):
        super(ResNet152, self).__init__()
        self.drop_prob = drop_prob
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3, stride=1,drop_prob = drop_prob)
        self.layer2 = self._make_layer(128, 8, stride=2,drop_prob = drop_prob)
        self.layer3 = self._make_layer(256, 36, stride=2,drop_prob = drop_prob)
        self.layer4 = self._make_layer(512, 3, stride=2,drop_prob = drop_prob)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride,drop_prob):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_channels, out_channels, stride, drop_prob=self.drop_prob))
            self.in_channels = out_channels * Bottleneck.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
class ResNet50(nn.Module):
    def __init__(self, num_classes=1000, drop_prob=0.0):
        super(ResNet50, self).__init__()
        self.drop_prob = drop_prob
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 3, stride=1,drop_prob = drop_prob)
        self.layer2 = self._make_layer(128, 4, stride=2,drop_prob = drop_prob)
        self.layer3 = self._make_layer(256, 6, stride=2,drop_prob = drop_prob)
        self.layer4 = self._make_layer(512, 3, stride=2,drop_prob = drop_prob)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * Bottleneck.expansion, num_classes)

    def _make_layer(self, out_channels, num_blocks, stride,drop_prob)   :
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Bottleneck(self.in_channels, out_channels, stride, drop_prob=self.drop_prob))
            self.in_channels = out_channels * Bottleneck.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
