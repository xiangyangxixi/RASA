import torch
import torch.nn as nn
import torch.nn.functional as F
## 基础残差块BasicBlock:适用于较浅的 ResNet（如 ResNet-18、ResNet-34）
class BasicBlock(nn.Module):
    # expansion = 1：表示输出通道数与中间卷积层通道数相同
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        # downsample参数：当输入输出通道数不匹配或需要下采样时，用于调整输入维度
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        # 包含两个 3×3 卷积层，中间通过 BatchNorm 和 ReLU 激活函数连接
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        # 残差连接：将输入（identity）与卷积结果相加，解决梯度消失问题
        ###残差连接：通过out += identity实现，允许梯度直接传播，解决深层网络训练难题
        out += identity
        out = F.relu(out)

        return out


## 瓶颈残差块Bottleneck:用于更深的网络（如 ResNet-50、ResNet-101）
class Bottleneck(nn.Module):
    # expansion = 4：表示输出通道数是中间卷积层通道数的 4 倍
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        # 包含 1×1、3×3、1×1 三个卷积层，形成 "瓶颈" 结构（减少计算量）
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion,
                               kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = F.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)

        return out


## 主ResNet模型,这是整个网络的主体结构
class ResNet(nn.Module):
    # 初始化方法（init）
    # block：指定使用 BasicBlock 还是 Bottleneck,layers：列表，指定每个残差层包含的残差块数量,num_classes：分类任务的类别数
    ###zero_init_residual：是否将残差分支最后一个 BN 层初始化为 0（论文中的技巧，有助于模型收敛）
    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        # 初始卷积层(提取初始特征)
        # 7×7 卷积层（64 通道，步长 2，填充 3）：初步提取图像特征
        # 初始卷积层（Conv1）：1 层
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # BatchNorm2d：批标准化，加速训练
        self.bn1 = nn.BatchNorm2d(64)
        # ReLU：激活函数
        self.relu = nn.ReLU(inplace=True)
        # 3×3 最大池化（步长 2）：下采样，减少特征图尺寸
        # 最大池化层（MaxPool）：1 层
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # 残差层（layer1、layer2、layer3、layer4）：每个残差层由若干残差块（BasicBlock 或 Bottleneck）组成
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        # 分类器
        # AdaptiveAvgPool2d：自适应平均池化，将特征图转为 1×1
        # 平均池化层（AvgPool）：1 层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层（fc）：将特征映射到类别空间
        # 全连接层（FC）：1 层
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 权重初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # 零初始化残差分支的最后一个BN层（论文中的技巧）
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    ##残差层构建（_make_layer 方法）
    def _make_layer(self, block, planes, blocks, stride=1):
        # 动态创建残差层，每个残差层由多个残差块组成
        downsample = None
        # 当需要改变特征图尺寸（stride≠1）或通道数时，通过downsample调整输入维度
        ###下采样策略：通过步长 > 1 的卷积或downsample模块实现，逐步减少特征图尺寸
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = []
        # 第一个残差块可能包含下采样，后续残差块保持相同维度
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


## 真正的ResNet-12: 11个卷积层 + 1个全连接层 = 12层
def resnet12(num_classes=1000, **kwargs):
    """
    ResNet-12:
    - 初始conv: 1层
    - BasicBlock: 5个BasicBlock × 2个卷积层 = 10层
    - 全连接: 1层
    - 总计: 1 + 10 + 1 = 12层
    """
    return ResNet(BasicBlock, [2, 1, 1, 1], num_classes=num_classes, **kwargs)#Layer1 (64通道) 是特征提取的起点，增加其深度有助于基础特征学习