import torch
import torch.nn as nn
import torch.nn.functional as F
###与基础的 InceptionModule 相比，它在每个卷积操作后增加了批归一化和激活函数，进一步提升了模型的训练稳定性和收敛速度。
#参数含义：in_channels：输入特征图的通道数 ch1x1：1x1 卷积分支的输出通道数 ch3x3red：3x3 卷积分支中，1x1 降维卷积的输出通道数 ch3x3：3x3 卷积分支中，3x3 卷积的输出通道数。
#ch5x5red：5x5 卷积分支中，1x1 降维卷积的输出通道数  ch5x5：5x5 卷积分支中，5x5 卷积的输出通道数。
class InceptionModuleBN(nn.Module):
    def __init__(self, in_channels, ch1x1, ch3x3red, ch3x3, ch5x5red, ch5x5, pool_proj):
        super(InceptionModuleBN, self).__init__()
        
        # 1x1卷积分支
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, ch1x1, kernel_size=1),
            nn.BatchNorm2d(ch1x1),# 批归一化
            nn.ReLU(inplace=True)
        )
        
        # 3x3卷积分支（先降维再卷积）
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, ch3x3red, kernel_size=1), # 1x1 降维卷积
            nn.BatchNorm2d(ch3x3red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch3x3red, ch3x3, kernel_size=3, padding=1),# 3x3 卷积
            nn.BatchNorm2d(ch3x3),
            nn.ReLU(inplace=True)#通过原地修改输入张量减少内存占用，同时保持 ReLU 激活函数的非线性特性。
        )
        
        # 5x5卷积分支
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, ch5x5red, kernel_size=1),# 1x1 降维卷积
            nn.BatchNorm2d(ch5x5red),
            nn.ReLU(inplace=True),
            nn.Conv2d(ch5x5red, ch5x5, kernel_size=5, padding=2),# 5x5 卷积
            nn.BatchNorm2d(ch5x5),
            nn.ReLU(inplace=True)
        )
        
        # 池化分支
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),# 最大池化
            nn.Conv2d(in_channels, pool_proj, kernel_size=1),# 1x1 卷积调整通道
            nn.BatchNorm2d(pool_proj),
            nn.ReLU(inplace=True)
        )
        
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        branch3 = self.branch3(x)
        branch4 = self.branch4(x)
        #最终输出的通道数为：ch1x1 + ch3x3 + ch5x5 + pool_proj
        outputs = [branch1, branch2, branch3, branch4]
        return torch.cat(outputs, 1)
