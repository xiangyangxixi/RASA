# 在main.py中定义RARS网络（结合ResNet和Inception）
from torch import nn
from resnet12 import ResNet, BasicBlock
from InceptionModuleBN import InceptionModuleBN
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transform import audio_to_melspectrogram


class RARS_Net(nn.Module):
    def __init__(self, num_classes=5):  # 假设5类录制源
        super(RARS_Net, self).__init__()
        # 基础ResNet-12特征提取
        self.resnet = ResNet(BasicBlock, [2, 1, 1, 1], num_classes=num_classes)
        # 替换ResNet的layer3为Inception模块（增强多尺度特征）
        self.resnet.layer3 = InceptionModuleBN(
            in_channels=128,  # layer2输出通道为128
            ch1x1=64, ch3x3red=32, ch3x3=64,
            ch5x5red=16, ch5x5=32, pool_proj=32
        )
        # 调整后续层输入通道（Inception输出通道：64+64+32+32=192）
        self.resnet.in_planes = 192
        self.resnet.layer4 = self.resnet._make_layer(BasicBlock, 512, 1, stride=2)

    def forward(self, x):
        return self.resnet(x)

    # 自定义音频数据集
    class AudioDataset(Dataset):
        def __init__(self, audio_paths, labels):
            self.audio_paths = audio_paths
            self.labels = labels

        def __getitem__(self, idx):
            mel_spec = audio_to_melspectrogram(self.audio_paths[idx])
            return mel_spec, self.labels[idx]

        def __len__(self):
            return len(self.audio_paths)

    # 初始化模型、数据、优化器
    model = RARS_Net(num_classes=5)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # 训练循环
    for epoch in range(50):
        model.train()
        for mel_spec, label in train_loader:
            optimizer.zero_grad()
            output = model(mel_spec)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
        # 验证...