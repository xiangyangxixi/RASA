# 在main.py中定义RARS网络（结合ResNet和Inception）
from torch import nn
from resnet12 import ResNet, BasicBlock
from InceptionModuleBN import InceptionModuleBN
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transform import audio_to_melspectrogram


