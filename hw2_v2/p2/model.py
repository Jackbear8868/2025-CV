# ============================================================================
# File: model.py
# Date: 2025-03-11
# Author: TA
# Description: Model architecture.
# ============================================================================

import torch
import torch.nn as nn
import torchvision.models as models

class MyNet(nn.Module):
    # Squeeze-and-Excitation 模組
    class SEModule(nn.Module):
        def __init__(self, channels, reduction=4):
            super(MyNet.SEModule, self).__init__()
            self.avg_pool = nn.AdaptiveAvgPool2d(1)
            self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1)
            self.relu = nn.ReLU(inplace=True)
            self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1)
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, x):
            out = self.avg_pool(x)
            out = self.fc1(out)
            out = self.relu(out)
            out = self.fc2(out)
            out = self.sigmoid(out)
            return x * out

    # MBConv 模組（Mobile Inverted Bottleneck）
    class MBConv(nn.Module):
        def __init__(self, in_channels, out_channels, kernel_size, stride, expand_ratio, drop_connect_rate=0.0):
            super(MyNet.MBConv, self).__init__()
            self.stride = stride
            self.drop_connect_rate = drop_connect_rate
            hidden_dim = in_channels * expand_ratio
            self.use_residual = (self.stride == 1 and in_channels == out_channels)
            layers = []
            # Expansion phase (若 expand_ratio 不為 1)
            if expand_ratio != 1:
                layers.append(nn.Conv2d(in_channels, hidden_dim, kernel_size=1, bias=False))
                layers.append(nn.BatchNorm2d(hidden_dim))
                layers.append(nn.SiLU())
            # Depthwise convolution
            layers.append(nn.Conv2d(hidden_dim, hidden_dim, kernel_size=kernel_size, stride=stride,
                                    padding=kernel_size//2, groups=hidden_dim, bias=False))
            layers.append(nn.BatchNorm2d(hidden_dim))
            layers.append(nn.SiLU())
            # Squeeze and Excitation
            layers.append(MyNet.SEModule(hidden_dim, reduction=4))
            # Projection phase
            layers.append(nn.Conv2d(hidden_dim, out_channels, kernel_size=1, bias=False))
            layers.append(nn.BatchNorm2d(out_channels))
            self.block = nn.Sequential(*layers)
            
        def forward(self, x):
            out = self.block(x)
            if self.use_residual:
                out = out + x
            return out

    def __init__(self, num_classes=10, drop_connect_rate=0.2):
        super(MyNet, self).__init__()
        # 為 CIFAR-10 調整：輸入解析度 32x32
        # Stem: 原 EfficientNet-B0 stem 為 Conv3x3, stride=2，但 CIFAR-10 調整為 stride=1
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        
        # 定義 MBConv 的配置 (依據 EfficientNet-B0 標準配置，但保留原 stride)
        # 格式：(expand_ratio, out_channels, repeats, kernel_size, stride)
        cfg = [
            (1, 16, 1, 3, 1),   # Stage 1
            (6, 24, 2, 3, 2),   # Stage 2
            (6, 40, 2, 5, 2),   # Stage 3
            (6, 80, 3, 3, 2),   # Stage 4
            (6, 112, 3, 5, 1),  # Stage 5
            (6, 192, 4, 5, 2),  # Stage 6
            (6, 320, 1, 3, 1),  # Stage 7
        ]
        
        # 建立 MBConv 區塊
        in_channels = 32
        blocks = []
        total_blocks = sum([cfg[i][2] for i in range(len(cfg))])
        block_id = 0
        for expand_ratio, out_channels, repeats, kernel_size, stride in cfg:
            for i in range(repeats):
                s = stride if i == 0 else 1
                # 可設定 drop_connect_rate 分段線性遞增
                drop_rate = drop_connect_rate * block_id / total_blocks
                blocks.append(self.MBConv(in_channels, out_channels, kernel_size, s, expand_ratio, drop_connect_rate=drop_rate))
                in_channels = out_channels
                block_id += 1
        self.blocks = nn.Sequential(*blocks)
        
        # Head: Conv1x1 將通道擴展到 1280
        self.head_conv = nn.Sequential(
            nn.Conv2d(in_channels, 1280, kernel_size=1, bias=False),
            nn.BatchNorm2d(1280),
            nn.SiLU()
        )
        
        # 全局平均池化、Dropout 與全連接層
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(1280, num_classes)
        
        # 權重初始化
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # 使用 nonlinearity='linear' 得到的 gain 為 1.0，對於 SiLU 來說通常也是合適的
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='linear')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.stem(x)          # 32x32
        x = self.blocks(x)        # 依配置下採樣，例如 32->16->8->4->2...
        x = self.head_conv(x)     # 通道變為 1280
        x = self.avgpool(x)       # 變為 1x1
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x







class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        ############################################
        # NOTE:                                    #
        # Pretrain weights on ResNet18 is allowed. #
        ############################################

        # (batch_size, 3, 32, 32)
        # try to load the pretrained weights
        self.resnet = models.resnet18(weights="DEFAULT")  # Python3.8 w/ torch 2.2.1
        # self.resnet = models.resnet18(pretrained=False)  # Python3.6 w/ torch 1.10.1
        # (batch_size, 512)
        self.resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.resnet.maxpool = nn.Identity()
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, 10)
        # (batch_size, 10)

        #######################################################################
        # TODO (optional):                                                     #
        # Some ideas to improve accuracy if you can't pass the strong         #
        # baseline:                                                           #
        #   1. reduce the kernel size, stride of the first convolution layer. # 
        #   2. remove the first maxpool layer (i.e. replace with Identity())  #
        # You can run model.py for resnet18's detail structure                #
        #######################################################################


        ############################## TODO End ###############################

    def forward(self, x):
        return self.resnet(x)
    
if __name__ == '__main__':
    resnet = ResNet18()
    print("ResNet18 parameters:", sum(p.numel() for p in resnet.parameters() if p.requires_grad))

    mynet = MyNet()
    print("MyNet parameters:", sum(p.numel() for p in mynet.parameters() if p.requires_grad))
