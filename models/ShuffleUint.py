import torch
import torch.nn as nn
import torch.nn.functional as F

# 1D channel shuffle
def channel_shuffle(x, groups):
    batchsize, num_channels, length = x.size()  # 注意这里的维度是 batchsize, num_channels, length
    channels_per_group = num_channels // groups
    # reshape: b, num_channels, length  -->  b, groups, channels_per_group, length
    x = x.view(batchsize, groups, channels_per_group, length)
    # channelshuffle
    x = torch.transpose(x, 1, 2).contiguous()
    # flatten
    x = x.view(batchsize, -1, length)
    return x

# 修改后的 ShuffleNet 单元，处理 1D 数据
class shuffleNet_unit_1d(nn.Module):
    def __init__(self, in_channels, out_channels, stride, groups):
        super(shuffleNet_unit_1d, self).__init__()

        mid_channels = out_channels // 4
        self.stride = stride
        self.groups = groups
        # 1D Group Convolution
        self.GConv1 = nn.Sequential(
            nn.Conv1d(in_channels, mid_channels, kernel_size=1, stride=1, groups=self.groups, bias=False),
            nn.BatchNorm1d(mid_channels),
            nn.ReLU(inplace=True)
        )
        # 1D Depthwise Convolution
        self.DWConv = nn.Sequential(
            nn.Conv1d(mid_channels, mid_channels, kernel_size=3, stride=self.stride, padding=1, groups=self.groups, bias=False),
            nn.BatchNorm1d(mid_channels)
        )
        # 1D Group Convolution
        self.GConv2 = nn.Sequential(
            nn.Conv1d(mid_channels, out_channels, kernel_size=1, stride=1, groups=self.groups, bias=False),
            nn.BatchNorm1d(out_channels)
        )
        # For stride=2, apply average pooling to downsample the shortcut
        if self.stride == 2:
            self.shortcut = nn.Sequential(
                nn.AvgPool1d(kernel_size=3, stride=2, padding=1)
            )
        else:
            self.shortcut = nn.Sequential()

    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # 打印输入形状
        out = self.GConv1(x)
        # print(f"After GConv1 shape: {out.shape}")  # 打印 GConv1 后的形状
        out = channel_shuffle(out, groups=self.groups)
        # print(f"After channel shuffle shape: {out.shape}")  # 打印 channel shuffle 后的形状
        out = self.DWConv(out)
        # print(f"After DWConv shape: {out.shape}")  # 打印 DWConv 后的形状
        out = self.GConv2(out)
        # print(f"After GConv2 shape: {out.shape}")  # 打印 GConv2 后的形状
        short = self.shortcut(x)
        # print(f"Shortcut shape: {short.shape}")  # 打印 shortcut 的形状
        if self.stride == 2:
            out = F.relu(torch.cat([out, short], dim=1))  # Concatenate along the channel dimension
        else:
            out = F.relu(out + short)  # Add shortcut connection
        # print(f"Output shape: {out.shape}")  # 打印输出形状
        return out

# 定义 TripleFusionNet 加入 Dropout
class TripleFusionNet(nn.Module):
    def __init__(self, groups, drop_ratio, in_channels, hidden_channel, shufunit_out_channel, out_channels=128):
        super(TripleFusionNet, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channel, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=False),
        )
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = shuffleNet_unit_1d(in_channels=hidden_channel, out_channels=shufunit_out_channel, stride=2, groups=groups)
        # self.layer1 = shuffleNet_unit_1d(in_channels=hidden_channel, out_channels=hidden_channel, stride=2, groups=groups)
        # self.layer2 = shuffleNet_unit_1d(in_channels=hidden_channel*2, out_channels=hidden_channel*2, stride=1, groups=groups)
        # self.globalpool = nn.AdaptiveAvgPool1d(output_size=1)
        self.globalpool = nn.AvgPool1d(kernel_size=7, stride=1)
        self.fc = nn.Sequential(
            nn.Dropout(p=drop_ratio),  # Dropout
            nn.Linear((hidden_channel+shufunit_out_channel)*90, out_channels),
            nn.ReLU(inplace=False),
        )

    def forward(self, x):
        # print(f"Input shape: {x.shape}")  # 打印输入形状
        x = self.conv1(x)
        # print(f"After conv1 shape: {x.shape}")  # 打印输入形状
        x = self.maxpool(x)
        # print(f"After maxpool shape: {x.shape}")  # 打印 maxpool 后的形状
        x = self.layer1(x)
        # print(f"After layer1 shape: {x.shape}")  # 打印 layer1 后的形状
        # x = self.layer2(x)
        # print(f"After layer2 shape: {x.shape}")  # 打印 layer1 后的形状
        x = self.globalpool(x)
        # print(f"After globalpool shape: {x.shape}")  # 打印 globalpool 后的形状
        x = x.view(x.size(0), -1)
        # print(f"After flatten shape: {x.shape}")  # 打印 flatten 后的形状
        out = self.fc(x)
        # print(f"Output shape: {out.shape}")
        return out
    
    
if __name__ == "__main__":
    model = TripleFusionNet(groups=4, drop_ratio=0.5, in_channels=80, hidden_channel=320, shufunit_out_channel=320, out_channels=128)
    print(model)

    # Create a random tensor with shape (batch_size, in_channels, length)
    x = torch.randn(1, 80, 768)

    # Forward pass through the model
    def print_shape(name, tensor):
        print(f"{name} shape: {tensor.shape}")
    print_shape("Input", x)
    x = model.conv1(x)
    print_shape("After conv1", x)
    x = model.maxpool(x)
    print_shape("After maxpool", x)
    x = model.layer1(x)
    print_shape("After layer1", x)
    print('='*20)
    x = model.layer2(x)
    print_shape("After layer2", x)
    print('='*20)
    x = model.globalpool(x)
    print_shape("After globalpool", x)
    x = x.view(x.size(0), -1)
    print_shape("After flatten", x)
    out = model.fc(x)
    print_shape("Output", out)