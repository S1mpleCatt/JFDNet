import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiJointModel(nn.Module):
    def __init__(self, joints=6, axes=3):
        super().__init__()
        self.joints = joints
        self.axes = axes

    def split(self, x):
        expected = self.joints * self.axes
        if x.ndim != 3 or x.size(1) != expected:
            raise ValueError(f"Expected input shape [batch, {expected}, time]")
        return x.view(x.size(0), self.joints, self.axes, x.size(-1))


class WaveNetActivation(nn.Module):
    def __init__(self, dropout=0.05):
        super().__init__()
        self.dropout = nn.Dropout1d(dropout)

    def forward(self, x):
        return self.dropout(torch.tanh(x) * torch.sigmoid(x))


class CBAM1D(nn.Module):
    def __init__(self, channels, reduction=0.5):
        super().__init__()
        hidden = max(1, int(channels * reduction))
        self.first = nn.Linear(channels, hidden)
        self.second = nn.Linear(hidden, channels)
        self.temporal = nn.Conv1d(2, 1, kernel_size=3, padding=1)

    def channel_map(self, x):
        return self.second(F.relu(self.first(x)))

    def forward(self, x):
        average = x.mean(dim=2)
        maximum = x.amax(dim=2)
        channel = (self.channel_map(average) + self.channel_map(maximum)).unsqueeze(2)
        x = x * channel
        pooled = torch.cat((x.mean(dim=1, keepdim=True), x.amax(dim=1, keepdim=True)), dim=1)
        return x * torch.sigmoid(self.temporal(pooled))


class MultiScaleResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels=16, dilation=1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, 16, kernel_size=64, stride=8, padding=31),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(16, out_channels, kernel_size=3, padding=dilation, dilation=dilation),
            WaveNetActivation(),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.network(x)


class WDCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Tanh(),
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.MaxPool1d(2),
        )

    def forward(self, x):
        return self.network(x)


class MJARBranch(nn.Module):
    def __init__(self, axes=3, channels=32):
        super().__init__()
        self.first = nn.Conv1d(axes, 8, kernel_size=16, padding=7)
        self.second = nn.Conv1d(8, channels, kernel_size=3, padding=1)
        self.residual = nn.Conv1d(axes, channels, kernel_size=1)
        self.activation = WaveNetActivation()

    def forward(self, x):
        output = F.leaky_relu(self.first(x), 0.2, inplace=True)
        output = F.leaky_relu(self.second(output), 0.2, inplace=True)
        residual = self.residual(x)
        length = min(output.size(-1), residual.size(-1))
        return self.activation(output[..., :length] + residual[..., :length])


class MJAR(MultiJointModel):
    def __init__(self, joints=6, axes=3):
        super().__init__(joints, axes)
        self.branches = nn.ModuleList([MJARBranch(axes) for _ in range(joints)])
        width = 32 * joints
        self.first_attention = CBAM1D(width)
        self.multiscale = nn.ModuleList([MultiScaleResidualBlock(width, dilation=d) for d in (1, 2, 3)])
        self.second_attention = CBAM1D(48)
        self.detector = nn.Sequential(WDCNNBlock(48, 64), WDCNNBlock(64, 32), WDCNNBlock(32, 32))
        self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(joints)])

    def forward(self, x):
        joints = self.split(x)
        private = [branch(joints[:, index]) for index, branch in enumerate(self.branches)]
        length = min(item.size(-1) for item in private)
        private = [item[..., :length] for item in private]
        combined = self.first_attention(torch.cat(private, dim=1))
        global_map = torch.cat([block(combined) for block in self.multiscale], dim=1)
        global_vector = self.detector(self.second_attention(global_map)).mean(dim=2)
        return torch.cat(
            [head(feature.mean(dim=2) + global_vector) for head, feature in zip(self.heads, private)], dim=1
        )


class CNNLSTMBranch(nn.Module):
    def __init__(self, axes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(axes, 32, kernel_size=7, padding=3),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.network(x)


class CNNLSTM(MultiJointModel):
    def __init__(self, joints=6, axes=3, hidden=64):
        super().__init__(joints, axes)
        self.branches = nn.ModuleList([CNNLSTMBranch(axes) for _ in range(joints)])
        self.lstm = nn.LSTM(32 * joints, hidden, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(2 * hidden, 32)
        self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(joints)])

    def forward(self, x):
        joints = self.split(x)
        private = [branch(joints[:, index]) for index, branch in enumerate(self.branches)]
        length = min(item.size(-1) for item in private)
        private = [item[..., :length] for item in private]
        sequence = torch.cat(private, dim=1).transpose(1, 2)
        global_vector = self.projection(self.lstm(sequence)[0].mean(dim=1))
        return torch.cat(
            [head(feature.mean(dim=2) + global_vector) for head, feature in zip(self.heads, private)], dim=1
        )


class TICNNBranch(nn.Module):
    def __init__(self, axes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(axes, 32, kernel_size=64, stride=2, padding=31),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=16, padding=8),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.network(x)


class TICNN(MultiJointModel):
    def __init__(self, joints=6, axes=3):
        super().__init__(joints, axes)
        self.branches = nn.ModuleList([TICNNBranch(axes) for _ in range(joints)])
        self.global_network = nn.Sequential(
            nn.Conv1d(32 * joints, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )
        self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(joints)])

    def forward(self, x):
        joints = self.split(x)
        private = [branch(joints[:, index]) for index, branch in enumerate(self.branches)]
        length = min(item.size(-1) for item in private)
        private = [item[..., :length] for item in private]
        global_vector = self.global_network(torch.cat(private, dim=1)).mean(dim=2)
        return torch.cat(
            [head(feature.mean(dim=2) + global_vector) for head, feature in zip(self.heads, private)], dim=1
        )


class MixedScaleBranch(nn.Module):
    def __init__(self, axes=3):
        super().__init__()
        self.kernel_three = nn.Conv1d(axes, 16, kernel_size=3, padding=1)
        self.kernel_five = nn.Conv1d(axes, 16, kernel_size=5, padding=2)
        self.fusion = nn.Sequential(nn.ReLU(inplace=True), nn.Conv1d(32, 32, kernel_size=1), nn.ReLU(inplace=True))
        self.tail = nn.Sequential(nn.Conv1d(32, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True), nn.MaxPool1d(2))

    def forward(self, x):
        return self.tail(self.fusion(torch.cat((self.kernel_three(x), self.kernel_five(x)), dim=1)))


class MSMCNN(MultiJointModel):
    def __init__(self, joints=6, axes=3):
        super().__init__(joints, axes)
        self.branches = nn.ModuleList([MixedScaleBranch(axes) for _ in range(joints)])
        width = 32 * joints
        self.global_convolutions = nn.ModuleList(
            [nn.Conv1d(width, 32, kernel_size=3, padding=d, dilation=d) for d in (1, 2, 3)]
        )
        self.fusion = nn.Sequential(nn.ReLU(inplace=True), nn.Conv1d(96, 32, kernel_size=1), nn.ReLU(inplace=True))
        self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(joints)])

    def forward(self, x):
        joints = self.split(x)
        private = [branch(joints[:, index]) for index, branch in enumerate(self.branches)]
        length = min(item.size(-1) for item in private)
        private = [item[..., :length] for item in private]
        concatenated = torch.cat(private, dim=1)
        global_vector = self.fusion(torch.cat([layer(concatenated) for layer in self.global_convolutions], dim=1)).mean(dim=2)
        return torch.cat(
            [head(feature.mean(dim=2) + global_vector) for head, feature in zip(self.heads, private)], dim=1
        )


class MLP(MultiJointModel):
    def __init__(self, joints=6, axes=3, bins=32):
        super().__init__(joints, axes)
        self.bins = bins
        self.pool = nn.AdaptiveAvgPool1d(bins)
        self.heads = nn.ModuleList([nn.Linear(axes * bins, 1) for _ in range(joints)])

    def forward(self, x):
        joints = self.split(x)
        return torch.cat(
            [head(self.pool(joints[:, index]).flatten(1)) for index, head in enumerate(self.heads)], dim=1
        )


class CNNBranch(nn.Module):
    def __init__(self, axes=3, channels=16):
        super().__init__()
        self.convolution = nn.Conv1d(axes, channels, kernel_size=5, padding=2)

    def forward(self, x):
        return F.relu(self.convolution(x), inplace=True).mean(dim=2)


class CNN(MultiJointModel):
    def __init__(self, joints=6, axes=3):
        super().__init__(joints, axes)
        self.branches = nn.ModuleList([CNNBranch(axes) for _ in range(joints)])
        self.heads = nn.ModuleList([nn.Linear(16, 1) for _ in range(joints)])

    def forward(self, x):
        joints = self.split(x)
        return torch.cat(
            [head(branch(joints[:, index])) for index, (branch, head) in enumerate(zip(self.branches, self.heads))], dim=1
        )


class LSTM(MultiJointModel):
    def __init__(self, joints=6, axes=3, hidden=32):
        super().__init__(joints, axes)
        self.branches = nn.ModuleList([nn.LSTM(axes, hidden, batch_first=True) for _ in range(joints)])
        self.heads = nn.ModuleList([nn.Linear(hidden, 1) for _ in range(joints)])

    def forward(self, x):
        joints = self.split(x)
        logits = []
        for index, (branch, head) in enumerate(zip(self.branches, self.heads)):
            output, _ = branch(joints[:, index].transpose(1, 2))
            logits.append(head(output.mean(dim=1)))
        return torch.cat(logits, dim=1)


class GRUBranch(nn.Module):
    def __init__(self, axes=3, hidden=64):
        super().__init__()
        self.recurrent = nn.GRU(axes, hidden, batch_first=True, bidirectional=True)
        self.projection = nn.Linear(2 * hidden, 32)

    def forward(self, x):
        output, _ = self.recurrent(x.transpose(1, 2))
        return self.projection(output.mean(dim=1))


class GRU(MultiJointModel):
    def __init__(self, joints=6, axes=3):
        super().__init__(joints, axes)
        self.branches = nn.ModuleList([GRUBranch(axes) for _ in range(joints)])
        self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(joints)])

    def forward(self, x):
        joints = self.split(x)
        features = torch.stack([branch(joints[:, index]) for index, branch in enumerate(self.branches)], dim=1)
        global_vector = features.mean(dim=1)
        return torch.cat(
            [head(features[:, index] + global_vector) for index, head in enumerate(self.heads)], dim=1
        )


class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.first = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.first_norm = nn.BatchNorm1d(out_channels)
        self.second = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.second_norm = nn.BatchNorm1d(out_channels)
        self.shortcut = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if stride != 1 or in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x):
        output = F.relu(self.first_norm(self.first(x)), inplace=True)
        output = self.second_norm(self.second(output))
        return F.relu(output + self.shortcut(x), inplace=True)


class ResNetBranch(nn.Module):
    def __init__(self, axes=3):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(axes, 16, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            ResidualBlock1D(16, 16),
            ResidualBlock1D(16, 32, stride=2),
            ResidualBlock1D(32, 32),
        )

    def forward(self, x):
        return self.network(x)


class ResNet(MultiJointModel):
    def __init__(self, joints=6, axes=3):
        super().__init__(joints, axes)
        self.branches = nn.ModuleList([ResNetBranch(axes) for _ in range(joints)])
        self.fusion = nn.Sequential(
            nn.Conv1d(32 * joints, 32, kernel_size=1, bias=False), nn.BatchNorm1d(32), nn.ReLU(inplace=True)
        )
        self.heads = nn.ModuleList([nn.Linear(32, 1) for _ in range(joints)])

    def forward(self, x):
        joints = self.split(x)
        private = [branch(joints[:, index]) for index, branch in enumerate(self.branches)]
        length = min(item.size(-1) for item in private)
        private = [item[..., :length] for item in private]
        global_vector = self.fusion(torch.cat(private, dim=1)).mean(dim=2)
        return torch.cat(
            [head(feature.mean(dim=2) + global_vector) for head, feature in zip(self.heads, private)], dim=1
        )
