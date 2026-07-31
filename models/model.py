import math

import pywt
import torch
import torch.nn as nn
import torch.nn.functional as F


class TrainableDiscreteWaveletConvolution(nn.Module):
    def __init__(self, channels, wavelet="db4"):
        super().__init__()
        filters = pywt.Wavelet(wavelet).filter_bank
        low = torch.tensor(filters[0], dtype=torch.float32).view(1, 1, -1)
        high = torch.tensor(filters[1], dtype=torch.float32).view(1, 1, -1)
        self.kernel_size = low.size(-1)
        self.channels = channels
        self.weight = nn.Parameter(torch.cat((low, high), dim=0).repeat(channels, 1, 1))

    def forward(self, x):
        length = x.size(-1)
        output_length = pywt.dwt_coeff_len(length, self.kernel_size, mode="zero")
        padding = 2 * (output_length - 1) - length + self.kernel_size
        left = padding // 2
        right = padding - left
        transformed = F.conv1d(F.pad(x, (left, right)), self.weight, stride=2, groups=self.channels)
        return transformed[:, 0::2].contiguous(), transformed[:, 1::2].contiguous()


class TDWCBlock(nn.Module):
    def __init__(self, in_channels, out_channels, wavelet="db4", dropout=0.1):
        super().__init__()
        self.wavelet = TrainableDiscreteWaveletConvolution(in_channels, wavelet)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Sequential(
            nn.Conv1d(2 * in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        low, high = self.wavelet(x)
        return self.projection(self.dropout(torch.cat((low, high), dim=1)))


class AxisInteractionModule(nn.Module):
    def __init__(self, channels, axis_token_dim=None, dropout=0.0):
        super().__init__()
        self.axis_token_dim = axis_token_dim or math.ceil(channels / 3)
        projected = 3 * self.axis_token_dim
        self.pre_projection = nn.Linear(channels, projected)
        self.query = nn.Linear(self.axis_token_dim, self.axis_token_dim)
        self.key = nn.Linear(self.axis_token_dim, self.axis_token_dim)
        self.value = nn.Linear(self.axis_token_dim, self.axis_token_dim)
        self.dropout = nn.Dropout(dropout)
        self.post_projection = nn.Linear(projected, channels)

    def forward(self, x):
        batch, steps, _ = x.shape
        tokens = self.pre_projection(x).view(batch * steps, 3, self.axis_token_dim)
        query = self.query(tokens)
        key = self.key(tokens)
        value = self.value(tokens)
        weights = torch.softmax(query @ key.transpose(-1, -2) / math.sqrt(self.axis_token_dim), dim=-1)
        output = self.dropout(weights @ value).view(batch, steps, 3 * self.axis_token_dim)
        return self.post_projection(output)


class TemporalAttention(nn.Module):
    def __init__(self, channels, kernel_size=3):
        super().__init__()
        self.channels = channels
        self.gate = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)

    def forward(self, x):
        transposed = x.transpose(1, 2)
        pooled = torch.cat((transposed.mean(dim=1, keepdim=True), transposed.amax(dim=1, keepdim=True)), dim=1)
        weights = torch.sigmoid(self.gate(pooled))
        return (transposed * weights).transpose(1, 2).contiguous()


class SpatialAttention(nn.Module):
    def __init__(self, channels, reduction=4):
        super().__init__()
        hidden = max(1, channels // reduction)
        self.network = nn.Sequential(nn.Linear(channels, hidden), nn.ReLU(inplace=True), nn.Linear(hidden, channels))

    def forward(self, x):
        average = x.mean(dim=1)
        maximum = x.amax(dim=1)
        weights = torch.sigmoid(self.network(average) + self.network(maximum)).unsqueeze(1)
        return x * weights


class CBAMAttention(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.spatial = SpatialAttention(channels)
        self.temporal = TemporalAttention(channels)

    def forward(self, x):
        return self.temporal(self.spatial(x))


class SingleHeadScaledDotProductSelfAttention(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.query = nn.Linear(channels, channels)
        self.key = nn.Linear(channels, channels)
        self.value = nn.Linear(channels, channels)
        self.projection = nn.Linear(channels, channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normalized = self.norm(x)
        query = self.query(normalized)
        key = self.key(normalized)
        value = self.value(normalized)
        weights = torch.softmax(query @ key.transpose(-1, -2) / math.sqrt(x.size(-1)), dim=-1)
        return x + self.dropout(self.projection(weights @ value))


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, channels, heads=4, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attention = nn.MultiheadAttention(channels, heads, dropout=dropout, batch_first=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        normalized = self.norm(x)
        output, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        return x + self.dropout(output)


class DepthwiseTemporalConvolution(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x):
        return self.network(x.transpose(1, 2)).transpose(1, 2).contiguous()


class FrequencyGating(nn.Module):
    def __init__(self, channels, axis_token_dim=None):
        super().__init__()
        self.axis_token_dim = axis_token_dim or math.ceil(channels / 3)
        projected = 3 * self.axis_token_dim
        self.pre_projection = nn.Linear(channels, projected)
        self.gate = nn.Linear(3, 3)
        self.post_projection = nn.Linear(projected, channels)

    def forward(self, x):
        batch, steps, _ = x.shape
        tokens = self.pre_projection(x).view(batch, steps, 3, self.axis_token_dim)
        spectrum = torch.fft.rfft(tokens.permute(0, 2, 3, 1), dim=-1)
        energy = spectrum.abs().square().mean(dim=(-1, -2))
        weights = torch.softmax(self.gate(energy), dim=-1).view(batch, 1, 3, 1)
        output = (tokens * weights).reshape(batch, steps, 3 * self.axis_token_dim)
        return self.post_projection(output)


def build_attention(name, channels, dropout=0.1):
    normalized = name.lower().replace("-", "_")
    if normalized in {"none", "identity"}:
        return nn.Identity()
    if normalized in {"aim", "axis_interaction"}:
        return AxisInteractionModule(channels, dropout=dropout)
    if normalized in {"temporal", "temporal_attention"}:
        return TemporalAttention(channels)
    if normalized in {"spatial", "spatial_attention"}:
        return SpatialAttention(channels)
    if normalized == "cbam":
        return CBAMAttention(channels)
    if normalized in {"depthwise_conv", "depthwise"}:
        return DepthwiseTemporalConvolution(channels)
    if normalized in {"frequency_gating", "frequency"}:
        return FrequencyGating(channels)
    if normalized in {"sh_sdsa", "single_head"}:
        return SingleHeadScaledDotProductSelfAttention(channels, dropout)
    if normalized in {"mha", "multi_head"}:
        return MultiHeadSelfAttention(channels, dropout=dropout)
    raise ValueError(f"Unknown attention module: {name}")


class SJTAWBackbone(nn.Module):
    def __init__(self, wavelet="db4", base_channels=48, blocks=3, attention="aim", dropout=0.1):
        super().__init__()
        if blocks < 1:
            raise ValueError("SJTAW requires at least one TDWC block")
        widths = [base_channels] + [2 * base_channels] * max(0, blocks - 1)
        modules = []
        in_channels = 3
        for out_channels in widths:
            modules.append(TDWCBlock(in_channels, out_channels, wavelet, dropout))
            in_channels = out_channels
        self.tdwc_blocks = nn.ModuleList(modules)
        self.out_channels = widths[-1]
        self.axis_interaction = build_attention(attention, self.out_channels, dropout)

    def forward(self, x):
        for block in self.tdwc_blocks:
            x = block(x)
        return self.axis_interaction(x.transpose(1, 2).contiguous())


class MambaResidualStack(nn.Module):
    def __init__(self, channels, layers, dropout=0.1):
        super().__init__()
        try:
            from mamba_ssm import Mamba
        except ImportError as exc:
            raise ImportError("JFDNet requires mamba-ssm; install the dependencies before constructing the model") from exc
        self.layers = nn.ModuleList(
            [Mamba(d_model=channels, d_state=16, d_conv=4, expand=2) for _ in range(layers)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        for layer in self.layers:
            x = x + self.dropout(layer(x))
        return x


class BidirectionalMambaMixer(nn.Module):
    def __init__(self, channels, layers=5, dropout=0.1):
        super().__init__()
        self.forward_stack = MambaResidualStack(channels, layers, dropout)
        self.backward_stack = MambaResidualStack(channels, layers, dropout)

    def forward(self, x):
        forward = self.forward_stack(x)
        backward = torch.flip(self.backward_stack(torch.flip(x, dims=(1,))), dims=(1,))
        return 0.5 * (forward + backward)


class DIJRNeck(nn.Module):
    def __init__(
        self,
        channels,
        joints=6,
        bdmm_layers=5,
        pre_attention="temporal",
        post_attention="sh_sdsa",
        dropout=0.1,
        mixer=None,
    ):
        super().__init__()
        self.joints = joints
        self.channels = channels
        self.joint_channel_width = joints * channels
        self.pre_attention = build_attention(pre_attention, self.joint_channel_width, dropout)
        self.normalization = nn.LayerNorm(self.joint_channel_width)
        self.mixer = mixer or BidirectionalMambaMixer(self.joint_channel_width, bdmm_layers, dropout)
        self.projection = nn.Linear(self.joint_channel_width, channels)
        self.post_attention = build_attention(post_attention, channels, dropout)

    def forward(self, features):
        concatenated = torch.cat(features, dim=-1)
        normalized = self.normalization(self.pre_attention(concatenated))
        mixed = normalized + self.mixer(normalized)
        global_context = self.post_attention(self.projection(mixed))
        return [feature + global_context for feature in features], global_context


class JointHead(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        hidden = max(4, channels // 2)
        self.network = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.network(x.mean(dim=1)).squeeze(-1)


class JFDNet(nn.Module):
    def __init__(
        self,
        input_joints=6,
        axes_per_joint=3,
        wavelet="db4",
        base_channels=48,
        tdwc_blocks=3,
        bdmm_layers=5,
        aim_attention="aim",
        pre_attention="temporal",
        post_attention="sh_sdsa",
        dropout=0.1,
        use_neck=True,
        mixer=None,
        backbone_factory=None,
    ):
        super().__init__()
        self.input_joints = input_joints
        self.axes_per_joint = axes_per_joint
        self.use_neck = use_neck
        if backbone_factory is None:
            backbone_factory = lambda: SJTAWBackbone(
                wavelet=wavelet,
                base_channels=base_channels,
                blocks=tdwc_blocks,
                attention=aim_attention,
                dropout=dropout,
            )
        self.backbones = nn.ModuleList([backbone_factory() for _ in range(input_joints)])
        channels = self.backbones[0].out_channels
        self.neck = (
            DIJRNeck(
                channels,
                input_joints,
                bdmm_layers,
                pre_attention,
                post_attention,
                dropout,
                mixer,
            )
            if use_neck
            else None
        )
        self.heads = nn.ModuleList([JointHead(channels, dropout) for _ in range(input_joints)])

    def forward_features(self, x):
        expected = self.input_joints * self.axes_per_joint
        if x.ndim != 3 or x.size(1) != expected:
            raise ValueError(f"Expected input shape [batch, {expected}, time]")
        joint_inputs = torch.chunk(x, self.input_joints, dim=1)
        private_features = [backbone(joint) for backbone, joint in zip(self.backbones, joint_inputs)]
        if self.neck is None:
            return private_features, None
        return self.neck(private_features)

    def forward(self, x):
        features, _ = self.forward_features(x)
        return torch.stack([head(feature) for head, feature in zip(self.heads, features)], dim=1)
