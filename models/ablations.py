import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import BidirectionalMambaMixer, JFDNet, MambaResidualStack


class RecurrentMixer(nn.Module):
    def __init__(self, channels, cell="gru", bidirectional=False, layers=1, dropout=0.1):
        super().__init__()
        recurrent = nn.GRU if cell == "gru" else nn.LSTM
        self.network = recurrent(
            channels,
            channels,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.projection = nn.Linear(2 * channels, channels) if bidirectional else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output, _ = self.network(x)
        return self.dropout(self.projection(output))


class SinusoidalPositionEncoding(nn.Module):
    def __init__(self, channels, maximum_length=4096):
        super().__init__()
        position = torch.arange(maximum_length, dtype=torch.float32).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, channels, 2, dtype=torch.float32) * (-math.log(10000.0) / channels))
        encoding = torch.zeros(maximum_length, channels)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.encoding[:, : x.size(1)]


class TransformerMixer(nn.Module):
    def __init__(self, channels, layers=2, dropout=0.1):
        super().__init__()
        block = nn.TransformerEncoderLayer(
            d_model=channels,
            nhead=4,
            dim_feedforward=2 * channels,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.position = SinusoidalPositionEncoding(channels)
        self.network = nn.TransformerEncoder(block, num_layers=layers)
        self.norm = nn.LayerNorm(channels)

    def forward(self, x):
        return self.norm(self.network(x + self.position(x)))


class CNNMixer(nn.Module):
    def __init__(self, channels, layers=3, dropout=0.1):
        super().__init__()
        modules = []
        for _ in range(layers):
            modules.extend(
                [
                    nn.Conv1d(channels, channels, kernel_size=3, padding=1),
                    nn.BatchNorm1d(channels),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ]
            )
        self.network = nn.Sequential(*modules)

    def forward(self, x):
        return self.network(x.transpose(1, 2)).transpose(1, 2).contiguous()


class TCNBlock(nn.Module):
    def __init__(self, channels, dilation, dropout=0.1):
        super().__init__()
        self.first = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.second = nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation)
        self.first_norm = nn.BatchNorm1d(channels)
        self.second_norm = nn.BatchNorm1d(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        output = self.dropout(F.gelu(self.first_norm(self.first(x))))
        return F.gelu(self.second_norm(self.second(output)) + x)


class TCNMixer(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.blocks = nn.Sequential(*[TCNBlock(channels, dilation, dropout) for dilation in (1, 2, 4, 8)])

    def forward(self, x):
        return self.blocks(x.transpose(1, 2)).transpose(1, 2).contiguous()


class PeriodBlock(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, groups=channels),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x, period):
        batch, channels, steps = x.shape
        padding = (period - steps % period) % period
        padded = F.pad(x, (0, padding)) if padding else x
        output = self.network(padded.view(batch, channels, padded.size(-1) // period, period))
        return output.reshape(batch, channels, padded.size(-1))[..., :steps]


class TimesNetMixer(nn.Module):
    def __init__(self, channels, top_k=4, dropout=0.1):
        super().__init__()
        self.top_k = top_k
        self.block = PeriodBlock(channels, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        transposed = x.transpose(1, 2).contiguous()
        magnitude = torch.fft.rfft(transposed, dim=-1).abs().mean(dim=(0, 1))
        count = min(self.top_k, max(1, magnitude.numel() - 1))
        indices = torch.topk(magnitude[1:], k=count).indices + 1
        periods = [max(2, int(round(x.size(1) / index.item()))) for index in indices]
        output = torch.stack([self.block(transposed, period) for period in periods], dim=0).mean(dim=0)
        return self.dropout(output.transpose(1, 2).contiguous())


class FreTSMixer(nn.Module):
    def __init__(self, channels, dropout=0.1):
        super().__init__()
        latent = min(256, channels)
        block = nn.TransformerEncoderLayer(
            latent,
            nhead=4,
            dim_feedforward=2 * latent,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.input_projection = nn.Linear(2 * channels, latent)
        self.network = nn.TransformerEncoder(block, num_layers=1)
        self.output_projection = nn.Linear(latent, 2 * channels)
        self.norm = nn.LayerNorm(latent)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        batch, steps, channels = x.shape
        spectrum = torch.fft.rfft(x.transpose(1, 2), dim=-1)
        values = torch.cat((spectrum.real, spectrum.imag), dim=1).transpose(1, 2)
        values = self.output_projection(self.norm(self.network(self.input_projection(values))))
        values = values.transpose(1, 2)
        real, imaginary = torch.split(values, channels, dim=1)
        output = torch.fft.irfft(torch.complex(real, imaginary), n=steps, dim=-1)
        return self.dropout(output.transpose(1, 2).contiguous())


class DLinearMixer(nn.Module):
    def __init__(self, maximum_length=4096, moving_average=25, dropout=0.1):
        super().__init__()
        self.moving_average = moving_average
        self.seasonal = nn.Parameter(torch.eye(maximum_length))
        self.trend = nn.Parameter(torch.eye(maximum_length))
        kernel = torch.ones(1, 1, moving_average) / moving_average
        self.register_buffer("kernel", kernel, persistent=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        steps = x.size(1)
        transposed = x.transpose(1, 2)
        padding = (self.moving_average - 1) // 2
        padded = F.pad(transposed, (padding, padding), mode="reflect")
        trend = F.conv1d(padded, self.kernel.repeat(transposed.size(1), 1, 1), groups=transposed.size(1)).transpose(1, 2)
        seasonal = x - trend
        seasonal = torch.matmul(seasonal.transpose(1, 2), self.seasonal[:steps, :steps]).transpose(1, 2)
        trend = torch.matmul(trend.transpose(1, 2), self.trend[:steps, :steps]).transpose(1, 2)
        return self.dropout(seasonal + trend)


class UnidirectionalMambaMixer(nn.Module):
    def __init__(self, channels, layers=1, dropout=0.1):
        super().__init__()
        self.network = MambaResidualStack(channels, layers, dropout)

    def forward(self, x):
        return self.network(x)


class MambaJointBackbone(nn.Module):
    def __init__(self, out_channels=96, layers=3, dropout=0.1):
        super().__init__()
        self.out_channels = out_channels
        self.projection = nn.Linear(3, out_channels)
        self.norm = nn.LayerNorm(out_channels)
        self.network = MambaResidualStack(out_channels, layers, dropout)

    def forward(self, x):
        return self.network(self.norm(self.projection(x.transpose(1, 2))))


def build_temporal_mixer(name, channels, layers=5, dropout=0.1):
    normalized = name.lower().replace("-", "_")
    if normalized == "bdmm":
        return BidirectionalMambaMixer(channels, layers, dropout)
    if normalized == "bigru":
        return RecurrentMixer(channels, "gru", True, dropout=dropout)
    if normalized == "bilstm":
        return RecurrentMixer(channels, "lstm", True, dropout=dropout)
    if normalized == "lstm":
        return RecurrentMixer(channels, "lstm", False, dropout=dropout)
    if normalized == "transformer":
        return TransformerMixer(channels, dropout=dropout)
    if normalized == "frets":
        return FreTSMixer(channels, dropout)
    if normalized == "timesnet":
        return TimesNetMixer(channels, dropout=dropout)
    if normalized == "dlinear":
        return DLinearMixer(dropout=dropout)
    if normalized in {"cnn", "1dcnn"}:
        return CNNMixer(channels, dropout=dropout)
    if normalized == "tcn":
        return TCNMixer(channels, dropout)
    if normalized == "mamba":
        return UnidirectionalMambaMixer(channels, dropout=dropout)
    raise ValueError(f"Unknown temporal mixer: {name}")


def _model_dimensions(options):
    blocks = int(options.get("tdwc_blocks", 3))
    base = int(options.get("base_channels", 48))
    channels = base if blocks == 1 else 2 * base
    joints = int(options.get("input_joints", 6))
    return channels, joints


def build_architecture_variant(name, options):
    normalized = name.lower().replace("-", "_")
    local = dict(options)
    if normalized == "without_aim":
        local["aim_attention"] = "none"
        return JFDNet(**local)
    if normalized == "without_neck":
        local["use_neck"] = False
        return JFDNet(**local)
    if normalized in {"mamba_backbone_with_dijr", "mamba_backbone_without_neck"}:
        local["backbone_factory"] = lambda: MambaJointBackbone(2 * local.get("base_channels", 48), 3, local.get("dropout", 0.1))
        local["use_neck"] = normalized.endswith("with_dijr")
        return JFDNet(**local)
    if normalized == "jfdnet":
        return JFDNet(**local)
    raise ValueError(f"Unknown architecture variant: {name}")


def build_aim_variant(name, options):
    local = dict(options)
    local["aim_attention"] = name
    return JFDNet(**local)


def build_neck_attention_variant(pre_attention, post_attention, options):
    local = dict(options)
    local["pre_attention"] = pre_attention
    local["post_attention"] = post_attention
    return JFDNet(**local)


def build_temporal_mixer_variant(name, options):
    local = dict(options)
    channels, joints = _model_dimensions(local)
    local["mixer"] = build_temporal_mixer(
        name,
        channels * joints,
        int(local.get("bdmm_layers", 5)),
        float(local.get("dropout", 0.1)),
    )
    return JFDNet(**local)
