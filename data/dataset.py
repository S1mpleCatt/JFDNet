from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class RobotJointDataset(Dataset):
    def __init__(
        self,
        root,
        split,
        signal_file="signals.npy",
        label_file="labels.npy",
        channels=18,
        sequence_length=2560,
        output_joints=6,
        standardize=True,
    ):
        super().__init__()
        directory = Path(root) / split
        signal_path = directory / signal_file
        label_path = directory / label_file
        if not signal_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"Missing {split} split under {directory}")
        self.signals = np.load(signal_path, mmap_mode="r", allow_pickle=False)
        self.labels = np.load(label_path, mmap_mode="r", allow_pickle=False)
        self.channels = channels
        self.sequence_length = sequence_length
        self.output_joints = output_joints
        self.standardize = standardize
        if len(self.signals) != len(self.labels):
            raise ValueError(f"Signal and label counts differ in {split}")
        if self.signals.ndim != 3 or self.labels.ndim != 2:
            raise ValueError(f"Invalid array rank in {split}")

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, index):
        signal = np.asarray(self.signals[index], dtype=np.float32)
        if signal.shape == (self.sequence_length, self.channels):
            signal = signal.T
        if signal.shape != (self.channels, self.sequence_length):
            raise ValueError(f"Invalid signal shape at index {index}: {signal.shape}")
        signal = torch.from_numpy(signal.copy())
        if self.standardize:
            mean = signal.mean(dim=1, keepdim=True)
            deviation = signal.std(dim=1, keepdim=True).clamp_min(1e-6)
            signal = (signal - mean) / deviation
        label = np.asarray(self.labels[index], dtype=np.float32).reshape(-1)
        if label.size > self.output_joints:
            label = label[: self.output_joints]
        if label.size < self.output_joints:
            label = np.pad(label, (0, self.output_joints - label.size))
        return signal, torch.from_numpy(label.copy()), int(index)


def build_loaders(config, train_size=None):
    data = config["data"]
    model = config["model"]
    training = config["training"]
    common = {
        "root": data["root"],
        "signal_file": data.get("signal_file", "signals.npy"),
        "label_file": data.get("label_file", "labels.npy"),
        "channels": int(model.get("input_joints", 6)) * int(model.get("axes_per_joint", 3)),
        "sequence_length": int(model.get("sequence_length", 2560)),
        "output_joints": int(model.get("input_joints", 6)),
        "standardize": bool(data.get("standardize", True)),
    }
    datasets = {split: RobotJointDataset(split=split, **common) for split in ("train", "validation", "test")}
    if train_size is not None:
        train_size = min(int(train_size), len(datasets["train"]))
        generator = np.random.default_rng(int(config.get("seed", 42)))
        indices = generator.permutation(len(datasets["train"]))[:train_size].tolist()
        datasets["train"] = Subset(datasets["train"], indices)
    seed = int(config.get("seed", 42))
    torch_generator = torch.Generator().manual_seed(seed)
    loader_options = {
        "batch_size": int(training.get("batch_size", 64)),
        "num_workers": int(training.get("num_workers", 4)),
        "pin_memory": bool(training.get("pin_memory", True)),
        "persistent_workers": bool(training.get("persistent_workers", True)) and int(training.get("num_workers", 4)) > 0,
    }
    return {
        "train": DataLoader(datasets["train"], shuffle=True, generator=torch_generator, **loader_options),
        "validation": DataLoader(datasets["validation"], shuffle=False, **loader_options),
        "test": DataLoader(datasets["test"], shuffle=False, **loader_options),
    }
