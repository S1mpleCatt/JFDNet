import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import yaml
from tqdm import tqdm

from .config import resolve_device, seed_everything
from .metrics import multilabel_metrics


def train_epoch(model, loader, optimizer, criterion, device, gradient_clip):
    model.train()
    losses = []
    progress = tqdm(loader, leave=False, desc="train")
    for signals, labels, _ in progress:
        signals = signals.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = model(signals)
        loss = criterion(logits, labels)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss became non-finite")
        loss.backward()
        if gradient_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), gradient_clip)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        progress.set_postfix(loss=f"{losses[-1]:.4f}")
    return float(np.mean(losses)) if losses else 0.0


def predict(model, loader, device):
    model.eval()
    probabilities = []
    labels = []
    with torch.inference_mode():
        for signals, targets, _ in tqdm(loader, leave=False, desc="evaluate"):
            logits = model(signals.to(device, non_blocking=True))
            probabilities.append(torch.sigmoid(logits).cpu().numpy())
            labels.append(targets.numpy())
    return np.concatenate(labels, axis=0), np.concatenate(probabilities, axis=0)


def evaluate(model, loader, device, evaluated_joints=4, threshold=0.5):
    labels, probabilities = predict(model, loader, device)
    return multilabel_metrics(labels, probabilities, evaluated_joints, threshold)


def load_checkpoint(model, path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    return checkpoint


def fit(model, loaders, config, run_name):
    seed = int(config.get("seed", 42))
    seed_everything(seed)
    training = config["training"]
    device = resolve_device(config.get("device", "auto"))
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(training.get("learning_rate", 1e-3)))
    criterion = nn.BCEWithLogitsLoss()
    epochs = int(training.get("epochs", 50))
    patience = int(training.get("early_stopping_patience", 10))
    minimum_delta = float(training.get("early_stopping_min_delta", 0.0))
    gradient_clip = training.get("gradient_clip", 1.0)
    gradient_clip = None if gradient_clip is None else float(gradient_clip)
    evaluated_joints = int(training.get("evaluated_joints", 4))
    threshold = float(training.get("threshold", 0.5))
    run_directory = Path(config.get("output_dir", "outputs")) / run_name
    run_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_directory / "best.pt"
    best_f1 = -1.0
    epochs_without_improvement = 0
    history = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, loaders["train"], optimizer, criterion, device, gradient_clip)
        validation = evaluate(model, loaders["validation"], device, evaluated_joints, threshold)
        record = {"epoch": epoch, "training_loss": loss, "validation": validation}
        history.append(record)
        print(
            f"epoch={epoch:02d} loss={loss:.6f} val_precision={validation['precision']:.4f} "
            f"val_recall={validation['recall']:.4f} val_f1={validation['f1']:.4f}"
        )
        if validation["f1"] > best_f1 + minimum_delta:
            best_f1 = validation["f1"]
            epochs_without_improvement = 0
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "epoch": epoch,
                    "validation": validation,
                    "run_name": run_name,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
        if epochs_without_improvement >= patience:
            break
    checkpoint = load_checkpoint(model, checkpoint_path, device)
    validation = evaluate(model, loaders["validation"], device, evaluated_joints, threshold)
    test = evaluate(model, loaders["test"], device, evaluated_joints, threshold)
    result = {
        "run_name": run_name,
        "best_epoch": int(checkpoint["epoch"]),
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "validation": validation,
        "test": test,
        "checkpoint": str(checkpoint_path),
    }
    with (run_directory / "history.json").open("w", encoding="utf-8") as stream:
        json.dump(history, stream, indent=2)
    with (run_directory / "metrics.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    saved_config = {key: value for key, value in config.items() if key != "config_path"}
    with (run_directory / "config.yaml").open("w", encoding="utf-8") as stream:
        yaml.safe_dump(saved_config, stream, sort_keys=False)
    print(
        f"test_precision={test['precision']:.4f} test_recall={test['recall']:.4f} "
        f"test_f1={test['f1']:.4f}"
    )
    return result
