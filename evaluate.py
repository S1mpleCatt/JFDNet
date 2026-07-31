import argparse
import json

from data import build_loaders
from models import build_model
from utils import evaluate, load_checkpoint, load_config, resolve_device


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jfdnet.yaml")
    parser.add_argument("--checkpoint", required=True)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    config = load_config(arguments.config)
    device = resolve_device(config.get("device", "auto"))
    name = config["model"].get("name", "jfdnet")
    model = build_model(name, config["model"]).to(device)
    load_checkpoint(model, arguments.checkpoint, device)
    loaders = build_loaders(config)
    metrics = evaluate(
        model,
        loaders["test"],
        device,
        int(config["training"].get("evaluated_joints", 4)),
        float(config["training"].get("threshold", 0.5)),
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
