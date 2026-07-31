import argparse

from data import build_loaders
from models import build_model
from utils import fit, load_config, seed_everything


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jfdnet.yaml")
    parser.add_argument("--run-name", default=None)
    return parser.parse_args()


def main():
    arguments = parse_arguments()
    config = load_config(arguments.config)
    seed_everything(int(config.get("seed", 42)))
    name = config["model"].get("name", "jfdnet")
    model = build_model(name, config["model"])
    loaders = build_loaders(config)
    fit(model, loaders, config, arguments.run_name or name)


if __name__ == "__main__":
    main()
