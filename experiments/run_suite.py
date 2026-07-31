import argparse
import json
import sys
from itertools import product
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data import build_loaders
from models import (
    JFDNet,
    build_aim_variant,
    build_architecture_variant,
    build_model,
    build_neck_attention_variant,
    build_temporal_mixer_variant,
)
from utils import fit, load_config, seed_everything


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/jfdnet.yaml")
    parser.add_argument(
        "--suite",
        required=True,
        choices=(
            "baselines",
            "architecture",
            "aim",
            "neck_attention",
            "temporal_mixer",
            "depth_sensitivity",
            "training_size",
        ),
    )
    parser.add_argument("--variants", nargs="*", default=None)
    return parser.parse_args()


def model_options(config):
    excluded = {"name", "sequence_length"}
    return {key: value for key, value in config["model"].items() if key not in excluded}


def suite_runs(name, options):
    if name == "baselines":
        return [(item, lambda item=item: build_model(item, options), None) for item in (
            "mjar",
            "cnn_lstm",
            "ticnn",
            "msmcnn",
            "mlp",
            "cnn",
            "lstm",
            "gru",
            "resnet",
            "jfdnet",
        )]
    if name == "architecture":
        variants = (
            "without_aim",
            "without_neck",
            "mamba_backbone_with_dijr",
            "mamba_backbone_without_neck",
            "jfdnet",
        )
        return [(item, lambda item=item: build_architecture_variant(item, options), None) for item in variants]
    if name == "aim":
        variants = (
            "spatial_attention",
            "temporal_attention",
            "cbam",
            "depthwise_conv",
            "frequency_gating",
            "sh_sdsa",
            "mha",
            "aim",
        )
        return [(item, lambda item=item: build_aim_variant(item, options), None) for item in variants]
    if name == "neck_attention":
        attentions = ("none", "sh_sdsa", "mha", "spatial", "temporal", "cbam")
        return [
            (
                f"pre_{pre}_post_{post}",
                lambda pre=pre, post=post: build_neck_attention_variant(pre, post, options),
                None,
            )
            for pre, post in product(attentions, attentions)
        ]
    if name == "temporal_mixer":
        variants = ("bigru", "bilstm", "lstm", "transformer", "frets", "timesnet", "dlinear", "cnn", "tcn", "mamba", "bdmm")
        return [(item, lambda item=item: build_temporal_mixer_variant(item, options), None) for item in variants]
    if name == "depth_sensitivity":
        return [
            (
                f"tdwc_{tdwc}_bdmm_{bdmm}",
                lambda tdwc=tdwc, bdmm=bdmm: JFDNet(**{**options, "tdwc_blocks": tdwc, "bdmm_layers": bdmm}),
                None,
            )
            for tdwc, bdmm in product(range(1, 6), range(1, 8))
        ]
    if name == "training_size":
        return [
            (f"train_{size}", lambda: JFDNet(**options), size)
            for size in (300, 600, 900, 1200, 1500)
        ]
    raise ValueError(name)


def main():
    arguments = parse_arguments()
    config = load_config(arguments.config)
    options = model_options(config)
    runs = suite_runs(arguments.suite, options)
    if arguments.variants:
        selected = set(arguments.variants)
        runs = [run for run in runs if run[0] in selected]
    results = []
    for variant, builder, train_size in runs:
        print(f"suite={arguments.suite} variant={variant}")
        seed_everything(int(config.get("seed", 42)))
        loaders = build_loaders(config, train_size=train_size)
        result = fit(builder(), loaders, config, f"{arguments.suite}/{variant}")
        results.append(result)
    summary_directory = Path(config.get("output_dir", "outputs")) / arguments.suite
    summary_directory.mkdir(parents=True, exist_ok=True)
    with (summary_directory / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2)


if __name__ == "__main__":
    main()
