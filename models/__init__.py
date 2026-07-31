import inspect

from .ablations import (
    build_aim_variant,
    build_architecture_variant,
    build_neck_attention_variant,
    build_temporal_mixer_variant,
)
from .baselines import CNN, CNNLSTM, GRU, LSTM, MJAR, MLP, MSMCNN, ResNet, TICNN
from .model import JFDNet


MODEL_REGISTRY = {
    "jfdnet": JFDNet,
    "mjar": MJAR,
    "cnn_lstm": CNNLSTM,
    "ticnn": TICNN,
    "msmcnn": MSMCNN,
    "mlp": MLP,
    "cnn": CNN,
    "lstm": LSTM,
    "gru": GRU,
    "resnet": ResNet,
}


def build_model(name, options=None):
    normalized = name.lower().replace("-", "_")
    if normalized not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {name}")
    options = dict(options or {})
    if normalized == "jfdnet":
        signature = inspect.signature(MODEL_REGISTRY[normalized].__init__)
        allowed = {key for key in signature.parameters if key != "self"}
        return MODEL_REGISTRY[normalized](**{key: value for key, value in options.items() if key in allowed})
    return MODEL_REGISTRY[normalized](
        joints=int(options.get("input_joints", 6)),
        axes=int(options.get("axes_per_joint", 3)),
    )


__all__ = [
    "JFDNet",
    "MJAR",
    "CNNLSTM",
    "TICNN",
    "MSMCNN",
    "MLP",
    "CNN",
    "LSTM",
    "GRU",
    "ResNet",
    "build_model",
    "build_aim_variant",
    "build_architecture_variant",
    "build_neck_attention_variant",
    "build_temporal_mixer_variant",
]
