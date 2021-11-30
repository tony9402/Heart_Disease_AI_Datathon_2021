from .UNet import UNet
from .swinT import SwinTransformer
from .DeepLabV3 import DeepLabV3plus_ResNet50
from .DeepLabV3 import DeepLabV3plus_ResNet101
from .DeepLabV3 import DeepLabV3plus_HRNetV2_32

_models = [
    UNet,
    DeepLabV3plus_ResNet50,
    DeepLabV3plus_ResNet101,
    DeepLabV3plus_HRNetV2_32
]
_models_dict = {v.__name__: v for v in _models}

def get_model(model_name):
    if model_name not in _models_dict:
        raise KeyError(f"No Such Model '{model_name}'")

    return _models_dict[model_name]