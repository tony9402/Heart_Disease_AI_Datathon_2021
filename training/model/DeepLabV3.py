from .DeepLabV3Plus.modeling import deeplabv3_resnet50, deeplabv3_resnet101
from .DeepLabV3Plus.modeling import deeplabv3_hrnetv2_32, deeplabv3_hrnetv2_48
from .DeepLabV3Plus.modeling import deeplabv3_mobilenet

from .DeepLabV3Plus.modeling import deeplabv3plus_resnet50, deeplabv3plus_resnet101
from .DeepLabV3Plus.modeling import deeplabv3plus_hrnetv2_32, deeplabv3plus_hrnetv2_48
from .DeepLabV3Plus.modeling import deeplabv3plus_mobilenet

from .DeepLabV3Plus import convert_to_separable_conv

def DeepLabV3plus_ResNet50(**kwargs):
    model = deeplabv3plus_resnet50(**kwargs)
    return convert_to_separable_conv(model)

def DeepLabV3plus_ResNet101(**kwargs):
    model = deeplabv3plus_resnet101(**kwargs)
    return convert_to_separable_conv(model)

def DeepLabV3plus_HRNetV2_32(**kwargs):
    model = deeplabv3plus_hrnetv2_32(**kwargs)
    return convert_to_separable_conv(model)