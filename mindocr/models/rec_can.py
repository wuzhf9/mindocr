from ._registry import register_model
from .backbones.mindcv_models.utils import load_pretrained
from .base_model import BaseModel

__all__ = ["CAN", "can_densenet"]

class CAN(BaseModel):
    def __init__(self, config):
        BaseModel.__init__(self, config)


@register_model
def can_densenet(pretrained=False, **kwargs):
    model_config = {
        "backbone": {
            "name": "rec_densenet",
            "pretrained": False
        },
        "head": {
            "name": "CANHead",
            "out_channels": 111,
            "ratio": 16,
            "attdecoder_args": {
                "input_size": 256,
                "hidden_size": 256,
                "out_channels": 684,
                "attention_dim": 512,
                "word_num": 111,
                "counting_num": 111,
                "word_conv_kernel": 1,
                "dropout": 0.5
            }
        }
    }
    model = CAN(model_config)

    # load prtrained weights
    if pretrained:
        raise NotImplementedError("The default pretrained checkpoint for `can_densenet` backbone does not exist.")
    
    return model