import mmcls
# import mmdet

from .resnet import ResNet, ResNetV1c, ResNetV1d
from .resnet_cifar import ResNet_CIFAR
from .vgg import VGG
from .ssd_vgg import SSDVGG
from .reactnet import ReActNet


def init_backbones():
    model_dict = {
        'BiBench_ResNet': ResNet,
        'BiBench_ResNetV1c': ResNetV1c,
        'BiBench_ResNetV1d': ResNetV1d,
        'BiBench_ResNet_CIFAR': ResNet_CIFAR,
        'BiBench_VGG': VGG,
        'BiBench_SSDVGG': SSDVGG,
        'BiBench_ReActNet': ReActNet
    }

    try:
        from mmcls.models import BACKBONES as mmcls_BACKBONES
        for k, v in model_dict.items():
            mmcls_BACKBONES.register_module(name=k, module=v)
    except:
        pass

    try:
        from mmdet.models import BACKBONES as mmdet_BACKBONES
        for k, v in model_dict.items():
            mmdet_BACKBONES.register_module(name=k, module=v)
    except:
        pass

    try:
        from bipc.models import BACKBONES as bipc_BACKBONES
        for k, v in model_dict.items():
            bipc_BACKBONES.register_module(name=k, module=v)
    except:
        pass

    try:
        from binlp.models import BACKBONES as binlp_BACKBONES
        for k, v in model_dict.items():
            binlp_BACKBONES.register_module(name=k, module=v)
    except:
        pass
    
    try:
        from bispeech.models import BACKBONES as bispeech_BACKBONES
        for k, v in model_dict.items():
            bispeech_BACKBONES.register_module(name=k, module=v)
    except:
        pass
        
