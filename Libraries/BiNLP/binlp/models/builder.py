from mmcv.cnn import MODELS as MMCV_MODELS
from mmcv.utils import Registry


def build_from_cfg(cfg, registry, default_args=None):
    if cfg is None:
        return None
    return MMCV_MODELS.build_func(cfg, registry, default_args)

def build_transformer_from_cfg(cfg, registry, default_args=None):
    cfg_ = cfg.copy()
    transformer_type = cfg_.pop('type')
    if transformer_type not in registry:
        raise KeyError(f'Unrecognized transformer type {transformer_type}')
    else:
        transformer_cls = registry.get(transformer_type)
    return transformer_cls.from_pretrained(**cfg_)


TRANSFORMERS = Registry('transformers', build_func=build_transformer_from_cfg)
CLASSIFIER = Registry('classifier', build_func=build_from_cfg)

def build_transformer(cfg):
    """Build transformer."""
    return TRANSFORMERS.build(cfg)

def build_classifier(cfg):
    """Build classifier."""
    return CLASSIFIER.build(cfg)
