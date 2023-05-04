from mmcv.utils import Registry

from .simple_architecture import SimpleArchitecture


ARCHITECTURES = Registry('architectures')

ARCHITECTURES.register_module(name='SimpleArchitecture', module=SimpleArchitecture)


def build_architecture(cfg):
    """Build architecture."""
    return ARCHITECTURES.build(cfg)