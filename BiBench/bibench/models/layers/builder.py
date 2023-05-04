import torch.nn as nn
from mmcv.cnn import ACTIVATION_LAYERS, CONV_LAYERS
from .bireal import BiRealConv2d, BiRealConv1d, BiRealLinear
from .xnor import XNORConv2d, XNORConv1d, XNORLinear
from .bnn import BNNConv2d, BNNConv1d, BNNLinear
from .react import ReActConv2d, ReActConv1d, ReActLinear
from .dorefa import DoReFaConv2d, DoReFaConv1d, DoReFaLinear
from .xnorplusplus import XNORPlusPlusConv2d, XNORPlusPlusConv1d, XNORPlusPlusLinear
from .recu import ReCUConv2d, ReCUConv1d, ReCULinear
from .fda import FDAConv2d, FDAConv1d, FDALinear

from mmcv.cnn import CONV_LAYERS


def init_layers():
    ACTIVATION_LAYERS.register_module('Hardtanh', module=nn.Hardtanh)
    CONV_LAYERS.register_module(name='Linear', module=nn.Linear)
    
    CONV_LAYERS.register_module(name='BiRealConv', module=BiRealConv2d)
    CONV_LAYERS.register_module(name='XNORConv', module=XNORConv2d)
    CONV_LAYERS.register_module(name='BNNConv', module=BNNConv2d)
    CONV_LAYERS.register_module(name='ReActConv', module=ReActConv2d)
    CONV_LAYERS.register_module(name='DoReFaConv', module=DoReFaConv2d)
    CONV_LAYERS.register_module(name='XNORPlusPlusConv', module=XNORPlusPlusConv2d)
    CONV_LAYERS.register_module(name='ReCUConv', module=ReCUConv2d)
    CONV_LAYERS.register_module(name='FDAConv', module=FDAConv2d)

    CONV_LAYERS.register_module(name='BiRealConv2d', module=BiRealConv2d)
    CONV_LAYERS.register_module(name='XNORConv2d', module=XNORConv2d)
    CONV_LAYERS.register_module(name='BNNConv2d', module=BNNConv2d)
    CONV_LAYERS.register_module(name='ReActConv2d', module=ReActConv2d)
    CONV_LAYERS.register_module(name='DoReFaConv2d', module=DoReFaConv2d)
    CONV_LAYERS.register_module(name='XNORPlusPlusConv2d', module=XNORPlusPlusConv2d)
    CONV_LAYERS.register_module(name='ReCUConv2d', module=ReCUConv2d)
    CONV_LAYERS.register_module(name='FDAConv2d', module=FDAConv2d)

    CONV_LAYERS.register_module(name='BiRealConv1d', module=BiRealConv1d)
    CONV_LAYERS.register_module(name='XNORConv1d', module=XNORConv1d)
    CONV_LAYERS.register_module(name='BNNConv1d', module=BNNConv1d)
    CONV_LAYERS.register_module(name='ReActConv1d', module=ReActConv1d)
    CONV_LAYERS.register_module(name='DoReFaConv1d', module=DoReFaConv1d)
    CONV_LAYERS.register_module(name='XNORPlusPlusConv1d', module=XNORPlusPlusConv1d)
    CONV_LAYERS.register_module(name='ReCUConv1d', module=ReCUConv1d)
    CONV_LAYERS.register_module(name='FDAConv1d', module=FDAConv1d)

    CONV_LAYERS.register_module(name='BNNLinear', module=BNNLinear)
    CONV_LAYERS.register_module(name='XNORLinear', module=XNORLinear)
    CONV_LAYERS.register_module(name='DoReFaLinear', module=DoReFaLinear)
    CONV_LAYERS.register_module(name='BiRealLinear', module=BiRealLinear)
    CONV_LAYERS.register_module(name='XNORPlusPlusLinear', module=XNORPlusPlusLinear)
    CONV_LAYERS.register_module(name='ReActLinear', module=ReActLinear)
    CONV_LAYERS.register_module(name='ReCULinear', module=ReCULinear)
    CONV_LAYERS.register_module(name='FDALinear', module=FDALinear)

