# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 9:46 PM
# @Author  : morningstarwang
# @FileName: __init__.py
# @Blog: wangchenxing.com

from .nin_model import NINModel
from .nin_feature_output_model import NINFeatureOutputModel
from .dcgan_demo_model import DCGANDemoModel
from .AlexNet import AlexNet
from .GoogLeNet import GoogLeNet
from .AGG16 import AGG16
from .DenseNet201 import DenseNet201
from .ResNet101 import ResNet101
from .gcn_model import GCNModel
from .baseline_2_model import BaseLine2Model
from .baseline_4_model import BaseLine4Model
from .baseline_gcn_model import BaselineGCNModel
from .dilated_model import DilatedModel
from .nin_dilated_model import NINDilatedModel
from .RNN import RNN
from .dilated_feature_output_model import DilatedFeatureOutputModel
from .nin_dilated_feature_output_model import NINDilatedFeatureOutputModel

__all__ = ['NINModel', 'DCGANDemoModel', 'GoogLeNet', 'AlexNet',
           'AGG16', 'DenseNet201', 'ResNet101', 'NINFeatureOutputModel', 'GCNModel', 'BaseLine2Model', 'RNN',
           'BaseLine4Model', 'BaselineGCNModel', 'DilatedModel', 'DilatedFeatureOutputModel',
           'NINDilatedFeatureOutputModel']
