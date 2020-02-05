# -*- coding: utf-8 -*-
# @Time    : 2019/12/4 9:46 PM
# @Author  : morningstarwang
# @FileName: __init__.py
# @Blog: wangchenxing.com


from .nin_instance import NINInstance
from .dcgan_demo_instance import DCGANDemoInstance
from .AlexNet_instance import AlexNetInstance
from .GoogLeNet_instance import GoogLeNetInstance
from .AGG16_instance import AGG16Instance
from .ResNet101_instance import ResNet101Instance
from .DenseNet201_instance import DenseNet201Instance
from .gcn_instance import GCNInstance
from .baseline_2_instance import Baseline2Instance
from .baseline_4_instance import Baseline4Instance
from .RNN_instance import RNNInstance
from .baseline_gcn_instance import BaselineGCNInstance
from .RNN_instance import RNNInstance

__all__ = ['NINInstance', 'DCGANDemoInstance', 'AlexNetInstance', 'GoogLeNetInstance',
           'AGG16Instance', 'ResNet101Instance', 'DenseNet201Instance', 'GCNInstance', 'Baseline2Instance',
           'RNNInstance', 'Baseline4Instance', 'BaselineGCNInstance']
