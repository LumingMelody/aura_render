"""
IMS Converter - VGP to Aliyun IMS Timeline Converter

将VGP (Video Generate Protocol) 的输出转换为阿里云智能媒体服务(IMS)的Timeline格式
"""

from .converter import IMSConverter
from .configs.mappings import (
    VGP_TO_IMS_TRANSITION,
    VGP_TO_IMS_FILTER_PRESET,
    VGP_TO_IMS_EFFECT,
    VGP_TO_IMS_FLOWER_STYLE
)

__all__ = [
    'IMSConverter',
    'VGP_TO_IMS_TRANSITION',
    'VGP_TO_IMS_FILTER_PRESET',
    'VGP_TO_IMS_EFFECT',
    'VGP_TO_IMS_FLOWER_STYLE'
]

__version__ = '1.0.0'
