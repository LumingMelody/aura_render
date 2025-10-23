"""
IMS转换器测试用例

测试VGP到IMS的各种转换功能
"""

import json
import logging
from ims_converter import IMSConverter

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def test_basic_conversion():
    """测试基础转换功能"""
    logger.info("=" * 60)
    logger.info("测试1: 基础转换 (转场+滤镜)")
    logger.info("=" * 60)

    # 模拟VGP输出
    vgp_result = {
        "filter_sequence_id": [
            {
                "id": "clip_001",
                "source_url": "https://example.com/video1.mp4",
                "start": 0.0,
                "end": 5.0,
                "duration": 5.0,
                "transition_out": {
                    "type": "cross_dissolve",
                    "name": "叠化",
                    "duration": 1.2
                },
                "color_filter": {
                    "preset": "cinematic",
                    "name": "电��感",
                    "intensity": 0.8,
                    "applied_params": {
                        "contrast": 1.15,
                        "saturation": 0.95,
                        "temperature": 0.05
                    }
                }
            },
            {
                "id": "clip_002",
                "source_url": "https://example.com/video2.mp4",
                "start": 5.0,
                "end": 10.0,
                "duration": 5.0,
                "transition_out": {
                    "type": "fade_in_out",
                    "name": "淡入淡出",
                    "duration": 1.0
                },
                "color_filter": {
                    "preset": "vibrant",
                    "name": "鲜艳",
                    "intensity": 1.0
                }
            }
        ]
    }

    # 创建转换器 (使用预设模式)
    converter = IMSConverter(use_filter_preset=True)

    # 执行转换
    ims_timeline = converter.convert(vgp_result)

    # 输出结果
    print("\n" + "=" * 60)
    print("IMS Timeline结果:")
    print("=" * 60)
    print(json.dumps(ims_timeline, indent=2, ensure_ascii=False))

    # 获取摘要
    summary = converter.get_conversion_summary(vgp_result)
    print("\n" + "=" * 60)
    print("转换摘要:")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def test_precise_filter_conversion():
    """测试精确滤镜参数转换"""
    logger.info("\n" + "=" * 60)
    logger.info("测试2: 精确滤镜参数转换")
    logger.info("=" * 60)

    vgp_result = {
        "filter_sequence_id": [
            {
                "id": "clip_001",
                "source_url": "https://example.com/video1.mp4",
                "start": 0.0,
                "end": 5.0,
                "duration": 5.0,
                "color_filter": {
                    "preset": "custom",
                    "applied_params": {
                        "brightness": 1.3,      # 增亮30%
                        "contrast": 1.2,        # 增加对比度20%
                        "saturation": 0.8,      # 降低饱和度20%
                        "temperature": 0.3,     # 暖色调
                        "vignette": 0.2         # 轻微暗角
                    }
                }
            }
        ]
    }

    # 使用精确参数模式
    converter = IMSConverter(use_filter_preset=False)
    ims_timeline = converter.convert(vgp_result)

    print("\n" + "=" * 60)
    print("精确参数转换结果:")
    print("=" * 60)
    print(json.dumps(ims_timeline, indent=2, ensure_ascii=False))


def test_effects_conversion():
    """测试特效转换"""
    logger.info("\n" + "=" * 60)
    logger.info("测试3: 特效转换")
    logger.info("=" * 60)

    vgp_result = {
        "effects_sequence_id": [
            {
                "id": "clip_001",
                "source_url": "https://example.com/video1.mp4",
                "start": 0.0,
                "end": 5.0,
                "duration": 5.0,
                "visual_effects": [
                    {
                        "name": "镜头光晕",
                        "type": "lens_flare",
                        "position": {"x": 960, "y": 540},
                        "opacity": 0.6
                    },
                    {
                        "name": "胶片颗粒",
                        "type": "film_grain",
                        "intensity": 0.3
                    }
                ]
            },
            {
                "id": "clip_002",
                "source_url": "https://example.com/video2.mp4",
                "start": 5.0,
                "end": 10.0,
                "duration": 5.0,
                "visual_effects": [
                    {
                        "name": "下雨",
                        "type": "rain"
                    }
                ]
            }
        ]
    }

    converter = IMSConverter()
    ims_timeline = converter.convert(vgp_result)

    print("\n" + "=" * 60)
    print("特效转换结果:")
    print("=" * 60)
    print(json.dumps(ims_timeline, indent=2, ensure_ascii=False))


def test_text_overlay_conversion():
    """测试花字转换"""
    logger.info("\n" + "=" * 60)
    logger.info("测试4: 花字/文字叠加转换")
    logger.info("=" * 60)

    vgp_result = {
        "text_overlay_track_id": {
            "track_name": "text_overlay",
            "track_type": "text",
            "clips": [
                {
                    "text": "太好吃了!",
                    "start": 2.0,
                    "duration": 3.0,
                    "position": "top-center",
                    "style": {
                        "color": "#FFFFFF",
                        "stroke": "#000000",
                        "size": 36,
                        "bold": True
                    }
                },
                {
                    "text": "这一刻",
                    "start": 7.0,
                    "duration": 2.5,
                    "position": "center",
                    "style": {
                        "color": "#FFD700",
                        "stroke": "#000000",
                        "size": 48,
                        "bold": True
                    }
                }
            ]
        }
    }

    converter = IMSConverter()
    ims_timeline = converter.convert(vgp_result)

    print("\n" + "=" * 60)
    print("花字转换结果:")
    print("=" * 60)
    print(json.dumps(ims_timeline, indent=2, ensure_ascii=False))


def test_auxiliary_media_conversion():
    """测试辅助媒体转换"""
    logger.info("\n" + "=" * 60)
    logger.info("测试5: 辅助媒���/叠加转换")
    logger.info("=" * 60)

    vgp_result = {
        "auxiliary_track_id": {
            "track_name": "auxiliary",
            "track_type": "overlay_media",
            "clips": [
                {
                    "media_id": "chart_001",
                    "file_path": "https://example.com/assets/chart.png",
                    "start": 3.0,
                    "duration": 4.0,
                    "type": "image",
                    "placement": "overlay",
                    "position": "top-right",
                    "opacity": 0.95
                },
                {
                    "media_id": "logo_002",
                    "file_path": "https://example.com/assets/logo.png",
                    "start": 0.0,
                    "duration": 10.0,
                    "type": "image",
                    "placement": "overlay",
                    "position": "bottom-right",
                    "opacity": 0.7
                }
            ]
        }
    }

    converter = IMSConverter()
    ims_timeline = converter.convert(vgp_result)

    print("\n" + "=" * 60)
    print("辅助媒体转换结果:")
    print("=" * 60)
    print(json.dumps(ims_timeline, indent=2, ensure_ascii=False))


def test_complete_conversion():
    """测试完整转换 (所有功能组合)"""
    logger.info("\n" + "=" * 60)
    logger.info("测试6: 完整转换 (转场+滤镜+特效+花字+辅助媒体)")
    logger.info("=" * 60)

    vgp_result = {
        "effects_sequence_id": [
            {
                "id": "clip_001",
                "source_url": "https://example.com/video1.mp4",
                "start": 0.0,
                "end": 5.0,
                "duration": 5.0,
                "transition_out": {
                    "type": "cross_dissolve",
                    "duration": 1.0
                },
                "color_filter": {
                    "preset": "cinematic",
                    "intensity": 0.8
                },
                "visual_effects": [
                    {
                        "type": "lens_flare",
                        "name": "镜头光晕"
                    }
                ]
            },
            {
                "id": "clip_002",
                "source_url": "https://example.com/video2.mp4",
                "start": 5.0,
                "end": 10.0,
                "duration": 5.0,
                "transition_out": {
                    "type": "zoom_transition",
                    "duration": 0.8
                },
                "color_filter": {
                    "preset": "vibrant",
                    "intensity": 1.0
                },
                "visual_effects": []
            }
        ],
        "text_overlay_track_id": {
            "clips": [
                {
                    "text": "太震撼了!",
                    "start": 2.0,
                    "duration": 2.0,
                    "position": "top-center",
                    "style": {
                        "color": "#FFFFFF",
                        "stroke": "#000000",
                        "size": 42,
                        "bold": True
                    }
                }
            ]
        },
        "auxiliary_track_id": {
            "clips": [
                {
                    "file_path": "https://example.com/logo.png",
                    "start": 0.0,
                    "duration": 10.0,
                    "type": "image",
                    "position": "bottom-right",
                    "opacity": 0.8
                }
            ]
        }
    }

    converter = IMSConverter(use_filter_preset=True)

    # 转换为完整的IMS请求
    ims_request = converter.convert_to_ims_request(
        vgp_result,
        output_config={
            "MediaURL": "oss://my-bucket/output/video.mp4",
            "Width": 1920,
            "Height": 1080,
            "VideoCodec": "H.264",
            "AudioCodec": "AAC"
        }
    )

    print("\n" + "=" * 60)
    print("完整IMS请求体:")
    print("=" * 60)
    print(json.dumps(ims_request, indent=2, ensure_ascii=False))

    # 获取转换摘要
    summary = converter.get_conversion_summary(vgp_result)
    print("\n" + "=" * 60)
    print("转换摘要:")
    print("=" * 60)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("*" + " " * 58 + "*")
    print("*" + "  IMS转换器测试套件".center(56) + "*")
    print("*" + " " * 58 + "*")
    print("*" * 60)
    print("\n")

    # 运行所有测试
    test_basic_conversion()
    test_precise_filter_conversion()
    test_effects_conversion()
    test_text_overlay_conversion()
    test_auxiliary_media_conversion()
    test_complete_conversion()

    print("\n" + "=" * 60)
    print("所有测试完成!")
    print("=" * 60)
