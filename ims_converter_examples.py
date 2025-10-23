"""
IMS转换器快速使用示例

演示如何将VGP输出转换为阿里云IMS Timeline
"""

from ims_converter import IMSConverter
import json


def example_1_basic():
    """示例1: 基础转换 - 转场和滤镜"""
    print("\n" + "=" * 60)
    print("示例1: 基础转换 - 转场和滤镜")
    print("=" * 60)

    vgp_result = {
        "filter_sequence_id": [
            {
                "source_url": "oss://my-bucket/video1.mp4",
                "start": 0.0,
                "end": 5.0,
                "transition_out": {
                    "type": "cross_dissolve",
                    "duration": 1.0
                },
                "color_filter": {
                    "preset": "cinematic"
                }
            },
            {
                "source_url": "oss://my-bucket/video2.mp4",
                "start": 5.0,
                "end": 10.0,
                "transition_out": {
                    "type": "zoom_transition",
                    "duration": 0.8
                },
                "color_filter": {
                    "preset": "vibrant"
                }
            }
        ]
    }

    converter = IMSConverter(use_filter_preset=True)
    ims_timeline = converter.convert(vgp_result)

    print(json.dumps(ims_timeline, indent=2, ensure_ascii=False))


def example_2_complete():
    """示例2: 完整示例 - 所有功能"""
    print("\n" + "=" * 60)
    print("示例2: 完整示例 - 转场+滤镜+特效+花字+Logo")
    print("=" * 60)

    vgp_result = {
        "effects_sequence_id": [
            {
                "source_url": "oss://my-bucket/clip1.mp4",
                "start": 0.0,
                "end": 8.0,
                "duration": 8.0,
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
                "source_url": "oss://my-bucket/clip2.mp4",
                "start": 8.0,
                "end": 15.0,
                "duration": 7.0,
                "transition_out": {
                    "type": "burn",
                    "duration": 1.2
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
                    "text": "精彩瞬间",
                    "start": 3.0,
                    "duration": 2.5,
                    "position": "top-center",
                    "style": {
                        "color": "#FFFFFF",
                        "stroke": "#000000",
                        "size": 48,
                        "bold": True
                    }
                },
                {
                    "text": "震撼来袭!",
                    "start": 10.0,
                    "duration": 2.0,
                    "position": "center",
                    "style": {
                        "color": "#FFD700",
                        "stroke": "#000000",
                        "size": 56,
                        "bold": True
                    }
                }
            ]
        },
        "auxiliary_track_id": {
            "clips": [
                {
                    "file_path": "oss://my-bucket/logo.png",
                    "start": 0.0,
                    "duration": 15.0,
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
            "MediaURL": "oss://my-bucket/output/final_video.mp4",
            "Width": 1920,
            "Height": 1080,
            "VideoCodec": "H.264",
            "AudioCodec": "AAC",
            "FrameRate": 30
        }
    )

    print(json.dumps(ims_request, indent=2, ensure_ascii=False))

    # 获取转换摘要
    summary = converter.get_conversion_summary(vgp_result)
    print("\n转换摘要:")
    print(f"  - 总片段数: {summary['total_clips']}")
    print(f"  - 转场效果: {summary['transitions']}")
    print(f"  - 滤镜应用: {summary['filters']}")
    print(f"  - 特效应用: {summary['effects']}")
    print(f"  - 文字数量: {summary['texts']}")
    print(f"  - 叠加媒体: {summary['overlays']}")


def example_3_precise_filter():
    """示例3: 精确滤镜参数转换"""
    print("\n" + "=" * 60)
    print("示例3: 精确滤镜参数转换")
    print("=" * 60)

    vgp_result = {
        "filter_sequence_id": [
            {
                "source_url": "oss://my-bucket/video.mp4",
                "start": 0.0,
                "end": 10.0,
                "color_filter": {
                    "preset": "custom",
                    "applied_params": {
                        "brightness": 1.2,      # 增亮20%
                        "contrast": 1.15,       # 增加对比15%
                        "saturation": 0.9,      # 降低饱和10%
                        "temperature": 0.2,     # 轻微暖色
                        "vignette": 0.15        # 轻微暗角
                    }
                }
            }
        ]
    }

    # 使用精确参数模式
    converter = IMSConverter(use_filter_preset=False)
    ims_timeline = converter.convert(vgp_result)

    print("使用精确参数模式转换:")
    print(json.dumps(ims_timeline, indent=2, ensure_ascii=False))


def example_4_production_ready():
    """示例4: 生产环境可用的完整配置"""
    print("\n" + "=" * 60)
    print("示例4: 生产环境配置")
    print("=" * 60)

    # 模拟真实的VGP输出
    vgp_result = {
        "effects_sequence_id": [
            {
                "id": "clip_001",
                "source_url": "oss://video-production/raw/interview_part1.mp4",
                "start": 0.0,
                "end": 12.5,
                "duration": 12.5,
                "shot_type": "medium",
                "transition_out": {
                    "type": "cross_dissolve",
                    "name": "叠化",
                    "duration": 1.0
                },
                "color_filter": {
                    "preset": "cinematic",
                    "name": "电影感",
                    "intensity": 0.85,
                    "applied_params": {
                        "contrast": 1.15,
                        "saturation": 0.95,
                        "temperature": 0.05
                    }
                },
                "visual_effects": [
                    {
                        "type": "film_grain",
                        "name": "胶片颗粒",
                        "intensity": 0.2
                    }
                ]
            },
            {
                "id": "clip_002",
                "source_url": "oss://video-production/raw/broll_cityscape.mp4",
                "start": 12.5,
                "end": 20.0,
                "duration": 7.5,
                "shot_type": "wide",
                "transition_out": {
                    "type": "zoom_transition",
                    "name": "缩放转场",
                    "duration": 0.8
                },
                "color_filter": {
                    "preset": "vibrant",
                    "name": "鲜艳",
                    "intensity": 1.0
                },
                "visual_effects": []
            },
            {
                "id": "clip_003",
                "source_url": "oss://video-production/raw/interview_part2.mp4",
                "start": 20.0,
                "end": 32.0,
                "duration": 12.0,
                "shot_type": "close_up",
                "transition_out": {
                    "type": "fade_in_out",
                    "name": "淡入淡出",
                    "duration": 1.5
                },
                "color_filter": {
                    "preset": "natural",
                    "name": "自然",
                    "intensity": 0.9
                },
                "visual_effects": []
            }
        ],
        "text_overlay_track_id": {
            "track_name": "titles",
            "clips": [
                {
                    "text": "专家访谈",
                    "start": 1.0,
                    "duration": 3.0,
                    "position": "bottom-center",
                    "style": {
                        "color": "#FFFFFF",
                        "stroke": "#000000",
                        "size": 36,
                        "bold": True
                    }
                },
                {
                    "text": "城市风光",
                    "start": 13.0,
                    "duration": 2.5,
                    "position": "top-left",
                    "style": {
                        "color": "#FFD700",
                        "stroke": "#000000",
                        "size": 32,
                        "bold": False
                    }
                }
            ]
        },
        "auxiliary_track_id": {
            "track_name": "branding",
            "clips": [
                {
                    "media_id": "watermark",
                    "file_path": "oss://video-production/assets/company_logo.png",
                    "start": 0.0,
                    "duration": 32.0,
                    "type": "image",
                    "position": "bottom-right",
                    "opacity": 0.7
                }
            ]
        }
    }

    converter = IMSConverter(use_filter_preset=True)

    # 生成IMS请求
    ims_request = converter.convert_to_ims_request(
        vgp_result,
        output_config={
            "MediaURL": "oss://video-production/output/final_edit_v1.mp4",
            "Width": 1920,
            "Height": 1080,
            "VideoCodec": "H.264",
            "AudioCodec": "AAC",
            "FrameRate": 30,
            "VideoBitrate": "5000",
            "AudioBitrate": "128"
        }
    )

    print("生产环境IMS请求:")
    print(json.dumps(ims_request, indent=2, ensure_ascii=False))

    summary = converter.get_conversion_summary(vgp_result)
    print("\n转换摘要:")
    for key, value in summary.items():
        if key != "warnings":
            print(f"  - {key}: {value}")


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("*  IMS转换器使用示例  *".center(60))
    print("*" * 60)

    example_1_basic()
    example_2_complete()
    example_3_precise_filter()
    example_4_production_ready()

    print("\n" + "=" * 60)
    print("所有示例完成!")
    print("=" * 60)
    print("\n更多信息请参考: ims_converter/README.md")
