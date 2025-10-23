"""
测试IMS转换器集成修复

验证转场、滤镜、特效是否正确集成到Timeline
"""

import json


def test_ims_converter_with_vgp_data():
    """测试IMS转换器处理真实VGP数据"""
    from ims_converter import IMSConverter

    # 模拟从日志中提取的VGP数据
    vgp_result = {
        "filter_sequence_id": [
            {
                "id": "clip_f284ecf0",
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
                    "preset": "cyberpunk",
                    "name": "赛博朋克",
                    "intensity": 0.8,
                    "applied_params": {
                        "contrast": 1.24,
                        "saturation": 1.32,
                        "temperature": 0.04
                    }
                }
            },
            {
                "id": "clip_40fc4155",
                "source_url": "https://example.com/video2.mp4",
                "start": 5.5,
                "end": 10.5,
                "duration": 5.0,
                "transition_out": {
                    "type": "none",
                    "duration": 0.0
                },
                "color_filter": {
                    "preset": "cyberpunk",
                    "name": "赛博朋克",
                    "intensity": 0.8
                }
            }
        ]
    }

    print("=" * 60)
    print("测试: IMS转换器处理VGP数据")
    print("=" * 60)

    # 创建转换器
    converter = IMSConverter(use_filter_preset=True)

    # 转换
    result = converter.convert(vgp_result)

    print("\n✅ 转换成功!")
    print("\n转换后的IMS Timeline:")
    print(json.dumps(result, indent=2, ensure_ascii=False))

    # 获取摘要
    summary = converter.get_conversion_summary(vgp_result)
    print("\n转换摘要:")
    print(f"  - 总片段数: {summary['total_clips']}")
    print(f"  - 转场数量: {summary['transitions']}")
    print(f"  - 滤镜数量: {summary['filters']}")
    print(f"  - 特效数量: {summary['effects']}")

    # 验证关键部分
    print("\n验证:")
    video_tracks = result.get("VideoTracks", [])
    if video_tracks and video_tracks[0].get("VideoTrackClips"):
        clips = video_tracks[0]["VideoTrackClips"]
        print(f"  ✅ VideoTracks包含 {len(clips)} 个片段")

        # 检查转场
        clip_with_transition = clips[0]
        if clip_with_transition.get("Effects"):
            print(f"  ✅ Clip 1 有转场效果: {clip_with_transition['Effects']}")
        else:
            print(f"  ❌ Clip 1 缺少转场效果")
    else:
        print(f"  ❌ VideoTracks为空")

    effect_tracks = result.get("EffectTracks", [])
    if effect_tracks:
        print(f"  ✅ EffectTracks包含 {len(effect_tracks)} 个轨道")
        for i, track in enumerate(effect_tracks):
            items = track.get("EffectTrackItems", [])
            print(f"     轨道 {i+1}: {len(items)} 个效果")
            if items:
                print(f"       示例: {items[0]}")
    else:
        print(f"  ❌ EffectTracks为空")

    return result


def test_integration_logic():
    """测试集成逻辑模拟"""
    print("\n" + "=" * 60)
    print("测试: 模拟Timeline集成逻辑")
    print("=" * 60)

    # 模拟基础timeline
    timeline = {
        "VideoTracks": [{
            "VideoTrackClips": [
                {"MediaURL": "https://video1.mp4", "Effects": []},
                {"MediaURL": "https://video2.mp4", "Effects": []}
            ]
        }]
    }

    # 模拟VGP上下文
    vgp_context = {
        "filter_sequence_id": [
            {
                "source_url": "https://video1.mp4",
                "transition_out": {"type": "fade", "duration": 1.0},
                "color_filter": {"preset": "cinematic"}
            },
            {
                "source_url": "https://video2.mp4",
                "transition_out": {"type": "none"},
                "color_filter": {"preset": "vibrant"}
            }
        ]
    }

    print("\n原始Timeline:")
    print(json.dumps(timeline, indent=2, ensure_ascii=False))

    # 应用转换
    from ims_converter import IMSConverter
    converter = IMSConverter(use_filter_preset=True)

    vgp_result = {
        "filter_sequence_id": vgp_context["filter_sequence_id"]
    }

    converted = converter.convert(vgp_result)

    # 合并
    if converted.get("VideoTracks"):
        converted_clips = converted["VideoTracks"][0].get("VideoTrackClips", [])
        for i, clip in enumerate(timeline["VideoTracks"][0]["VideoTrackClips"]):
            if i < len(converted_clips) and converted_clips[i].get("Effects"):
                clip["Effects"] = converted_clips[i]["Effects"]
                print(f"\n✅ Clip {i+1}: 添加转场 {clip['Effects']}")

    if converted.get("EffectTracks"):
        timeline["EffectTracks"] = converted["EffectTracks"]
        print(f"\n✅ 添加EffectTracks: {len(converted['EffectTracks'])} 个轨道")

    print("\n最终Timeline:")
    print(json.dumps(timeline, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    print("\n")
    print("*" * 60)
    print("*  IMS转换器集成修复测试  *".center(60))
    print("*" * 60)
    print("\n")

    try:
        # 测试1: 基础转换
        test_ims_converter_with_vgp_data()

        # 测试2: 集成逻辑
        test_integration_logic()

        print("\n" + "=" * 60)
        print("✅ 所有测试完成!")
        print("=" * 60)

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
