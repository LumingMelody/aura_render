"""
测试TTS音频生成功能

验证修复后的audio_tts_integration模块是否能正常生成音频
"""

import asyncio
import json
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from video_generate_protocol.nodes.audio_tts_integration import (
    generate_tts_audio_track,
    build_ims_audio_tracks,
    integrate_tts_to_timeline
)


async def test_tts_generation():
    """测试TTS音频生成"""

    # 准备测试用的字幕序列
    subtitle_sequence = {
        "clips": [
            {
                "start": 0.0,
                "end": 3.0,
                "duration": 3.0,
                "text": "展示产品便携设计与高清画质"
            },
            {
                "start": 3.0,
                "end": 5.0,
                "duration": 2.0,
                "text": "通过人的手部动作展现产品的易用性"
            },
            {
                "start": 6.0,
                "end": 9.0,
                "duration": 3.0,
                "text": "突出产品不仅适合家用也适合商业场景"
            },
            {
                "start": 9.0,
                "end": 10.0,
                "duration": 1.0,
                "text": "利用用户口碑来增强信任感"
            }
        ]
    }

    print("=" * 80)
    print("🎤 测试1: 生成分段TTS音频（推荐模式）")
    print("=" * 80)

    # 测试分段模式
    audio_track_info = await generate_tts_audio_track(
        subtitle_sequence,
        voice="Cherry",
        speed=1.0,
        upload_to_oss=True,
        use_segmented=True
    )

    if audio_track_info:
        print(f"\n✅ TTS音频生成成功！")
        print(f"   模式: {audio_track_info.get('mode')}")
        print(f"   总时长: {audio_track_info.get('total_duration')}秒")
        print(f"   音频片段数量: {len(audio_track_info.get('audio_clips', []))}")

        # 打印每个音频片段的信息
        for i, clip in enumerate(audio_track_info.get('audio_clips', []), 1):
            print(f"\n   片段 {i}:")
            print(f"      文本: {clip.get('text')}")
            print(f"      时间: {clip.get('timeline_in')}s - {clip.get('timeline_out')}s")
            print(f"      URL: {clip.get('audio_url')[:80]}...")

        # 测试构建IMS AudioTracks
        print(f"\n{'=' * 80}")
        print("🎵 测试2: 构建IMS AudioTracks格式")
        print("=" * 80)

        audio_tracks = build_ims_audio_tracks(audio_track_info)

        if audio_tracks:
            print(f"\n✅ IMS AudioTracks构建成功！")
            print(f"   AudioTracks数量: {len(audio_tracks)}")
            print(f"\n   完整结构:")
            print(json.dumps(audio_tracks, indent=2, ensure_ascii=False))
        else:
            print("\n❌ IMS AudioTracks构建失败")

        # 测试集成到Timeline
        print(f"\n{'=' * 80}")
        print("🎬 测试3: 集成到IMS Timeline")
        print("=" * 80)

        # 创建一个简单的timeline
        timeline = {
            "VideoTracks": [{
                "VideoTrackClips": [
                    {"MediaURL": "https://example.com/video1.mp4"}
                ]
            }]
        }

        # 集成TTS音频
        updated_timeline = await integrate_tts_to_timeline(
            timeline,
            subtitle_sequence,
            voice="Cherry",
            speed=1.0,
            upload_to_oss=True,
            use_segmented=True
        )

        if updated_timeline.get("AudioTracks"):
            print(f"\n✅ TTS音频已成功集成到Timeline！")
            print(f"   AudioTracks数量: {len(updated_timeline['AudioTracks'])}")

            total_audio_clips = sum(
                len(track.get("AudioTrackClips", []))
                for track in updated_timeline["AudioTracks"]
            )
            print(f"   总音频片段数: {total_audio_clips}")

            print(f"\n   完整Timeline:")
            print(json.dumps(updated_timeline, indent=2, ensure_ascii=False))
        else:
            print("\n❌ Timeline中没有AudioTracks")

    else:
        print("\n❌ TTS音频生成失败")

    print(f"\n{'=' * 80}")
    print("✅ 测试完成")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(test_tts_generation())
