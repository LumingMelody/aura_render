"""
测试分段TTS音频生成功能

验证音画同步是否正确
"""
import asyncio
import json
from video_generate_protocol.nodes.audio_tts_integration import (
    generate_tts_audio_track,
    build_ims_audio_tracks
)


async def test_segmented_tts():
    """测试分段TTS生成"""

    print("\n" + "="*80)
    print("🧪 测试分段TTS音频生成（音画同步）")
    print("="*80 + "\n")

    # 模拟字幕序列（与你的实际运行日志中的字幕一致）
    subtitle_sequence = {
        "clips": [
            {
                "start": 0.0,
                "end": 5.0,
                "text": "在家享受影院级体验！",
                "duration": 5.0
            },
            {
                "start": 5.0,
                "end": 10.0,
                "text": "清晰！便捷！智能！",
                "duration": 5.0
            }
        ]
    }

    print(f"📋 测试字幕序列:")
    for i, clip in enumerate(subtitle_sequence["clips"]):
        print(f"   {i+1}. \"{clip['text']}\" ({clip['start']:.1f}s - {clip['end']:.1f}s)")

    # ========== 测试1: 分段模式 ==========
    print(f"\n{'='*80}")
    print("🎯 测试1: 分段生成模式（推荐）")
    print("="*80 + "\n")

    audio_track_info = await generate_tts_audio_track(
        subtitle_sequence,
        voice="Cherry",
        speed=1.0,
        upload_to_oss=True,
        use_segmented=True  # 使用分段模式
    )

    if audio_track_info:
        print(f"\n✅ 分段TTS生成成功:")
        print(f"   模式: {audio_track_info['mode']}")
        print(f"   总时长: {audio_track_info['total_duration']}秒")
        print(f"   音频片段数: {len(audio_track_info['audio_clips'])}")

        print(f"\n📊 音频片段详情:")
        for i, clip in enumerate(audio_track_info["audio_clips"]):
            print(f"   {i+1}. \"{clip['text']}\"")
            print(f"      时间轴: {clip['timeline_in']:.1f}s - {clip['timeline_out']:.1f}s")
            print(f"      音频URL: {clip['audio_url'][:80]}...")

        # 构建IMS AudioTracks
        audio_tracks = build_ims_audio_tracks(audio_track_info)

        print(f"\n📦 IMS AudioTracks格式:")
        print(json.dumps(audio_tracks, indent=2, ensure_ascii=False))

        # 验证时间轴是否正确
        print(f"\n✅ 验证结果:")
        for i, clip in enumerate(audio_track_info["audio_clips"]):
            subtitle_clip = subtitle_sequence["clips"][i]
            timeline_in = clip["timeline_in"]
            timeline_out = clip["timeline_out"]
            expected_in = subtitle_clip["start"]
            expected_out = subtitle_clip["end"]

            if timeline_in == expected_in and timeline_out == expected_out:
                print(f"   ✅ 片段{i+1} 时间轴匹配: {timeline_in}s - {timeline_out}s")
            else:
                print(f"   ❌ 片段{i+1} 时间轴不匹配:")
                print(f"      实际: {timeline_in}s - {timeline_out}s")
                print(f"      期望: {expected_in}s - {expected_out}s")
    else:
        print(f"❌ 分段TTS生成失败")

    # ========== 测试2: 合并模式（对比） ==========
    print(f"\n{'='*80}")
    print("🔄 测试2: 合并生成模式（旧逻辑，仅作对比）")
    print("="*80 + "\n")

    audio_track_info_merged = await generate_tts_audio_track(
        subtitle_sequence,
        voice="Cherry",
        speed=1.0,
        upload_to_oss=True,
        use_segmented=False  # 使用合并模式
    )

    if audio_track_info_merged:
        print(f"\n✅ 合并TTS生成成功:")
        print(f"   模式: {audio_track_info_merged['mode']}")
        print(f"   总时长: {audio_track_info_merged['duration']}秒")
        print(f"   音频URL: {audio_track_info_merged['audio_url'][:80]}...")

        audio_tracks_merged = build_ims_audio_tracks(audio_track_info_merged)

        print(f"\n📦 IMS AudioTracks格式:")
        print(json.dumps(audio_tracks_merged, indent=2, ensure_ascii=False))

        print(f"\n⚠️ 问题:")
        print(f"   合并模式会将所有字幕文本合成一段连续语音")
        print(f"   无法保证每句话的时间与字幕精确对齐")
    else:
        print(f"❌ 合并TTS生成失败")

    print(f"\n{'='*80}")
    print("✅ 测试完成")
    print("="*80 + "\n")


if __name__ == "__main__":
    asyncio.run(test_segmented_tts())
