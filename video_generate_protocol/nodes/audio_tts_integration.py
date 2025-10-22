"""
音频TTS集成 - 将字幕转换为语音并添加到视频

这个模块提供了将subtitle_sequence转换为TTS音频的功能，
可以直接集成到IMS timeline的AudioTracks中
"""

import asyncio
import logging
from typing import Dict, List, Optional

# 使用项目日志系统
try:
    from utils.logger import get_logger, LogCategory
    logger = get_logger("audio_tts_integration").with_context(category=LogCategory.SYSTEM)
except ImportError:
    logger = logging.getLogger(__name__)


async def generate_tts_audio_track(
    subtitle_sequence: Dict,
    voice: str = "Cherry",  # ✅ 使用阿里云Qwen3-TTS支持的音色（芊悦-女声）
    speed: float = 1.0,
    upload_to_oss: bool = True,
    use_segmented: bool = True  # ✨ 新增：是否使用分段生成（推荐True，实现音画同步）
) -> Optional[Dict]:
    """
    根据字幕序列生成TTS音频轨道

    Args:
        subtitle_sequence: subtitle_node生成的字幕序列
        voice: 千问TTS音色
        speed: 语速
        upload_to_oss: 是否上传到OSS（推荐True，获取永久URL）
        use_segmented: 是否使用分段生成（True=每个字幕独立生成音频，实现精确同步）

    Returns:
        音频轨道信息，格式：
        {
            "audio_clips": [  # 音频片段列表（分段模式）
                {
                    "audio_url": "https://...",
                    "timeline_in": 0.0,
                    "timeline_out": 3.0,
                    "text": "欢迎来到"
                },
                ...
            ],
            "total_duration": 10.5,
            "mode": "segmented" | "merged"
        }
        或（旧格式，向后兼容）：
        {
            "audio_url": "https://...",  # 合并模式
            "duration": 10.5
        }

    Example:
        >>> subtitle_seq = {
        ...     "clips": [
        ...         {"start": 0.0, "end": 3.0, "text": "欢迎来到"},
        ...         {"start": 3.0, "end": 6.0, "text": "机器学习的世界"}
        ...     ]
        ... }
        >>> audio_track = await generate_tts_audio_track(subtitle_seq, use_segmented=True)
        >>> print(f"生成了 {len(audio_track['audio_clips'])} 个音频片段")
    """
    try:
        from core.cliptemplate.qwen import get_qwen_tts_generator

        if not subtitle_sequence or "clips" not in subtitle_sequence:
            logger.warning("⚠️ 字幕序列为空，跳过TTS生成")
            return None

        clips = subtitle_sequence.get("clips", [])
        if not clips:
            logger.warning("⚠️ 字幕序列无有效片段")
            return None

        logger.info(f"🎤 开始生成TTS音频，共 {len(clips)} 个字幕片段")
        logger.info(f"   模式: {'分段生成（精确同步）' if use_segmented else '合并生成（旧模式）'}")

        tts_generator = get_qwen_tts_generator()

        # ========== 方案1: 分段生成（推荐） ==========
        if use_segmented:
            return await _generate_segmented_audio(
                clips,
                tts_generator,
                voice,
                speed,
                upload_to_oss
            )

        # ========== 方案2: 合并生成（旧逻辑，保持向后兼容） ==========
        else:
            return await _generate_merged_audio(
                clips,
                tts_generator,
                voice,
                speed,
                upload_to_oss
            )

    except Exception as e:
        logger.error(f"❌ TTS音频生成异常: {e}")
        import traceback
        traceback.print_exc()
        return None


async def _generate_segmented_audio(
    clips: List[Dict],
    tts_generator,
    voice: str,
    speed: float,
    upload_to_oss: bool
) -> Dict:
    """
    分段生成TTS音频（每个字幕独立生成）

    优点：音画完美同步，支持字幕间停顿
    """
    logger.info(f"📊 分段生成模式：将为每个字幕片段生成独立音频")

    audio_clips = []
    successful_count = 0
    failed_count = 0

    # 并发控制：同时最多3个请求，避免API限流
    semaphore = asyncio.Semaphore(3)

    async def generate_single_clip(clip_index: int, clip: Dict):
        """生成单个字幕的TTS音频"""
        nonlocal successful_count, failed_count

        async with semaphore:
            text = clip.get("text", "").strip()
            if not text:
                logger.warning(f"   ⚠️ 片段 {clip_index + 1} 文本为空，跳过")
                return None

            start_time = clip.get("start", 0.0)
            end_time = clip.get("end", start_time + clip.get("duration", 0.0))

            try:
                logger.info(f"   🎵 生成片段 {clip_index + 1}/{len(clips)}: \"{text[:20]}...\" ({start_time:.1f}s - {end_time:.1f}s)")

                # 调用千问TTS
                audio_url = await tts_generator.generate_speech(
                    text=text,
                    voice=voice,
                    speed=speed,
                    upload_to_oss=upload_to_oss
                )

                if audio_url:
                    successful_count += 1
                    logger.info(f"      ✅ 片段 {clip_index + 1} 生成成功")
                    return {
                        "audio_url": audio_url,
                        "timeline_in": start_time,
                        "timeline_out": end_time,
                        "text": text,
                        "duration": end_time - start_time
                    }
                else:
                    failed_count += 1
                    logger.warning(f"      ❌ 片段 {clip_index + 1} 生成失败（API返回空）")
                    return None

            except Exception as e:
                failed_count += 1
                logger.error(f"      ❌ 片段 {clip_index + 1} 生成异常: {e}")
                return None

    # 并发生成所有音频片段
    tasks = [generate_single_clip(i, clip) for i, clip in enumerate(clips)]
    results = await asyncio.gather(*tasks)

    # 过滤成功的结果
    audio_clips = [r for r in results if r is not None]

    # 计算总时长
    total_duration = 0.0
    if clips:
        last_clip = clips[-1]
        total_duration = last_clip.get("end", last_clip.get("start", 0.0) + last_clip.get("duration", 0.0))

    logger.info(f"✅ 分段生成完成:")
    logger.info(f"   成功: {successful_count}/{len(clips)} 个片段")
    logger.info(f"   失败: {failed_count}/{len(clips)} 个片段")
    logger.info(f"   总时长: {total_duration:.1f}秒")

    if not audio_clips:
        logger.error("❌ 所有音频片段生成失败")
        return None

    return {
        "audio_clips": audio_clips,
        "total_duration": total_duration,
        "mode": "segmented",
        "voice": voice,
        "speed": speed,
        "source": "qwen_tts"
    }


async def _generate_merged_audio(
    clips: List[Dict],
    tts_generator,
    voice: str,
    speed: float,
    upload_to_oss: bool
) -> Dict:
    """
    合并生成TTS音频（旧逻辑，保持向后兼容）

    缺点：音画可能不同步
    """
    # 合并所有字幕文本（简单方案：一次性合成整段语音）
    full_text = " ".join([clip.get("text", "") for clip in clips if clip.get("text")])

    if not full_text.strip():
        logger.warning("⚠️ 字幕文本为空")
        return None

    logger.info(f"📝 合并字幕文本: {full_text[:100]}...")
    logger.info(f"   总长度: {len(full_text)} 字符")

    # 计算视频总时长（从字幕序列）
    total_duration = 0.0
    if clips:
        last_clip = clips[-1]
        total_duration = last_clip.get("end", last_clip.get("start", 0.0) + last_clip.get("duration", 0.0))

    # ✨ 智能调整语速：根据文本长度和视频时长自动计算合适的语速
    # 正常语速下，中文约为 4-5 字符/秒
    if speed == 1.0 and total_duration > 0:  # 只在默认语速时自动调整
        normal_chars_per_second = 4.5  # 正常语速：每秒约4.5个汉字
        estimated_duration = len(full_text) / normal_chars_per_second

        # 如果预估时长与视频时长差距较大，调整语速
        if estimated_duration > 0:
            speed_ratio = estimated_duration / total_duration

            # 语速范围限制在 0.5-2.0 之间（阿里云TTS限制）
            # 如果音频太短，放慢语速（最慢0.6倍，避免太慢）
            # 如果音频太长，加快语速（最快1.5倍，避免太快听不清）
            if speed_ratio < 1.0:  # 音频会太短，需要放慢
                speed = max(0.6, speed_ratio)
            elif speed_ratio > 1.0:  # 音频会太长，需要加快
                speed = min(1.5, speed_ratio)

            logger.info(f"🎯 智能语速调整:")
            logger.info(f"   视频时长: {total_duration:.1f}秒")
            logger.info(f"   文本长度: {len(full_text)}字符")
            logger.info(f"   预估时长: {estimated_duration:.1f}秒 (正常语速)")
            logger.info(f"   调整语速: {speed:.2f}x (原始: {speed_ratio:.2f}x)")

    # 调用千问TTS生成语音
    audio_url = await tts_generator.generate_speech(
        text=full_text,
        voice=voice,
        speed=speed,
        upload_to_oss=upload_to_oss
    )

    if not audio_url:
        logger.error("❌ TTS音频生成失败")
        return None

    logger.info(f"✅ TTS音频生成成功")
    logger.info(f"   音频URL: {audio_url}")
    logger.info(f"   视频时长: {total_duration}秒")
    logger.info(f"   实际语速: {speed:.2f}x")

    return {
        "audio_url": audio_url,
        "duration": total_duration,
        "mode": "merged",
        "voice": voice,
        "speed": speed,
        "source": "qwen_tts"
    }


def build_ims_audio_tracks(audio_track_info: Dict) -> List[Dict]:
    """
    构建IMS AudioTracks格式（支持分段和合并两种模式）

    Args:
        audio_track_info: generate_tts_audio_track返回的音频轨道信息
            分段模式: {"audio_clips": [...], "mode": "segmented"}
            合并模式: {"audio_url": "...", "mode": "merged"}

    Returns:
        IMS AudioTracks数组

    Example:
        >>> # 分段模式（推荐）
        >>> audio_info = {
        ...     "audio_clips": [
        ...         {"audio_url": "https://...", "timeline_in": 0.0, "timeline_out": 3.0},
        ...         {"audio_url": "https://...", "timeline_in": 3.0, "timeline_out": 6.0}
        ...     ],
        ...     "mode": "segmented"
        ... }
        >>> audio_tracks = build_ims_audio_tracks(audio_info)
        >>>
        >>> # 合并模式（旧格式）
        >>> audio_info = {"audio_url": "https://...", "duration": 10.5, "mode": "merged"}
        >>> audio_tracks = build_ims_audio_tracks(audio_info)
    """
    if not audio_track_info:
        return []

    mode = audio_track_info.get("mode", "merged")  # 默认为合并模式（向后兼容）

    # ========== 分段模式：每个字幕对应独立的音频片段 ==========
    if mode == "segmented" and "audio_clips" in audio_track_info:
        audio_clips = audio_track_info["audio_clips"]

        if not audio_clips:
            logger.warning("⚠️ 分段模式下audio_clips为空")
            return []

        # 构建IMS AudioTrackClips
        ims_clips = []
        for clip in audio_clips:
            ims_clip = {
                "MediaURL": clip["audio_url"],
                "TimelineIn": round(clip["timeline_in"], 2),   # ✨ 关键：设置音频入点
                "TimelineOut": round(clip["timeline_out"], 2)  # ✨ 关键：设置音频出点
            }
            ims_clips.append(ims_clip)

        audio_tracks = [
            {
                "AudioTrackClips": ims_clips
            }
        ]

        logger.info(f"📊 已构建IMS AudioTracks (分段模式):")
        logger.info(f"   片段数量: {len(ims_clips)}")
        logger.info(f"   时间范围: {ims_clips[0]['TimelineIn']:.1f}s - {ims_clips[-1]['TimelineOut']:.1f}s")
        return audio_tracks

    # ========== 合并模式：单个完整音频（旧逻辑，向后兼容） ==========
    elif "audio_url" in audio_track_info:
        audio_url = audio_track_info["audio_url"]

        audio_tracks = [
            {
                "AudioTrackClips": [
                    {
                        "MediaURL": audio_url
                        # 不设置TimelineIn/TimelineOut，系统会自动对齐到视频开头
                    }
                ]
            }
        ]

        logger.info(f"📊 已构建IMS AudioTracks (合并模式): {audio_url[:80]}...")
        return audio_tracks

    else:
        logger.warning("⚠️ 音频轨道信息格式不正确，缺少audio_clips或audio_url")
        return []


async def integrate_tts_to_timeline(
    timeline: Dict,
    subtitle_sequence: Dict,
    voice: str = "Cherry",  # ✅ 使用阿里云Qwen3-TTS支持的音色（芊悦-女声）
    speed: float = 1.0,
    upload_to_oss: bool = True,
    use_segmented: bool = True  # ✨ 新增：是否使用分段生成（推荐True，实现音画同步）
) -> Dict:
    """
    便捷函数：直接将TTS音频集成到IMS timeline

    Args:
        timeline: IMS timeline字典（会被修改）
        subtitle_sequence: 字幕序列
        voice: TTS音色
        speed: 语速
        upload_to_oss: 是否上传到OSS
        use_segmented: 是否使用分段生成（True=每个字幕独立生成音频，实现精确同步）

    Returns:
        修改后的timeline（原地修改+返回）

    Example:
        >>> timeline = {
        ...     "VideoTracks": [...],
        ...     "SubtitleTracks": [...]
        ... }
        >>> subtitle_seq = {...}
        >>> # 推荐：使用分段模式实现音画同步
        >>> timeline = await integrate_tts_to_timeline(timeline, subtitle_seq, use_segmented=True)
        >>> # 或者：使用合并模式（旧逻辑）
        >>> timeline = await integrate_tts_to_timeline(timeline, subtitle_seq, use_segmented=False)
    """
    logger.info("🎵 开始集成TTS音频到timeline...")

    # 生成TTS音频
    audio_track_info = await generate_tts_audio_track(
        subtitle_sequence,
        voice=voice,
        speed=speed,
        upload_to_oss=upload_to_oss,
        use_segmented=use_segmented
    )

    if not audio_track_info:
        logger.warning("⚠️ TTS音频生成失败，跳过AudioTracks")
        return timeline

    # 构建AudioTracks
    audio_tracks = build_ims_audio_tracks(audio_track_info)

    if audio_tracks:
        timeline["AudioTracks"] = audio_tracks
        logger.info("✅ AudioTracks已添加到timeline")
    else:
        logger.warning("⚠️ 未能构建AudioTracks")

    return timeline


# ========== 使用示例 ==========

async def example_usage():
    """使用示例"""

    # 假设从subtitle_node获取的字幕序列
    subtitle_sequence = {
        "clips": [
            {"start": 0.0, "end": 3.0, "text": "欢迎来到"},
            {"start": 3.0, "end": 6.0, "text": "机器学习的世界"},
            {"start": 6.0, "end": 10.0, "text": "让我们开始探索吧"}
        ]
    }

    # 方案1: 单独生成TTS音频
    audio_track_info = await generate_tts_audio_track(subtitle_sequence)
    if audio_track_info:
        print(f"音频URL: {audio_track_info['audio_url']}")

    # 方案2: 直接集成到timeline
    timeline = {
        "VideoTracks": [{
            "VideoTrackClips": [
                {"MediaURL": "https://example.com/video1.mp4"}
            ]
        }],
        "SubtitleTracks": [...]
    }

    timeline = await integrate_tts_to_timeline(timeline, subtitle_sequence)
    print(f"Timeline: {timeline}")


if __name__ == "__main__":
    asyncio.run(example_usage())
