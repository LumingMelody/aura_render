"""
IMS时间轴对齐工具

确保所有轨道（视频、音频、字幕、特效）的时间轴正确对齐
"""

from typing import List, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)


class TimelineAligner:
    """时间轴对齐器"""

    def __init__(self, tolerance: float = 0.1):
        """
        初始化时间轴对齐器

        Args:
            tolerance: 时间对齐容差（秒），小于此值视为对齐
        """
        self.tolerance = tolerance

    def align_timeline(
        self,
        timeline: Dict[str, Any],
        video_clips_info: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        对齐整个Timeline的所有轨道

        Args:
            timeline: IMS Timeline对象
            video_clips_info: 视频片段信息（用于计算片段边界）
                [{"url": "...", "duration": 5.0}, ...]

        Returns:
            对齐后的timeline
        """
        logger.info(f"⏱️ 开始时间轴对齐...")

        # 1. 计算视频片段的时间边界
        video_boundaries = self._calculate_video_boundaries(
            timeline.get("VideoTracks", []),
            video_clips_info
        )

        logger.info(f"   视频片段时间边界: {video_boundaries}")

        # 2. 对齐音频轨道
        audio_tracks = timeline.get("AudioTracks", [])
        if audio_tracks:
            timeline["AudioTracks"] = self._align_audio_tracks(
                audio_tracks,
                video_boundaries
            )

        # 3. 对齐字幕轨道
        subtitle_tracks = timeline.get("SubtitleTracks", [])
        if subtitle_tracks:
            timeline["SubtitleTracks"] = self._align_subtitle_tracks(
                subtitle_tracks,
                video_boundaries
            )

        # 4. 对齐特效轨道
        effect_tracks = timeline.get("EffectTracks", [])
        if effect_tracks:
            timeline["EffectTracks"] = self._align_effect_tracks(
                effect_tracks,
                video_boundaries
            )

        # 5. 对齐文字/花字轨道
        text_tracks = timeline.get("TextTracks", [])
        if text_tracks:
            timeline["TextTracks"] = self._align_text_tracks(
                text_tracks,
                video_boundaries
            )

        logger.info(f"✅ 时间轴对齐完成")
        return timeline

    def _calculate_video_boundaries(
        self,
        video_tracks: List[Dict[str, Any]],
        clips_info: List[Dict[str, Any]] = None
    ) -> List[Tuple[float, float]]:
        """
        计算视频片段的时间边界

        Returns:
            [(start, end), ...] 片段时间边界列表
        """
        boundaries = []

        if not video_tracks:
            # 如果没有视频轨道，使用clips_info计算
            if clips_info:
                current_time = 0.0
                for clip in clips_info:
                    duration = clip.get("duration", 5.0)
                    boundaries.append((current_time, current_time + duration))
                    current_time += duration
            return boundaries

        # ✅ 修复：从VideoTracks中提取边界，正确计算累积时间
        current_time = 0.0  # 累积时间
        for track in video_tracks:
            clips = track.get("VideoTrackClips", [])
            for idx, clip in enumerate(clips):
                timeline_in = clip.get("TimelineIn")
                timeline_out = clip.get("TimelineOut")

                # ✅ 如果TimelineIn未设置，使用累积时间
                if timeline_in is None:
                    timeline_in = current_time

                if timeline_out is None:
                    # 如果没有TimelineOut，尝试从clips_info获取duration
                    if clips_info and idx < len(clips_info):
                        duration = clips_info[idx].get("duration", 5.0)
                    else:
                        duration = 5.0  # 默认5秒
                    timeline_out = timeline_in + duration

                boundaries.append((float(timeline_in), float(timeline_out)))

                # ✅ 更新累积时间
                current_time = timeline_out

        return boundaries

    def _align_audio_tracks(
        self,
        audio_tracks: List[Dict[str, Any]],
        video_boundaries: List[Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        """对齐音频轨道到视频片段边界"""
        if not video_boundaries:
            return audio_tracks

        total_duration = video_boundaries[-1][1] if video_boundaries else 0.0

        for track in audio_tracks:
            clips = track.get("AudioTrackClips", [])
            valid_clips = []  # ✨ 新增：收集有效的clips

            for clip in clips:
                timeline_in = float(clip.get("TimelineIn", 0.0))
                timeline_out = float(clip.get("TimelineOut", 10.0))

                # 跳过超出视频时长的音频片段
                if timeline_in >= total_duration:
                    logger.info(
                        f"   ⚠️ 跳过超出视频时长的音频片段: "
                        f"[{timeline_in:.2f}s-{timeline_out:.2f}s]"
                    )
                    continue

                # 确保不超出视频总时长
                if timeline_out > total_duration:
                    original_out = timeline_out
                    timeline_out = total_duration

                    logger.info(
                        f"   ⚠️ 音频片段超出视频时长 "
                        f"[{timeline_in:.2f}s-{original_out:.2f}s] → "
                        f"[{timeline_in:.2f}s-{timeline_out:.2f}s]"
                    )

                    # 同时调整音频的Out点
                    in_point = clip.get("In", 0.0)
                    out_point = clip.get("Out", 10.0)
                    original_duration = out_point - in_point
                    new_duration = timeline_out - timeline_in
                    if new_duration < original_duration:
                        clip["Out"] = in_point + new_duration

                # ✨ 防御性检查：如果TimelineOut <= TimelineIn，跳过该片段
                if timeline_out <= timeline_in:
                    logger.warning(
                        f"   ⚠️ 跳过时间范围无效的音频片段: "
                        f"[{timeline_in:.2f}s-{timeline_out:.2f}s]"
                    )
                    continue

                # ✨ 关键修复：转换为整数（去掉小数点）
                clip["TimelineIn"] = int(round(timeline_in))
                clip["TimelineOut"] = int(round(timeline_out))

                valid_clips.append(clip)

            # ✨ 更新为有效的clips列表
            track["AudioTrackClips"] = valid_clips

        return audio_tracks

    def _align_subtitle_tracks(
        self,
        subtitle_tracks: List[Dict[str, Any]],
        video_boundaries: List[Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        """对齐字幕轨道到视频片段边界"""
        if not video_boundaries:
            return subtitle_tracks

        total_duration = video_boundaries[-1][1] if video_boundaries else 0.0

        for track in subtitle_tracks:
            clips = track.get("SubtitleTrackClips", [])
            aligned_clips = []

            for clip in clips:
                timeline_in = float(clip.get("TimelineIn", 0.0))
                timeline_out = float(clip.get("TimelineOut", timeline_in + 3.0))

                # 跳过超出视频时长的字幕
                if timeline_in >= total_duration:
                    logger.info(
                        f"   ⚠️ 跳过超出视频时长的字幕: "
                        f"[{timeline_in:.2f}s-{timeline_out:.2f}s]"
                    )
                    continue

                # 截断超出部分
                if timeline_out > total_duration:
                    timeline_out = total_duration

                # 检查字幕是否跨越视频片段边界
                clip_boundary = self._find_clip_boundary(timeline_in, video_boundaries)
                if clip_boundary and timeline_out > clip_boundary[1]:
                    logger.info(
                        f"   ℹ️ 字幕跨越片段边界，截断到片段结束: "
                        f"{timeline_out:.2f}s → {clip_boundary[1]:.2f}s"
                    )
                    timeline_out = clip_boundary[1]

                # ✨ 防御性检查：如果TimelineOut <= TimelineIn，跳过该片段
                if timeline_out <= timeline_in:
                    logger.warning(
                        f"   ⚠️ 跳过时间范围无效的字幕片段: "
                        f"[{timeline_in:.2f}s-{timeline_out:.2f}s]"
                    )
                    continue

                # ✨ 关键修复：转换为整数（去掉小数点）
                clip["TimelineIn"] = int(round(timeline_in))
                clip["TimelineOut"] = int(round(timeline_out))

                aligned_clips.append(clip)

            track["SubtitleTrackClips"] = aligned_clips

        return subtitle_tracks

    def _align_effect_tracks(
        self,
        effect_tracks: List[Dict[str, Any]],
        video_boundaries: List[Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        """对齐特效轨道到视频片段"""
        if not video_boundaries:
            return effect_tracks

        for track in effect_tracks:
            items = track.get("EffectTrackItems", [])
            aligned_items = []

            for item in items:
                timeline_in = float(item.get("TimelineIn", 0.0))
                timeline_out = float(item.get("TimelineOut", timeline_in + 3.0))

                # 查找对应的视频片段
                clip_boundary = self._find_clip_boundary(timeline_in, video_boundaries)
                if not clip_boundary:
                    logger.info(
                        f"   ⚠️ 特效不在任何视频片段内，跳过: "
                        f"[{timeline_in:.2f}s-{timeline_out:.2f}s]"
                    )
                    continue

                # 确保特效在片段范围内
                if timeline_in < clip_boundary[0]:
                    timeline_in = clip_boundary[0]
                if timeline_out > clip_boundary[1]:
                    timeline_out = clip_boundary[1]

                # ✨ 防御性检查：如果TimelineOut <= TimelineIn，跳过该片段
                if timeline_out <= timeline_in:
                    logger.warning(
                        f"   ⚠️ 跳过时间范围无效的特效片段: "
                        f"[{timeline_in:.2f}s-{timeline_out:.2f}s]"
                    )
                    continue

                # ✨ 关键修复：转换为整数（去掉小数点）
                item["TimelineIn"] = int(round(timeline_in))
                item["TimelineOut"] = int(round(timeline_out))

                aligned_items.append(item)

            track["EffectTrackItems"] = aligned_items

        return effect_tracks

    def _align_text_tracks(
        self,
        text_tracks: List[Dict[str, Any]],
        video_boundaries: List[Tuple[float, float]]
    ) -> List[Dict[str, Any]]:
        """对齐文字/花字轨道"""
        # 文字轨道的对齐逻辑与字幕类似
        if not video_boundaries:
            return text_tracks

        total_duration = video_boundaries[-1][1] if video_boundaries else 0.0

        for track in text_tracks:
            clips = track.get("SubtitleClips", [])
            aligned_clips = []

            for clip in clips:
                timeline_in = float(clip.get("TimelineIn", 0.0))
                timeline_out = float(clip.get("TimelineOut", timeline_in + 3.0))

                # 跳过超出视频时长的文字
                if timeline_in >= total_duration:
                    logger.info(
                        f"   ⚠️ 跳过超出视频时长的文字片段: "
                        f"[{timeline_in:.2f}s-{timeline_out:.2f}s]"
                    )
                    continue

                # 截断超出部分
                if timeline_out > total_duration:
                    timeline_out = total_duration

                # ✨ 防御性检查：如果TimelineOut <= TimelineIn，跳过该片段
                if timeline_out <= timeline_in:
                    logger.warning(
                        f"   ⚠️ 跳过时间范围无效的文字片段: "
                        f"[{timeline_in:.2f}s-{timeline_out:.2f}s]"
                    )
                    continue

                # ✨ 关键修复：转换为整数（去掉小数点）
                clip["TimelineIn"] = int(round(timeline_in))
                clip["TimelineOut"] = int(round(timeline_out))

                aligned_clips.append(clip)

            track["SubtitleClips"] = aligned_clips

        return text_tracks

    def _find_clip_boundary(
        self,
        time: float,
        boundaries: List[Tuple[float, float]]
    ) -> Tuple[float, float]:
        """
        查找时间点所在的视频片段边界

        ✅ 修复：边界点优先归属到后一个片段
        例如：time=5.0 时，在 [(0,5), (5,10)] 中应归属到 (5,10)
        """
        for start, end in boundaries:
            # ✅ 修复：使用严格的区间判断，边界点归属到开始位置
            # 对于 time=5.0: 不匹配 [0,5)，匹配 [5,10)
            if start <= time < end:
                return (start, end)

        # ✅ 兜底：如果time正好等于最后一个片段的end，归属到最后一个片段
        if boundaries and abs(time - boundaries[-1][1]) <= self.tolerance:
            return boundaries[-1]

        return None


def align_ims_timeline(
    timeline: Dict[str, Any],
    video_clips_info: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    便捷函数：对齐IMS Timeline

    Args:
        timeline: IMS Timeline对象
        video_clips_info: 视频片段信息

    Returns:
        对齐后的timeline
    """
    aligner = TimelineAligner(tolerance=0.1)
    return aligner.align_timeline(timeline, video_clips_info)
