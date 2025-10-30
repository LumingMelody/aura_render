"""
音频混音优化工具

实现智能的多音轨Gain调整策略，避免音频爆音和混响问题
"""

from typing import List, Dict, Any
import math
import logging

logger = logging.getLogger(__name__)


class AudioMixerOptimizer:
    """音频混音优化器"""

    # 音轨类型的默认音量参考值 (dB)
    DEFAULT_VOLUMES = {
        "tts": -12.0,      # 人声：最响（核心内容）
        "narration": -12.0,  # 旁白：与TTS同级
        "sfx": -16.0,      # 音效：略低
        "bgm": -12.0,      # BGM：调高到-12dB，让BGM更明显（原来是-20dB太小了）
        "ambient": -22.0   # 环境音：更低
    }

    # 多音轨同时播放时的衰减系数
    CONCURRENT_ATTENUATION = {
        2: 0.85,  # 2个音轨同时播放：衰减到85%
        3: 0.70,  # 3个音轨同时播放：衰减到70%
        4: 0.60,  # 4个或更多音轨：衰减到60%
    }

    def __init__(self, target_peak_db: float = -3.0, target_rms_db: float = -16.0):
        """
        初始化音频混音优化器

        Args:
            target_peak_db: 目标峰值电平（防止削波）
            target_rms_db: 目标RMS响度（平均响度）
        """
        self.target_peak_db = target_peak_db
        self.target_rms_db = target_rms_db

    def optimize_audio_tracks(
        self,
        audio_tracks: List[Dict[str, Any]],
        total_duration: float
    ) -> List[Dict[str, Any]]:
        """
        优化多个音频轨道的音量配置

        Args:
            audio_tracks: IMS AudioTracks数组
                [
                    {
                        "AudioTrackClips": [
                            {
                                "MediaURL": "...",
                                "TimelineIn": 0.0,
                                "TimelineOut": 10.0,
                                "Effects": [{"Type": "Volume", "Gain": 1.0}]
                            }
                        ]
                    }
                ]
            total_duration: 视频总时长（用于计算并发度）

        Returns:
            优化后的audio_tracks
        """
        if not audio_tracks:
            return audio_tracks

        logger.info(f"🎚️ 开始优化 {len(audio_tracks)} 个音频轨道...")

        # 1. 分析每个时间段的并发音轨数
        concurrent_map = self._analyze_concurrency(audio_tracks, total_duration)

        # 2. 识别音轨类型并分配基础音量
        for i, track in enumerate(audio_tracks):
            track_type = self._identify_track_type(track, i)
            base_db = self.DEFAULT_VOLUMES.get(track_type, -18.0)

            logger.info(f"   音轨 {i+1}: 类型={track_type}, 基础音量={base_db}dB")

            # 3. 为每个clip调整Gain
            clips = track.get("AudioTrackClips", [])
            for clip in clips:
                timeline_in = clip.get("TimelineIn", 0.0)
                timeline_out = clip.get("TimelineOut", 10.0)

                # 计算该时间段的平均并发度
                avg_concurrent = self._get_average_concurrency(
                    concurrent_map,
                    timeline_in,
                    timeline_out
                )

                # 应用并发衰减
                attenuation = self._get_attenuation_factor(avg_concurrent)
                adjusted_db = base_db + self._linear_to_db(attenuation)

                # 转换为IMS Gain值
                gain = self._db_to_gain(adjusted_db)

                # 更新Gain（保留已有的更高音量设置）
                if "Effects" not in clip:
                    clip["Effects"] = []

                volume_effect = next(
                    (e for e in clip["Effects"] if e.get("Type") == "Volume"),
                    None
                )

                if volume_effect:
                    # 如果已有Gain值且大于计算值，保留用户设置（尊重手动调整）
                    existing_gain = volume_effect.get("Gain", 0.0)
                    if existing_gain > gain:
                        logger.info(
                            f"      Clip [{timeline_in:.1f}s-{timeline_out:.1f}s]: "
                            f"保留已有音量 Gain={existing_gain:.3f} (大于计算值{gain:.3f})"
                        )
                        gain = existing_gain  # 保留原值
                    volume_effect["Gain"] = gain
                else:
                    clip["Effects"].append({
                        "Type": "Volume",
                        "Gain": gain
                    })

                logger.info(
                    f"      Clip [{timeline_in:.1f}s-{timeline_out:.1f}s]: "
                    f"并发={avg_concurrent:.1f}, 衰减={attenuation:.2f}, "
                    f"最终Gain={gain:.3f}"
                )

        logger.info(f"✅ 音频混音优化完成")
        return audio_tracks

    def _identify_track_type(self, track: Dict[str, Any], index: int) -> str:
        """
        识别音轨类型

        根据URL、文件名、轨道索引等推断音轨类型
        """
        clips = track.get("AudioTrackClips", [])
        if not clips:
            return "bgm"

        # 检查第一个clip的URL
        url = clips[0].get("MediaURL", "").lower()

        if "tts" in url or "voice" in url or "speech" in url:
            return "tts"
        elif "bgm" in url or "music" in url or "background" in url:
            return "bgm"
        elif "sfx" in url or "sound" in url or "effect" in url:
            return "sfx"
        elif "ambient" in url or "atmosphere" in url:
            return "ambient"

        # 根据轨道顺序推断（第一轨通常是BGM）
        if index == 0:
            return "bgm"
        elif index == 1:
            return "sfx"
        else:
            return "ambient"

    def _analyze_concurrency(
        self,
        audio_tracks: List[Dict[str, Any]],
        total_duration: float,
        resolution: float = 0.5
    ) -> Dict[float, int]:
        """
        分析每个时间点的并发音轨数

        Args:
            audio_tracks: 音频轨道数组
            total_duration: 总时长
            resolution: 时间分辨率（秒）

        Returns:
            {时间点: 并发数} 的映射
        """
        concurrent_map = {}
        num_samples = int(total_duration / resolution) + 1

        for t_idx in range(num_samples):
            time = t_idx * resolution
            concurrent_count = 0

            for track in audio_tracks:
                clips = track.get("AudioTrackClips", [])
                for clip in clips:
                    timeline_in = clip.get("TimelineIn", 0.0)
                    timeline_out = clip.get("TimelineOut", 10.0)

                    if timeline_in <= time < timeline_out:
                        concurrent_count += 1
                        break  # 每个轨道只计数一次

            concurrent_map[time] = concurrent_count

        return concurrent_map

    def _get_average_concurrency(
        self,
        concurrent_map: Dict[float, int],
        start_time: float,
        end_time: float
    ) -> float:
        """计算某个时间段的平均并发度"""
        relevant_times = [
            (t, count) for t, count in concurrent_map.items()
            if start_time <= t < end_time
        ]

        if not relevant_times:
            return 1.0

        total_count = sum(count for _, count in relevant_times)
        return total_count / len(relevant_times)

    def _get_attenuation_factor(self, concurrent_count: float) -> float:
        """根据并发数获取衰减系数"""
        if concurrent_count < 2:
            return 1.0

        concurrent_int = int(concurrent_count)
        if concurrent_int in self.CONCURRENT_ATTENUATION:
            return self.CONCURRENT_ATTENUATION[concurrent_int]
        elif concurrent_int >= 4:
            return self.CONCURRENT_ATTENUATION[4]
        else:
            # 线性插值
            lower = self.CONCURRENT_ATTENUATION[concurrent_int]
            upper = self.CONCURRENT_ATTENUATION[concurrent_int + 1]
            fraction = concurrent_count - concurrent_int
            return lower + (upper - lower) * fraction

    def _db_to_gain(self, db: float) -> float:
        """将dB转换为IMS Gain值（0-2）"""
        if db <= -60:
            return 0.0
        gain = math.pow(10, db / 20.0)
        return max(0.0, min(2.0, gain))

    def _linear_to_db(self, linear: float) -> float:
        """线性值转dB"""
        if linear <= 0:
            return -60.0
        return 20.0 * math.log10(linear)


def optimize_ims_audio_tracks(
    timeline: Dict[str, Any],
    total_duration: float = None
) -> Dict[str, Any]:
    """
    便捷函数：优化IMS Timeline中的音频轨道

    Args:
        timeline: IMS Timeline对象
        total_duration: 总时长（如果不提供，会自动计算）

    Returns:
        优化后的timeline
    """
    audio_tracks = timeline.get("AudioTracks", [])
    if not audio_tracks:
        return timeline

    # 自动计算总时长
    if total_duration is None:
        max_time = 0.0
        for track in audio_tracks:
            clips = track.get("AudioTrackClips", [])
            for clip in clips:
                max_time = max(max_time, clip.get("TimelineOut", 0.0))
        total_duration = max_time

    # 优化音频轨道
    optimizer = AudioMixerOptimizer()
    timeline["AudioTracks"] = optimizer.optimize_audio_tracks(
        audio_tracks,
        total_duration
    )

    return timeline
