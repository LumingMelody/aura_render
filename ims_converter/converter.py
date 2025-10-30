"""
IMS主转换器

将完整的VGP输出转换为阿里云IMS Timeline格式
"""

from typing import Dict, List, Any, Optional
import logging
from .utils import (
    TransitionConverter,
    FilterConverter,
    EffectConverter,
    FlowerTextConverter,
    OverlayConverter
)

logger = logging.getLogger(__name__)


class IMSConverter:
    """VGP到IMS的主转换器"""

    def __init__(self, use_filter_preset: bool = True):
        """
        初始化转换器

        Args:
            use_filter_preset: 是否使用滤镜预设(True)或精确参数(False)
        """
        self.use_filter_preset = use_filter_preset

    def convert(self, vgp_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        将完整的VGP输出转换为IMS Timeline

        Args:
            vgp_result: VGP完整输出，包含所有节点的结果
                {
                    "transition_sequence_id": [...],
                    "filter_sequence_id": [...],
                    "effects_sequence_id": [...],
                    "text_overlay_track_id": {...},
                    "auxiliary_track_id": {...},
                    "bgm_composition_id": {...}  # 新增：BGM轨道
                }

        Returns:
            IMS Timeline对象
                {
                    "VideoTracks": [...],
                    "AudioTracks": [...],  # 新增
                    "EffectTracks": [...],
                    "TextTracks": [...]
                }
        """
        timeline = {
            "VideoTracks": [],
            "AudioTracks": [],  # ✅ 新增：音频轨道
            "EffectTracks": [],
            "SubtitleTracks": []  # ✅ 修正：使用SubtitleTracks而不是TextTracks（IMS标准）
        }

        # 1. 转换主视频轨道 + 转场
        logger.info("开始转换视频轨道和转场...")
        video_clips = self._convert_video_clips(vgp_result)
        if video_clips:
            timeline["VideoTracks"].append({
                "VideoTrackClips": video_clips
            })

        # 2. 转换音频轨道 (BGM + TTS + SFX)
        logger.info("开始转换音频轨道...")
        logger.info(f"   🎵 检查BGM数据: bgm_composition_id存在={('bgm_composition_id' in vgp_result)}")
        if "bgm_composition_id" in vgp_result:
            bgm = vgp_result["bgm_composition_id"]
            logger.info(f"   🎵 BGM类型: {type(bgm).__name__}")
            if isinstance(bgm, dict):
                logger.info(f"   🎵 BGM keys: {list(bgm.keys())}")
                logger.info(f"   🎵 BGM clips数量: {len(bgm.get('clips', []))}")

        audio_tracks = self._convert_audio_tracks(vgp_result)
        if audio_tracks:
            timeline["AudioTracks"] = audio_tracks
            logger.info(f"   ✅ 成功添加 {len(audio_tracks)} 个音频轨道")

        # 3. 转换滤镜轨道
        logger.info("开始转换滤镜...")
        filter_track = self._convert_filters(vgp_result)
        if filter_track:
            timeline["EffectTracks"].append(filter_track)

        # 4. 转换特效轨道
        logger.info("开始转换特效...")
        effect_track = self._convert_effects(vgp_result)
        if effect_track:
            timeline["EffectTracks"].append(effect_track)

        # 5. 转换文字轨道(花字) - 添加到SubtitleTracks
        logger.info("开始转换文字/花字...")
        text_track = self._convert_text_overlay(vgp_result)
        if text_track:
            timeline["SubtitleTracks"].append(text_track)  # ✅ 添加到SubtitleTracks
            logger.info(f"   ✅ 已添加花字轨道")

        # 6. 转换辅助媒体 (作为额外的视频轨道)
        logger.info("开始转换辅助媒体...")
        aux_track = self._convert_auxiliary_media(vgp_result)
        if aux_track:
            timeline["VideoTracks"].append(aux_track)

        logger.info("IMS Timeline转换完成")
        return timeline

    def _convert_audio_tracks(self, vgp_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        转换音频轨道 (BGM + TTS + SFX)

        从bgm_composition_id、tts_track、sfx_track等提取
        """
        audio_tracks = []

        # 1. 处理BGM轨道
        bgm_data = vgp_result.get("bgm_composition_id")
        logger.debug(f"🎵 BGM数据检查: type={type(bgm_data)}, has_clips={isinstance(bgm_data, dict) and 'clips' in bgm_data}")
        if isinstance(bgm_data, dict):
            logger.debug(f"   BGM数据keys: {list(bgm_data.keys())}")
            clips_count = len(bgm_data.get("clips", []))
            logger.debug(f"   BGM clips数量: {clips_count}")

        if bgm_data and isinstance(bgm_data, dict):
            bgm_clips = bgm_data.get("clips", [])
            if bgm_clips:
                audio_track_clips = []
                for clip in bgm_clips:
                    audio_info = clip.get("audio", {})
                    audio_url = audio_info.get("url", "")

                    # ✅ 过滤无效的占位符URL（模拟数据）
                    if not audio_url or audio_url.startswith("https://audio.com/") or audio_url.startswith("https://assets.example.com/"):
                        logger.warning(f"⚠️ 跳过无效的BGM URL: {audio_url}")
                        continue

                    # 智能音量调整：BGM默认音量适中（背景音乐不应太小，但也不能盖过人声）
                    # VGP的volume_db通常是-18到-20dB，转换后太小（0.1-0.12）
                    # 修正策略：使用固定的合理音量（0.3-0.4，即30%-40%音量）
                    volume_db = clip.get("volume_db", -18.0)

                    # 根据VGP的音量建议调整，但确保在合理范围内
                    if volume_db >= -10:  # 较响
                        gain = 0.5  # 50%音量
                    elif volume_db >= -15:  # 中等
                        gain = 0.4  # 40%音量
                    elif volume_db >= -20:  # 较轻
                        gain = 0.3  # 30%音量
                    else:  # 很轻
                        gain = 0.25  # 25%音量

                    logger.debug(f"   BGM音量调整: {volume_db}dB → Gain {gain}")

                    # 计算时间范围
                    timeline_in = float(clip.get("start", 0.0))
                    timeline_out = float(clip.get("end", clip.get("start", 0.0) + clip.get("duration", 0.0)))

                    # 防御性检查：确保timeline_out > timeline_in
                    if timeline_out <= timeline_in:
                        logger.warning(f"⚠️ BGM片段时间范围无效 [{timeline_in} - {timeline_out}]，跳过")
                        continue

                    ims_clip = {
                        "MediaURL": audio_info.get("url", ""),
                        "TimelineIn": int(round(timeline_in)),
                        "TimelineOut": int(round(timeline_out)),
                        "In": audio_info.get("in_point", 0.0),
                        "Out": audio_info.get("out_point", audio_info.get("duration", 10.0)),
                        "Effects": [
                            {
                                "Type": "Volume",
                                "Gain": gain  # 转换为IMS的Gain值 (0-2倍, 1为原始音量)
                            }
                        ]
                    }

                    # 添加淡入淡出效果
                    transition = clip.get("transition", "")
                    if "淡入" in transition or "fade" in transition.lower():
                        ims_clip["Effects"].append({
                            "Type": "AFade",
                            "StartTime": clip.get("start", 0.0),
                            "Duration": 2.0,
                            "FadeType": "In"
                        })

                    audio_track_clips.append(ims_clip)

                if audio_track_clips:
                    audio_tracks.append({
                        "AudioTrackClips": audio_track_clips
                    })
                    logger.info(f"   ✅ 添加BGM轨道，包含 {len(audio_track_clips)} 个有效音频片段")
                else:
                    logger.warning(f"   ⚠️ BGM轨道中没有有效的音频URL，跳过BGM轨道")

        # 2. 处理辅助音频（从auxiliary_track_id中提取音频类型的素材）
        aux_track = vgp_result.get("auxiliary_track_id")
        if aux_track and isinstance(aux_track, dict):
            clips = aux_track.get("clips", [])
            audio_clips = [c for c in clips if c.get("type") == "audio"]

            if audio_clips:
                sfx_track_clips = []
                for clip in audio_clips:
                    # 音效音量略高于BGM，但低于人声
                    gain = 0.5  # 50%音量

                    # 计算时间范围
                    timeline_in = float(clip.get("start", 0.0))
                    timeline_out = timeline_in + float(clip.get("duration", 3.0))

                    # 防御性检查：确保timeline_out > timeline_in
                    if timeline_out <= timeline_in:
                        logger.warning(f"⚠️ SFX片段时间范围无效 [{timeline_in} - {timeline_out}]，跳过")
                        continue

                    ims_clip = {
                        "MediaURL": clip.get("file_path", ""),
                        "TimelineIn": int(round(timeline_in)),
                        "TimelineOut": int(round(timeline_out)),
                        "Effects": [
                            {
                                "Type": "Volume",
                                "Gain": gain
                            }
                        ]
                    }
                    sfx_track_clips.append(ims_clip)

                if sfx_track_clips:
                    audio_tracks.append({
                        "AudioTrackClips": sfx_track_clips
                    })

        logger.info(f"转换了 {len(audio_tracks)} 个音频轨道")
        return audio_tracks

    def _db_to_gain(self, db: float) -> float:
        """
        将dB音量转换为IMS的Gain值

        Args:
            db: 分贝值 (-∞ to 0)

        Returns:
            Gain值 (0-2倍，1为原始音量)
        """
        # IMS Gain: 0 = 静音, 1 = 原始音量, 2 = 200%音量
        # dB转线性: gain = 10^(db/20)
        import math
        if db <= -60:
            return 0.0
        gain = math.pow(10, db / 20.0)
        # 限制在合理范围
        return max(0.0, min(2.0, gain))

    def _convert_video_clips(self, vgp_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        转换主视频轨道和转场

        从transition_sequence_id或filter_sequence_id中提取
        """
        # 优先使用有滤镜的序列，否则使用转场序列
        sequence = vgp_result.get("filter_sequence_id") or \
                   vgp_result.get("effects_sequence_id") or \
                   vgp_result.get("transition_sequence_id") or \
                   []

        if not sequence:
            logger.warning("未找到视频剪辑序列")
            return []

        video_clips = []

        for i, clip in enumerate(sequence):
            # 计算时间范围
            timeline_in = float(clip.get("start", 0.0))
            timeline_out = float(clip.get("end", clip.get("start", 0.0) + clip.get("duration", 0.0)))

            # 防御性检查：确保timeline_out > timeline_in
            if timeline_out <= timeline_in:
                logger.warning(f"⚠️ 视频片段 {i+1} 时间范围无效 [{timeline_in} - {timeline_out}]，跳过")
                continue

            ims_clip = {
                "MediaURL": clip.get("source_url", ""),
                "TimelineIn": int(round(timeline_in)),
                "TimelineOut": int(round(timeline_out)),
                "Effects": []
            }

            # 添加转场 (在clip的Effects中)
            if "transition_out" in clip:
                transition = TransitionConverter.convert(clip["transition_out"])
                if transition:
                    # 尝试推断方向
                    next_clip = sequence[i + 1] if i + 1 < len(sequence) else None
                    if clip["transition_out"].get("type") in ["wipe_push", "slide"]:
                        subtype = TransitionConverter.infer_direction(
                            clip["transition_out"],
                            current_clip=clip,
                            next_clip=next_clip
                        )
                        transition["SubType"] = subtype

                    ims_clip["Effects"].append(transition)

            video_clips.append(ims_clip)

        return video_clips

    def _convert_filters(self, vgp_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        转换滤镜为EffectTrack

        从filter_sequence_id中提取color_filter
        """
        sequence = vgp_result.get("filter_sequence_id") or \
                   vgp_result.get("effects_sequence_id") or \
                   []

        if not sequence:
            return None

        filter_items = []

        for clip in sequence:
            if "color_filter" not in clip:
                continue

            color_filter = clip["color_filter"]

            # 根据配置选择转换方式
            if self.use_filter_preset:
                ims_filter = FilterConverter.convert_preset(color_filter)
            else:
                ims_filter = FilterConverter.convert_params(color_filter)

            # 添加时间范围
            timeline_in = float(clip.get("start", 0.0))
            timeline_out = float(clip.get("end", clip.get("start", 0.0) + clip.get("duration", 0.0)))

            # 防御性检查：确保timeline_out > timeline_in
            if timeline_out <= timeline_in:
                logger.warning(f"⚠️ 滤镜片段时间范围无效 [{timeline_in} - {timeline_out}]，跳过")
                continue

            ims_filter["TimelineIn"] = int(round(timeline_in))
            ims_filter["TimelineOut"] = int(round(timeline_out))

            filter_items.append(ims_filter)

        if not filter_items:
            return None

        return {
            "EffectTrackItems": filter_items
        }

    def _convert_effects(self, vgp_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        转换特效为EffectTrack

        从effects_sequence_id中提取visual_effects
        """
        sequence = vgp_result.get("effects_sequence_id", [])

        if not sequence:
            return None

        effect_items = []

        for clip in sequence:
            visual_effects = clip.get("visual_effects", [])

            for vgp_effect in visual_effects:
                ims_effect = EffectConverter.convert(vgp_effect)
                if ims_effect:
                    # 添加时间范围
                    timeline_in = float(clip.get("start", 0.0))
                    timeline_out = float(clip.get("end", clip.get("start", 0.0) + clip.get("duration", 0.0)))

                    # 防御性检查：确保timeline_out > timeline_in
                    if timeline_out <= timeline_in:
                        logger.warning(f"⚠️ 特效片段时间范围无效 [{timeline_in} - {timeline_out}]，跳过")
                        continue

                    ims_effect["TimelineIn"] = int(round(timeline_in))
                    ims_effect["TimelineOut"] = int(round(timeline_out))

                    effect_items.append(ims_effect)

        if not effect_items:
            return None

        return {
            "EffectTrackItems": effect_items
        }

    def _convert_text_overlay(self, vgp_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        转换文字叠加为TextTrack (花字)

        从text_overlay_track_id中提取
        """
        text_track = vgp_result.get("text_overlay_track_id")

        logger.debug(f"🌸 花字数据检查: type={type(text_track)}, is_dict={isinstance(text_track, dict)}")
        if isinstance(text_track, dict):
            logger.debug(f"   花字数据keys: {list(text_track.keys())}")
            clips_count = len(text_track.get("clips", []))
            logger.debug(f"   花字clips数量: {clips_count}")

        if not text_track or not isinstance(text_track, dict):
            logger.debug("   ⚠️ 花字数据为空或格式不正确")
            return None

        clips = text_track.get("clips", [])
        if not clips:
            logger.debug("   ⚠️ 花字clips为空")
            return None

        subtitle_clips = []

        for vgp_text in clips:
            ims_subtitle = FlowerTextConverter.convert(vgp_text)
            subtitle_clips.append(ims_subtitle)

        return {
            "SubtitleTrackClips": subtitle_clips  # ✅ 修正字段名：IMS期望的是SubtitleTrackClips
        }

    def _convert_auxiliary_media(self, vgp_result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        转换辅助媒体为额外的VideoTrack

        从auxiliary_track_id中提取
        """
        aux_track = vgp_result.get("auxiliary_track_id")

        if not aux_track or not isinstance(aux_track, dict):
            return None

        clips = aux_track.get("clips", [])
        if not clips:
            return None

        overlay_clips = []

        for vgp_media in clips:
            ims_clip = OverlayConverter.convert(vgp_media)
            overlay_clips.append(ims_clip)

        if not overlay_clips:
            return None

        return {
            "VideoTrackClips": overlay_clips
        }

    def convert_to_ims_request(self, vgp_result: Dict[str, Any],
                               output_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        转换为完整的IMS SubmitMediaProducingJob请求

        Args:
            vgp_result: VGP输出
            output_config: 输出配置 (分辨率、格式等)

        Returns:
            IMS API请求体
        """
        timeline = self.convert(vgp_result)

        # 默认输出配置
        if output_config is None:
            output_config = {
                "MediaURL": "oss://bucket/output.mp4",
                "Width": 1920,
                "Height": 1080,
                "VideoCodec": "H.264",
                "AudioCodec": "AAC"
            }

        request = {
            "Timeline": timeline,
            "OutputMediaConfig": output_config
        }

        return request

    def get_conversion_summary(self, vgp_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        获取转换摘要信息

        Args:
            vgp_result: VGP输出

        Returns:
            转换摘要
                {
                    "total_clips": 10,
                    "transitions": 9,
                    "filters": 10,
                    "effects": 5,
                    "texts": 3,
                    "overlays": 2,
                    "audio_tracks": 2,  # 新增
                    "warnings": [...]
                }
        """
        summary = {
            "total_clips": 0,
            "transitions": 0,
            "filters": 0,
            "effects": 0,
            "texts": 0,
            "overlays": 0,
            "audio_tracks": 0,  # ✅ 新增
            "warnings": []
        }

        # 统计clips
        sequence = vgp_result.get("filter_sequence_id") or \
                   vgp_result.get("effects_sequence_id") or \
                   vgp_result.get("transition_sequence_id") or \
                   []
        summary["total_clips"] = len(sequence)

        # 统计转场
        for clip in sequence:
            if "transition_out" in clip:
                trans_type = clip["transition_out"].get("type")
                if trans_type not in ["cut", "match_cut", "none"]:
                    summary["transitions"] += 1

        # 统计滤镜
        filter_seq = vgp_result.get("filter_sequence_id", [])
        summary["filters"] = sum(1 for clip in filter_seq if "color_filter" in clip)

        # 统计特效
        effects_seq = vgp_result.get("effects_sequence_id", [])
        for clip in effects_seq:
            summary["effects"] += len(clip.get("visual_effects", []))

        # 统计文字
        text_track = vgp_result.get("text_overlay_track_id", {})
        summary["texts"] = len(text_track.get("clips", []))

        # 统计辅助媒体
        aux_track = vgp_result.get("auxiliary_track_id", {})
        summary["overlays"] = len(aux_track.get("clips", []))

        # ✅ 统计音频轨道
        bgm_data = vgp_result.get("bgm_composition_id", {})
        if bgm_data and bgm_data.get("clips"):
            summary["audio_tracks"] += 1

        aux_audio_clips = [c for c in aux_track.get("clips", []) if c.get("type") == "audio"]
        if aux_audio_clips:
            summary["audio_tracks"] += 1

        return summary
