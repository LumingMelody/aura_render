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
                    "auxiliary_track_id": {...}
                }

        Returns:
            IMS Timeline对象
                {
                    "VideoTracks": [...],
                    "EffectTracks": [...],
                    "TextTracks": [...]
                }
        """
        timeline = {
            "VideoTracks": [],
            "EffectTracks": [],
            "TextTracks": []
        }

        # 1. 转换主视频轨道 + 转场
        logger.info("开始转换视频轨道和转场...")
        video_clips = self._convert_video_clips(vgp_result)
        if video_clips:
            timeline["VideoTracks"].append({
                "VideoTrackClips": video_clips
            })

        # 2. 转换滤镜轨道
        logger.info("开始转换滤镜...")
        filter_track = self._convert_filters(vgp_result)
        if filter_track:
            timeline["EffectTracks"].append(filter_track)

        # 3. 转换特效轨道
        logger.info("开始转换特效...")
        effect_track = self._convert_effects(vgp_result)
        if effect_track:
            timeline["EffectTracks"].append(effect_track)

        # 4. 转换文字轨道(花字)
        logger.info("开始转换文字/花字...")
        text_track = self._convert_text_overlay(vgp_result)
        if text_track:
            timeline["TextTracks"].append(text_track)

        # 5. 转换辅助媒体 (作为额外的视频轨道)
        logger.info("开始转换辅助媒体...")
        aux_track = self._convert_auxiliary_media(vgp_result)
        if aux_track:
            timeline["VideoTracks"].append(aux_track)

        logger.info("IMS Timeline转换完成")
        return timeline

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
            ims_clip = {
                "MediaURL": clip.get("source_url", ""),
                "TimelineIn": clip.get("start", 0.0),
                "TimelineOut": clip.get("end", clip.get("start", 0.0) + clip.get("duration", 0.0)),
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
            ims_filter["TimelineIn"] = clip.get("start", 0.0)
            ims_filter["TimelineOut"] = clip.get("end", clip.get("start", 0.0) + clip.get("duration", 0.0))

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
                    ims_effect["TimelineIn"] = clip.get("start", 0.0)
                    ims_effect["TimelineOut"] = clip.get("end", clip.get("start", 0.0) + clip.get("duration", 0.0))

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

        if not text_track or not isinstance(text_track, dict):
            return None

        clips = text_track.get("clips", [])
        if not clips:
            return None

        subtitle_clips = []

        for vgp_text in clips:
            ims_subtitle = FlowerTextConverter.convert(vgp_text)
            subtitle_clips.append(ims_subtitle)

        return {
            "SubtitleClips": subtitle_clips
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

        return summary
