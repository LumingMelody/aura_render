import json
from typing import Dict, Any
from typing import List, Optional, Union
from .vgp_model import (
    Project, Track, TrackType, Clip, Source, Effect,
    EffectType, Transition, TransitionType, Marker,
    EffectParameterKeyframe, InterpolationType
)


class TimelineParser:
    """
    解析符合标准格式的 JSON 字符串或字典，生成 Project 对象。
    """

    @staticmethod
    def parse(json_input: Union[str, Dict[str, Any]]) -> Project:
        """
        解析 JSON 输入。

        Args:
            json_input: JSON 字符串或已加载的字典。

        Returns:
            解析后的 Project 对象。
        """
        if isinstance(json_input, str):
            data = json.loads(json_input)
        else:
            data = json_input

        # 创建 Project
        project = Project(
            version=data["version"],
            name=data.get("name", "Untitled Project"),
            description=data.get("description"),
            duration=data.get("duration"),
            fps=data.get("fps", 24.0),
            resolution=data.get("resolution", {"width": 1920, "height": 1080})
        )

        # 解析 Timeline
        timeline_data = data["timeline"]
        tracks = []

        for track_data in timeline_data["tracks"]:
            track_type = TrackType(track_data["type"])
            clips = []

            for clip_data in track_data["clips"]:
                # 解析 Source
                source_data = clip_data["source"]
                source = Source(
                    id=source_data["id"],
                    type=source_data["type"],
                    path=source_data.get("path"),
                    duration=source_data.get("duration"),
                    resolution=source_data.get("resolution"),
                    fps=source_data.get("fps")
                )

                # 解析 Effects
                effects = []
                for effect_data in clip_data.get("effects", []):
                    keyframes = []
                    for kf_data in effect_data.get("keyframes", []):
                        keyframes.append(EffectParameterKeyframe(
                            time=kf_data["time"],
                            parameters=kf_data["parameters"],
                            interpolation=InterpolationType(kf_data.get("interpolation", "linear"))
                        ))
                    effects.append(Effect(
                        id=effect_data["id"],
                        type=effect_data["type"],
                        enabled=effect_data.get("enabled", True),
                        parameters=effect_data.get("parameters", {}),
                        keyframes=keyframes
                    ))

                # 解析 Transitions
                trans_in = None
                if "transition_in" in clip_data:
                    trans_data = clip_data["transition_in"]
                    trans_in = Transition(
                        type=trans_data["type"],
                        duration=trans_data["duration"],
                        parameters=trans_data.get("parameters", {})
                    )

                trans_out = None
                if "transition_out" in clip_data:
                    trans_data = clip_data["transition_out"]
                    trans_out = Transition(
                        type=trans_data["type"],
                        duration=trans_data["duration"],
                        parameters=trans_data.get("parameters", {})
                    )

                # 解析 Markers
                markers = []
                for marker_data in clip_data.get("markers", []):
                    markers.append(Marker(
                        time=marker_data["time"],
                        label=marker_data["label"],
                        color=marker_data.get("color"),
                        note=marker_data.get("note")
                    ))

                # 创建 Clip
                # 注意：out_point 需要 source.duration。如果 source.duration 未知，则无法正确设置。
                # 这里假设 source.duration 在 JSON 中已提供。实际中可能需要外部元数据。
                out_point = clip_data.get("out")
                if out_point is None and source.duration is not None:
                    out_point = source.duration

                clip = Clip(
                    id=clip_data["id"],
                    source=source,
                    start=clip_data["start"],
                    duration=clip_data["duration"],
                    in_point=clip_data.get("in"),
                    out_point=out_point,
                    speed=clip_data.get("speed", 1.0),
                    reverse=clip_data.get("reverse", False),
                    enabled=clip_data.get("enabled", True),
                    name=clip_data.get("name"),
                    effects=effects,
                    transition_in=trans_in,
                    transition_out=trans_out,
                    markers=markers
                )
                clips.append(clip)

            # 创建 Track
            track = Track(
                id=track_data["id"],
                type=track_type,
                clips=clips,
                name=track_data.get("name"),
                enabled=track_data.get("enabled", True),
                locked=track_data.get("locked", False),
                volume=track_data.get("volume", 1.0)
            )
            tracks.append(track)

        # 创建 Timeline 并赋给 Project
        project.timeline = Timeline(
            tracks=tracks,
            metadata=timeline_data.get("metadata")
        )

        return project