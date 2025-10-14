from abc import ABC, abstractmethod
from typing import Any, Dict

class AbstractVideoEditor(ABC):
    """
    抽象视频编辑器接口。定义了剪辑操作的通用方法。
    具体的实现（如 MockEditor, FFmpegEditor, MoviePyEditor）需要继承并实现这些方法。
    """

    @abstractmethod
    def load_source(self, source: Any) -> Any:
        """
        加载源媒体文件。返回一个内部表示（如代理对象）。
        Args:
            source: Source 对象。
        Returns:
            内部媒体对象引用。
        """
        pass

    @abstractmethod
    def create_clip(self, media_ref: Any, in_point: float, out_point: float, 
                   speed: float = 1.0, reverse: bool = False) -> Any:
        """
        从源媒体创建一个剪辑片段（定义入出点、速度、方向）。
        Args:
            media_ref: load_source 返回的引用。
            in_point, out_point: 裁剪点 (秒)。
            speed: 播放速度。
            reverse: 是否倒放。
        Returns:
            内部剪辑对象引用。
        """
        pass

    @abstractmethod
    def apply_effect(self, clip_ref: Any, effect: Any) -> Any:
        """
        向剪辑应用一个效果。
        Args:
            clip_ref: create_clip 返回的引用。
            effect: Effect 对象。
        Returns:
            应用效果后的新剪辑引用（可能原地修改或返回新对象）。
        """
        pass

    @abstractmethod
    def add_transition(self, clip1_ref: Any, clip2_ref: Any, transition: Any) -> Any:
        """
        在两个剪辑之间添加转场。通常修改 clip1 的结尾和/或 clip2 的开头。
        Args:
            clip1_ref: 前一个剪辑的引用。
            clip2_ref: 后一个剪辑的引用。
            transition: Transition 对象。
        Returns:
            组合后的剪辑引用或表示连接状态的对象。
        """
        pass

    @abstractmethod
    def add_to_track(self, track_type: str, clip_ref: Any, timeline_start: float) -> None:
        """
        将处理好的剪辑添加到指定类型的时间线轨道上。
        Args:
            track_type: 轨道类型字符串。
            clip_ref: 剪辑引用。
            timeline_start: 在时间线上的开始时间 (秒)。
        """
        pass

    @abstractmethod
    def set_track_volume(self, track_type: str, volume: float) -> None:
        """
        设置轨道音量（或视频轨道不透明度）。
        Args:
            track_type: 轨道类型字符串。
            volume: 0.0 - 1.0。
        """
        pass

    @abstractmethod
    def render(self, output_path: str, fps: float, resolution: Dict[str, int]) -> None:
        """
        渲染最终视频。
        Args:
            output_path: 输出文件路径。
            fps: 帧率。
            resolution: 分辨率 {"width": w, "height": h}。
        """
        pass