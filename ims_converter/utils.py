"""
IMS转换工具类

提供VGP各类参数到IMS格式的转换函数
"""

from typing import Dict, List, Optional, Any
import logging
from .configs.mappings import (
    VGP_TO_IMS_TRANSITION,
    VGP_TO_IMS_FILTER_PRESET,
    VGP_TO_IMS_EFFECT,
    VGP_TO_IMS_FLOWER_STYLE,
    VGP_TO_IMS_POSITION,
    COLOR_TO_FLOWER_STYLE,
    FILTER_PARAM_CONVERSION
)

logger = logging.getLogger(__name__)


class TransitionConverter:
    """转场效果转换器"""

    @staticmethod
    def convert(vgp_transition: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将VGP转场转换为IMS转场

        Args:
            vgp_transition: VGP转场对象
                {
                    "type": "cross_dissolve",
                    "name": "叠化",
                    "duration": 1.2,
                    "description": "..."
                }

        Returns:
            IMS转场对象或None(如果是cut/match_cut)
                {
                    "Type": "Transition",
                    "SubType": "fade",
                    "Duration": 1.2
                }
        """
        vgp_type = vgp_transition.get("type", "")
        ims_subtype = VGP_TO_IMS_TRANSITION.get(vgp_type)

        if ims_subtype is None:
            logger.debug(f"转场类型 '{vgp_type}' 不需要IMS转场效果 (cut或match_cut)")
            return None

        duration = vgp_transition.get("duration", 1.0)

        return {
            "Type": "Transition",
            "SubType": ims_subtype,
            "Duration": duration
        }

    @staticmethod
    def infer_direction(vgp_transition: Dict[str, Any],
                       current_clip: Optional[Dict] = None,
                       next_clip: Optional[Dict] = None) -> str:
        """
        根据镜头信息推断转场方向

        Args:
            vgp_transition: VGP转场对象
            current_clip: 当前镜头信息
            next_clip: 下一个镜头信息

        Returns:
            方向性转场的SubType (如 wiperight, wipeleft等)
        """
        vgp_type = vgp_transition.get("type", "")

        # 如果VGP已经指定方向
        if vgp_type in ["slide_right", "slide_left", "slide_up", "slide_down"]:
            return VGP_TO_IMS_TRANSITION[vgp_type]

        # 根据镜头运动推断方向
        if current_clip and next_clip:
            current_shot = current_clip.get("shot_type", "")
            next_shot = next_clip.get("shot_type", "")

            # 如果是推进镜头 (从远到近)
            if "aerial" in current_shot or "wide" in current_shot:
                if "close_up" in next_shot:
                    return "wiperight"  # 向右推进

            # 如果是拉远镜头 (从近到远)
            if "close_up" in current_shot:
                if "wide" in next_shot or "aerial" in next_shot:
                    return "wipeleft"  # 向左拉远

        # 默认向右
        return "wiperight"


class FilterConverter:
    """滤镜效果转换器"""

    @staticmethod
    def convert_preset(vgp_filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用预设滤镜转换 (推荐方式)

        Args:
            vgp_filter: VGP滤镜对象
                {
                    "preset": "cinematic",
                    "name": "电影感",
                    "intensity": 0.8,
                    ...
                }

        Returns:
            IMS滤镜对象
                {
                    "Type": "Filter",
                    "SubType": "m1"
                }
        """
        preset = vgp_filter.get("preset", "natural")
        ims_subtype = VGP_TO_IMS_FILTER_PRESET.get(preset, "pl1")  # 默认: 清新-暗影

        return {
            "Type": "Filter",
            "SubType": ims_subtype
        }

    @staticmethod
    def convert_params(vgp_filter: Dict[str, Any]) -> Dict[str, Any]:
        """
        使用精确参数转换 (高级方式)

        Args:
            vgp_filter: VGP滤镜对象 (包含applied_params)

        Returns:
            IMS自定义color滤镜对象
                {
                    "Type": "Filter",
                    "SubType": "color",
                    "ExtParams": "brightness=80,contrast=79,..."
                }
        """
        applied_params = vgp_filter.get("applied_params", {})

        # 转换各个参数
        ext_params_list = []

        # 亮度 (VGP: 倍数 → IMS: 偏移量)
        if "brightness" in applied_params:
            vgp_brightness = applied_params["brightness"]
            ims_brightness = FilterConverter._convert_param(
                vgp_brightness, "brightness"
            )
            ext_params_list.append(f"brightness={ims_brightness}")

        # 对比度
        if "contrast" in applied_params:
            vgp_contrast = applied_params["contrast"]
            ims_contrast = FilterConverter._convert_param(
                vgp_contrast, "contrast"
            )
            ext_params_list.append(f"contrast={ims_contrast}")

        # 饱和度
        if "saturation" in applied_params:
            vgp_saturation = applied_params["saturation"]
            ims_saturation = FilterConverter._convert_param(
                vgp_saturation, "saturation"
            )
            ext_params_list.append(f"saturation={ims_saturation}")

        # 色温
        if "temperature" in applied_params:
            vgp_temp = applied_params["temperature"]
            kelvin, ratio = FilterConverter._convert_temperature(vgp_temp)
            ext_params_list.append(f"kelvin_temperature={kelvin}")
            ext_params_list.append(f"temperature_ratio={ratio}")

        # 暗角 (如果有)
        if "vignette" in applied_params or "dark_corner" in applied_params:
            vignette = applied_params.get("vignette", applied_params.get("dark_corner", 0))
            dark_corner = int(vignette * 100)
            ext_params_list.append(f"dark_corner_ratio={dark_corner}")

        # 色调 (如果有)
        if "tint" in applied_params:
            tint = int(applied_params["tint"] * 100)
            ext_params_list.append(f"tint={tint}")

        ext_params = ",".join(ext_params_list) if ext_params_list else "brightness=0,contrast=0,saturation=0"

        return {
            "Type": "Filter",
            "SubType": "color",
            "ExtParams": ext_params
        }

    @staticmethod
    def _convert_param(vgp_value: float, param_name: str) -> int:
        """
        转换单个色彩参数

        Args:
            vgp_value: VGP参数值 (倍数, 1.0为基准)
            param_name: 参数名称

        Returns:
            IMS参数值 (偏移量, 0为基准)
        """
        config = FILTER_PARAM_CONVERSION.get(param_name)
        if not config:
            return 0

        vgp_center = config["vgp_center"]
        ims_center = config["ims_center"]
        ims_range = config["ims_range"]

        # 计算偏移量
        vgp_offset = vgp_value - vgp_center

        # 转换到IMS范围
        if vgp_offset > 0:
            # 正向偏移
            ims_value = ims_center + int(vgp_offset * ims_range[1])
        else:
            # 负向偏移
            ims_value = ims_center + int(vgp_offset * abs(ims_range[0]))

        # 限制在合法范围
        ims_value = max(ims_range[0], min(ims_range[1], ims_value))

        return ims_value

    @staticmethod
    def _convert_temperature(vgp_temp: float) -> tuple:
        """
        转换色温参数

        Args:
            vgp_temp: VGP色温 (-1.0冷 ~ 1.0暖)

        Returns:
            (kelvin, temperature_ratio)
        """
        config = FILTER_PARAM_CONVERSION["temperature"]
        neutral_kelvin = config["neutral_kelvin"]

        # 色温K值: 6000K为中性
        kelvin = neutral_kelvin + int(vgp_temp * 10000)
        kelvin = max(1000, min(40000, kelvin))

        # 强度比例: 0-100
        ratio = int(abs(vgp_temp) * 100)
        ratio = max(0, min(100, ratio))

        return kelvin, ratio


class EffectConverter:
    """特效转换器"""

    @staticmethod
    def convert(vgp_effect: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        将VGP特效转换为IMS特效

        Args:
            vgp_effect: VGP特效对象
                {
                    "name": "镜头光晕",
                    "type": "overlay",
                    "position": {"x": 960, "y": 540},
                    "opacity": 0.6,
                    ...
                }

        Returns:
            IMS特效对象或None(如果不支持)
                {
                    "Type": "VFX",
                    "SubType": "colorfulradial"
                }
        """
        # 从name或type推断特效类型
        effect_type = vgp_effect.get("type", "")
        effect_name = vgp_effect.get("name", "")

        # 尝试从映射表匹配
        ims_subtype = VGP_TO_IMS_EFFECT.get(effect_type)

        if ims_subtype is None:
            logger.warning(f"特效类型 '{effect_type}' 在IMS中不支持")
            return None

        return {
            "Type": "VFX",
            "SubType": ims_subtype
        }


class FlowerTextConverter:
    """花字转换器"""

    @staticmethod
    def convert(vgp_text: Dict[str, Any]) -> Dict[str, Any]:
        """
        将VGP文字转换为IMS花字

        Args:
            vgp_text: VGP文字对象
                {
                    "text": "精致视界",
                    "start": 0.0,
                    "duration": 5.0,
                    "position": "top-center",
                    "font": "https://fonts.com/ali-future.ttf",
                    "style": {
                        "color": "#FFFFFF",
                        "stroke": "#000000",
                        "size": 36,
                        "bold": True
                    }
                }

        Returns:
            IMS花字对象（符合阿里云IMS标准）
                {
                    "Type": "Text",
                    "Content": "精致视界",
                    "TimelineIn": 0,
                    "TimelineOut": 5,
                    "X": 0.5,
                    "Y": 0.1,
                    "Alignment": "TopCenter",
                    "FontSize": 150,
                    "EffectColorStyle": "CS0001-000011"
                }
        """
        style = vgp_text.get("style", {})

        # 转换位置
        position_data = FlowerTextConverter._convert_position(vgp_text.get("position", "top-center"))

        # 选择花字样式
        effect_color_style = FlowerTextConverter._select_flower_style(style)

        # 转换字体大小（IMS使用较大的绝对值，VGP使用相对小的值）
        # 修正策略：根据视频分辨率调整字号
        # - 720p视频（1280x720）：适中大小，不遮挡画面
        # - 1080p视频（1920x1080）：可以稍大
        # 默认按720p计算（大部分短视频都是720p）
        vgp_size = style.get("size", 36)

        # 智能字号映射（避免过大，确保不遮挡画面）
        # ✅ 再次减半：用户反馈当前大小还是太大
        if vgp_size <= 24:  # 小字
            font_size = 20  # 从40减半到20
        elif vgp_size <= 36:  # 中等字（默认）
            font_size = 28  # 从55减半到28
        elif vgp_size <= 48:  # 大字
            font_size = 35  # 从70减半到35
        else:  # 超大字
            font_size = 43  # 从85减半到43

        result = {
            "Type": "Text",  # ✅ IMS使用"Text"而不是"Subtitle"
            "Content": vgp_text.get("text", ""),
            "TimelineIn": int(vgp_text.get("start", 0.0)),  # IMS使用整数时间
            "TimelineOut": int(vgp_text.get("start", 0.0) + vgp_text.get("duration", 3.0)),
            "FontSize": font_size,
            "EffectColorStyle": effect_color_style,  # ✅ 使用EffectColorStyle而不是EffectColorStyleId
            **position_data  # ✅ 展开X, Y, Alignment
        }

        return result

    @staticmethod
    def _convert_position(vgp_position: str) -> Dict[str, Any]:
        """
        将VGP位置转换为IMS位置坐标

        Args:
            vgp_position: VGP位置字符串（如"top-center", "bottom-left"）

        Returns:
            IMS位置对象 {"X": 0.5, "Y": 0.1, "Alignment": "TopCenter"}
        """
        # VGP位置到IMS坐标的映射
        position_map = {
            "top-left":      {"X": 0.1, "Y": 0.1, "Alignment": "TopLeft"},
            "top-center":    {"X": 0.5, "Y": 0.1, "Alignment": "TopCenter"},
            "top-right":     {"X": 0.9, "Y": 0.1, "Alignment": "TopRight"},
            "center":        {"X": 0.5, "Y": 0.5, "Alignment": "Center"},
            "center-bottom": {"X": 0.5, "Y": 0.7, "Alignment": "TopCenter"},
            "bottom-left":   {"X": 0.1, "Y": 0.9, "Alignment": "BottomLeft"},
            "bottom-center": {"X": 0.5, "Y": 0.9, "Alignment": "BottomCenter"},
            "bottom-right":  {"X": 0.9, "Y": 0.9, "Alignment": "BottomRight"}
        }

        return position_map.get(vgp_position.lower(), position_map["top-center"])

    @staticmethod
    def _select_flower_style(style: Dict[str, Any]) -> str:
        """
        根据VGP样式选择IMS花字预设

        Args:
            style: VGP样式对象

        Returns:
            IMS EffectColorStyle ID（格式：CS0001-000001）
        """
        # 优先根据颜色匹配
        color = style.get("color", "#FFFFFF").upper()
        if color in COLOR_TO_FLOWER_STYLE:
            return COLOR_TO_FLOWER_STYLE[color]

        # 根据粗细和描边选择合适的系列
        is_bold = style.get("bold", False)
        has_stroke = "stroke" in style and style.get("stroke")

        if is_bold and has_stroke:
            # 粗体+描边：使用CS0001系列（醒目效果）
            return "CS0001-000011"  # 高级花字效果
        elif is_bold:
            # 粗体：使用CS0002系列（干净粗体）
            return "CS0002-000009"  # 粗体干净效果
        elif has_stroke:
            # 有描边：使用CS0001系列（优雅效果）
            return "CS0001-000004"  # 系统花字效果

        # 默认：使用经典白色花字
        return "CS0001-000001"  # ✅ 默认使用有效的IMS样式ID


class OverlayConverter:
    """辅助媒体叠加转换器"""

    @staticmethod
    def convert(vgp_media: Dict[str, Any]) -> Dict[str, Any]:
        """
        ���VGP辅助媒体转换为IMS视频轨道clip

        Args:
            vgp_media: VGP辅助媒体对象
                {
                    "file_path": "https://example.com/chart.png",
                    "start": 5.0,
                    "duration": 3.0,
                    "type": "image",
                    "placement": "overlay",
                    "position": "center",
                    "opacity": 0.95
                }

        Returns:
            IMS VideoTrackClip对象
                {
                    "MediaURL": "https://example.com/chart.png",
                    "TimelineIn": 5.0,
                    "TimelineOut": 8.0,
                    "Type": "Image",
                    "X": 0.5,
                    "Y": 0.5,
                    "Width": 0.3,
                    "Height": 0.3
                }
        """
        position_str = vgp_media.get("position", "center")
        position = VGP_TO_IMS_POSITION.get(position_str, {"X": 0.5, "Y": 0.5})

        media_type = vgp_media.get("type", "image")
        ims_type = "Image" if media_type == "image" else "Video"

        # ✅ IMS要求TimelineIn/TimelineOut必须是int类型
        timeline_in = int(round(vgp_media.get("start", 0.0)))
        timeline_out = int(round(vgp_media.get("start", 0.0) + vgp_media.get("duration", 3.0)))

        return {
            "MediaURL": vgp_media.get("file_path", ""),
            "TimelineIn": timeline_in,
            "TimelineOut": timeline_out,
            "Type": ims_type,
            "X": position["X"],
            "Y": position["Y"],
            "Width": 0.3,   # 默认占30%宽度
            "Height": 0.3   # 默认占30%高度
        }
