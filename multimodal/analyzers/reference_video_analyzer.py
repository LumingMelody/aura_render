"""
Reference Video Analyzer
参考视频分析器 - 提取风格特征、内容语义和技术参数
"""
import os
import logging
from typing import Dict, Any, Optional, List
from pathlib import Path
import asyncio
import json
from datetime import datetime

try:
    from moviepy import VideoFileClip
    import numpy as np
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    VideoFileClip = None

logger = logging.getLogger(__name__)


class VideoStyleExtractor:
    """视频风格特征提取器"""

    def __init__(self):
        self.logger = logger.getChild('VideoStyleExtractor')

    async def extract(self, video_path: str) -> Dict[str, Any]:
        """提取视频风格特征"""
        if not MOVIEPY_AVAILABLE:
            self.logger.warning("MoviePy不可用，使用fallback分析")
            return await self._fallback_style_analysis(video_path)

        try:
            # 异步处理视频文件
            return await asyncio.get_event_loop().run_in_executor(
                None, self._extract_style_sync, video_path
            )
        except Exception as e:
            self.logger.error(f"视频风格提取失败: {e}")
            return await self._fallback_style_analysis(video_path)

    def _extract_style_sync(self, video_path: str) -> Dict[str, Any]:
        """同步提取风格特征"""
        with VideoFileClip(video_path) as clip:
            # 1. 色彩分析 - 采样关键帧
            frames_to_sample = min(10, int(clip.duration))
            color_features = self._analyze_color_palette(clip, frames_to_sample)

            # 2. 节奏分析 - 分析剪辑节拍
            pace_features = self._analyze_video_pace(clip)

            # 3. 转场风格分析
            transition_features = self._analyze_transitions(clip)

            # 4. 整体美学评分
            aesthetic_score = self._calculate_aesthetic_score(color_features, pace_features)

            return {
                "color_palette": color_features,
                "pace_analysis": pace_features,
                "transition_style": transition_features,
                "aesthetic_score": aesthetic_score,
                "extraction_method": "moviepy_analysis"
            }

    def _analyze_color_palette(self, clip, sample_count: int) -> Dict[str, Any]:
        """分析视频色彩调色板"""
        try:
            sample_times = [i * clip.duration / sample_count for i in range(sample_count)]
            dominant_colors = []
            brightness_levels = []

            for t in sample_times:
                frame = clip.get_frame(t)
                # 简化的色彩分析
                avg_color = np.mean(frame, axis=(0, 1))
                brightness = np.mean(avg_color)

                dominant_colors.append(avg_color.tolist())
                brightness_levels.append(float(brightness))

            return {
                "dominant_colors": dominant_colors,
                "average_brightness": np.mean(brightness_levels),
                "color_variance": float(np.var(brightness_levels)),
                "color_temperature": self._estimate_color_temperature(dominant_colors)
            }
        except Exception as e:
            logger.error(f"色彩分析失败: {e}")
            return {"error": str(e), "method": "fallback"}

    def _analyze_video_pace(self, clip) -> Dict[str, Any]:
        """分析视频节奏"""
        try:
            duration = clip.duration
            fps = clip.fps

            # 简化的节奏分析
            if duration < 10:
                pace_category = "fast"
            elif duration < 30:
                pace_category = "medium"
            else:
                pace_category = "slow"

            return {
                "pace_category": pace_category,
                "average_shot_length": duration / max(1, duration / 3),  # 估算平均镜头长度
                "tempo_score": min(100, 60 / (duration / 10)),  # 节拍评分
                "dynamic_range": "medium"  # 暂时固定，后续可基于帧差分析
            }
        except Exception as e:
            logger.error(f"节奏分析失败: {e}")
            return {"error": str(e)}

    def _analyze_transitions(self, clip) -> Dict[str, Any]:
        """分析转场风格"""
        # 简化的转场分析
        duration = clip.duration

        if duration < 15:
            transition_style = "quick_cuts"
        elif duration < 60:
            transition_style = "smooth_fades"
        else:
            transition_style = "gradual_transitions"

        return {
            "style": transition_style,
            "frequency": "medium",
            "smoothness_score": 0.7
        }

    def _calculate_aesthetic_score(self, color_features: Dict, pace_features: Dict) -> float:
        """计算整体美学评分"""
        try:
            color_score = min(100, color_features.get("average_brightness", 50) * 2)
            pace_score = pace_features.get("tempo_score", 50)

            return (color_score + pace_score) / 2 / 100  # 标准化到0-1
        except:
            return 0.7  # 默认评分

    def _estimate_color_temperature(self, colors: List) -> str:
        """估算色温"""
        if not colors:
            return "neutral"

        avg_color = np.mean(colors, axis=0)
        if avg_color[2] > avg_color[0]:  # 蓝色分量高
            return "cool"
        elif avg_color[0] > avg_color[2]:  # 红色分量高
            return "warm"
        else:
            return "neutral"

    async def _fallback_style_analysis(self, video_path: str) -> Dict[str, Any]:
        """Fallback风格分析（无需MoviePy）"""
        file_size = os.path.getsize(video_path)
        filename = os.path.basename(video_path)

        # 基于文件名和大小的启发式分析
        if "product" in filename.lower() or "demo" in filename.lower():
            style_category = "professional"
            pace = "medium"
        elif "ad" in filename.lower() or "promo" in filename.lower():
            style_category = "dynamic"
            pace = "fast"
        else:
            style_category = "standard"
            pace = "medium"

        return {
            "style_category": style_category,
            "estimated_pace": pace,
            "file_size_mb": file_size / (1024 * 1024),
            "extraction_method": "fallback_heuristic",
            "confidence": 0.6
        }


class VideoContentAnalyzer:
    """视频内容语义理解分析器"""

    def __init__(self):
        self.logger = logger.getChild('VideoContentAnalyzer')

        # 预定义的场景和动作关键词
        self.scene_keywords = {
            "indoor": ["office", "home", "studio", "room"],
            "outdoor": ["street", "park", "nature", "city"],
            "product": ["demo", "showcase", "review", "unbox"],
            "people": ["person", "face", "human", "speaker"],
            "tech": ["computer", "phone", "device", "screen"]
        }

    async def analyze(self, video_path: str) -> Dict[str, Any]:
        """分析视频内容语义"""
        try:
            if MOVIEPY_AVAILABLE:
                return await self._analyze_with_frames(video_path)
            else:
                return await self._analyze_with_metadata(video_path)
        except Exception as e:
            self.logger.error(f"内容分析失败: {e}")
            return await self._fallback_content_analysis(video_path)

    async def _analyze_with_frames(self, video_path: str) -> Dict[str, Any]:
        """基于视频帧的内容分析"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._analyze_frames_sync, video_path
        )

    def _analyze_frames_sync(self, video_path: str) -> Dict[str, Any]:
        """同步帧分析"""
        with VideoFileClip(video_path) as clip:
            # 采样3个关键帧进行分析
            sample_times = [clip.duration * 0.2, clip.duration * 0.5, clip.duration * 0.8]

            scene_indicators = []
            motion_levels = []

            for t in sample_times:
                frame = clip.get_frame(t)

                # 简单的场景分析（基于颜色分布）
                scene_type = self._classify_scene_from_frame(frame)
                scene_indicators.append(scene_type)

                # 运动强度估算（基于梯度）
                motion = self._estimate_motion_level(frame)
                motion_levels.append(motion)

            # 综合分析结果
            dominant_scene = max(set(scene_indicators), key=scene_indicators.count)
            avg_motion = np.mean(motion_levels)

            return {
                "scene_type": dominant_scene,
                "motion_level": "high" if avg_motion > 0.7 else "medium" if avg_motion > 0.4 else "low",
                "content_complexity": self._assess_complexity(scene_indicators, motion_levels),
                "visual_elements": self._identify_visual_elements(scene_indicators),
                "analysis_method": "frame_analysis"
            }

    def _classify_scene_from_frame(self, frame) -> str:
        """从帧分类场景类型"""
        # 简化的场景分类（基于颜色特征）
        avg_brightness = np.mean(frame)
        color_std = np.std(frame)

        if avg_brightness < 50:
            return "indoor"
        elif avg_brightness > 200:
            return "outdoor"
        elif color_std > 50:
            return "complex"
        else:
            return "simple"

    def _estimate_motion_level(self, frame) -> float:
        """估算运动强度"""
        # 基于图像梯度的运动估计
        gray = np.mean(frame, axis=2) if len(frame.shape) == 3 else frame
        gradient = np.gradient(gray)
        motion_score = np.mean(np.abs(gradient))
        return min(1.0, motion_score / 50)  # 标准化

    def _assess_complexity(self, scenes: List, motions: List) -> str:
        """评估内容复杂度"""
        scene_variety = len(set(scenes))
        avg_motion = np.mean(motions)

        if scene_variety > 2 or avg_motion > 0.7:
            return "high"
        elif scene_variety > 1 or avg_motion > 0.4:
            return "medium"
        else:
            return "low"

    def _identify_visual_elements(self, scenes: List) -> List[str]:
        """识别视觉元素"""
        elements = []
        if "outdoor" in scenes:
            elements.append("natural_lighting")
        if "indoor" in scenes:
            elements.append("artificial_lighting")
        if "complex" in scenes:
            elements.append("multiple_objects")

        return elements or ["simple_composition"]

    async def _analyze_with_metadata(self, video_path: str) -> Dict[str, Any]:
        """基于元数据的内容分析"""
        filename = os.path.basename(video_path).lower()

        # 基于文件名的启发式分析
        detected_scenes = []
        for scene_type, keywords in self.scene_keywords.items():
            if any(kw in filename for kw in keywords):
                detected_scenes.append(scene_type)

        return {
            "detected_content_types": detected_scenes or ["general"],
            "filename_analysis": filename,
            "analysis_method": "metadata_heuristic",
            "confidence": 0.5
        }

    async def _fallback_content_analysis(self, video_path: str) -> Dict[str, Any]:
        """Fallback内容分析"""
        return {
            "content_type": "unknown",
            "analysis_method": "fallback",
            "confidence": 0.3,
            "error": "Unable to perform detailed content analysis"
        }


class VideoTechAnalyzer:
    """视频技术参数识别器"""

    def __init__(self):
        self.logger = logger.getChild('VideoTechAnalyzer')

    async def identify(self, video_path: str) -> Dict[str, Any]:
        """识别视频技术参数"""
        try:
            if MOVIEPY_AVAILABLE:
                return await self._analyze_with_moviepy(video_path)
            else:
                return await self._analyze_with_ffprobe(video_path)
        except Exception as e:
            self.logger.error(f"技术参数分析失败: {e}")
            return await self._fallback_tech_analysis(video_path)

    async def _analyze_with_moviepy(self, video_path: str) -> Dict[str, Any]:
        """使用MoviePy分析技术参数"""
        return await asyncio.get_event_loop().run_in_executor(
            None, self._moviepy_analysis_sync, video_path
        )

    def _moviepy_analysis_sync(self, video_path: str) -> Dict[str, Any]:
        """同步MoviePy分析"""
        with VideoFileClip(video_path) as clip:
            return {
                "duration": clip.duration,
                "fps": clip.fps,
                "resolution": {
                    "width": clip.w,
                    "height": clip.h
                },
                "aspect_ratio": round(clip.w / clip.h, 2),
                "has_audio": clip.audio is not None,
                "codec": "unknown",  # MoviePy不直接提供编码信息
                "analysis_method": "moviepy"
            }

    async def _analyze_with_ffprobe(self, video_path: str) -> Dict[str, Any]:
        """使用ffprobe分析（如果可用）"""
        # 简化实现，实际可以调用ffprobe命令
        file_stats = os.stat(video_path)

        return {
            "file_size": file_stats.st_size,
            "file_format": Path(video_path).suffix.lower(),
            "analysis_method": "file_system",
            "quality_estimate": self._estimate_quality_from_size(file_stats.st_size)
        }

    def _estimate_quality_from_size(self, file_size: int) -> str:
        """基于文件大小估算质量"""
        size_mb = file_size / (1024 * 1024)

        if size_mb > 100:
            return "high"
        elif size_mb > 20:
            return "medium"
        else:
            return "low"

    async def _fallback_tech_analysis(self, video_path: str) -> Dict[str, Any]:
        """Fallback技术分析"""
        file_extension = Path(video_path).suffix.lower()
        file_size = os.path.getsize(video_path)

        return {
            "file_format": file_extension,
            "file_size_mb": round(file_size / (1024 * 1024), 2),
            "estimated_duration": "unknown",
            "analysis_method": "fallback",
            "supported_format": file_extension in ['.mp4', '.mov', '.avi', '.mkv']
        }


class ReferenceVideoAnalyzer:
    """参考视频分析器主类"""

    def __init__(self):
        self.logger = logger.getChild('ReferenceVideoAnalyzer')
        self.style_extractor = VideoStyleExtractor()
        self.content_analyzer = VideoContentAnalyzer()
        self.tech_analyzer = VideoTechAnalyzer()

    async def analyze_reference_video(
        self,
        video_path: str,
        analysis_type: str,
        weight: float = 1.0
    ) -> Dict[str, Any]:
        """
        分析参考视频

        Args:
            video_path: 视频文件路径
            analysis_type: 分析类型 ('style_reference', 'content_reference', 'technique_reference')
            weight: 分析权重 (0.0-1.0)

        Returns:
            分析结果字典
        """
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"视频文件不存在: {video_path}")

        self.logger.info(f"开始分析参考视频: {video_path} (类型: {analysis_type}, 权重: {weight})")

        start_time = datetime.now()

        try:
            # 并行执行三个分析任务
            style_task = self.style_extractor.extract(video_path)
            content_task = self.content_analyzer.analyze(video_path)
            tech_task = self.tech_analyzer.identify(video_path)

            style_features, content_semantics, technical_params = await asyncio.gather(
                style_task, content_task, tech_task, return_exceptions=True
            )

            # 处理异常结果
            if isinstance(style_features, Exception):
                self.logger.error(f"风格分析失败: {style_features}")
                style_features = {"error": str(style_features)}

            if isinstance(content_semantics, Exception):
                self.logger.error(f"内容分析失败: {content_semantics}")
                content_semantics = {"error": str(content_semantics)}

            if isinstance(technical_params, Exception):
                self.logger.error(f"技术分析失败: {technical_params}")
                technical_params = {"error": str(technical_params)}

            analysis_time = (datetime.now() - start_time).total_seconds()

            result = {
                "video_path": video_path,
                "analysis_type": analysis_type,
                "analysis_weight": weight,
                "style_features": style_features,
                "content_semantics": content_semantics,
                "technical_params": technical_params,
                "analysis_metadata": {
                    "analysis_time": analysis_time,
                    "timestamp": datetime.now().isoformat(),
                    "moviepy_available": MOVIEPY_AVAILABLE,
                    "success": True
                }
            }

            self.logger.info(f"视频分析完成: {video_path} (耗时: {analysis_time:.2f}s)")
            return result

        except Exception as e:
            self.logger.error(f"视频分析失败: {e}")
            return {
                "video_path": video_path,
                "analysis_type": analysis_type,
                "analysis_weight": weight,
                "error": str(e),
                "analysis_metadata": {
                    "analysis_time": (datetime.now() - start_time).total_seconds(),
                    "timestamp": datetime.now().isoformat(),
                    "moviepy_available": MOVIEPY_AVAILABLE,
                    "success": False
                }
            }

    async def batch_analyze_videos(
        self,
        video_references: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        批量分析多个参考视频

        Args:
            video_references: 视频引用列表，格式为 [{"url": "path", "type": "style_reference", "weight": 0.7}, ...]

        Returns:
            分析结果列表
        """
        self.logger.info(f"开始批量分析 {len(video_references)} 个参考视频")

        tasks = [
            self.analyze_reference_video(
                video_ref["url"],
                video_ref.get("type", "style_reference"),
                video_ref.get("weight", 1.0)
            )
            for video_ref in video_references
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"视频 {i} 分析失败: {result}")
                processed_results.append({
                    "error": str(result),
                    "video_index": i,
                    "success": False
                })
            else:
                processed_results.append(result)

        self.logger.info(f"批量分析完成: {len(processed_results)} 个结果")
        return processed_results


# 用于测试的主函数
async def main():
    """测试ReferenceVideoAnalyzer"""
    analyzer = ReferenceVideoAnalyzer()

    # 测试单个视频分析
    test_video = "/path/to/test/video.mp4"  # 替换为实际测试视频路径

    if os.path.exists(test_video):
        result = await analyzer.analyze_reference_video(
            test_video,
            "style_reference",
            0.8
        )
        print("分析结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"测试视频文件不存在: {test_video}")
        print("请提供有效的视频文件路径进行测试")


if __name__ == "__main__":
    asyncio.run(main())