"""
智能素材匹配系统
核心功能：根据分镜列表智能匹配素材，保证全局一致性
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional, Tuple, Union
from enum import Enum
from dataclasses import dataclass, field
from pathlib import Path
import json
import hashlib
from datetime import datetime
import sys
import os

# 添加multimodal模块到路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from multimodal.vl_models import IntegratedVLSystem
    HAS_VL_SYSTEM = True
except ImportError:
    HAS_VL_SYSTEM = False
    print("警告: VL模型系统不可用，将使用模拟实现")

try:
    from multimodal.qwen_integration import HybridVideoUnderstanding
    HAS_QWEN_SYSTEM = True
except ImportError:
    HAS_QWEN_SYSTEM = False
    print("警告: 完整Qwen模型系统不可用")

try:
    from multimodal.lightweight_qwen import LightweightVideoUnderstanding
    HAS_LIGHTWEIGHT_QWEN = True
except ImportError:
    HAS_LIGHTWEIGHT_QWEN = False
    print("警告: 轻量级Qwen系统不可用，将使用模拟实现")

logger = logging.getLogger(__name__)


class MaterialType(Enum):
    """素材类型"""
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"
    AI_GENERATED = "ai_generated"
    DIGITAL_HUMAN = "digital_human"


class StyleConsistency(Enum):
    """风格一致性级别"""
    PERFECT = "perfect"      # 完美一致
    HIGH = "high"           # 高度一致
    MEDIUM = "medium"       # 中等一致
    LOW = "low"             # 低一致性
    INCONSISTENT = "inconsistent"  # 不一致


class GenerationStrategy(Enum):
    """生成策略"""
    PURE_AI = "pure_ai"           # 纯AI片段
    DIGITAL_HUMAN = "digital_human"  # 数字人口播
    MIXED = "mixed"               # 混合模式


@dataclass
class ShotBlock:
    """分镜块"""
    shot_id: str
    description: str
    duration: float
    start_time: float
    end_time: float
    user_material: Optional[Dict[str, Any]] = None  # 用户提供的素材
    required_style: Dict[str, Any] = field(default_factory=dict)
    content_keywords: List[str] = field(default_factory=list)
    visual_requirements: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MaterialMatch:
    """素材匹配结果"""
    material_id: str
    material_type: MaterialType
    source_path: str
    confidence_score: float
    style_match_score: float
    content_match_score: float
    technical_specs: Dict[str, Any]
    thumbnail_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StyleProfile:
    """风格档案"""
    style_id: str
    dominant_colors: List[str]
    color_palette: Dict[str, float]
    lighting_style: str
    composition_style: str
    mood: str
    visual_elements: List[str]
    consistency_score: float


class VLModelInterface:
    """VL模型接口 - 用于风格识别和内容分析"""

    def __init__(self):
        self.logger = logger.getChild('VLModel')
        # 初始化真实的VL系统
        if HAS_VL_SYSTEM:
            self.vl_system = IntegratedVLSystem()
            self.logger.info("使用真实VL模型系统")
        else:
            self.vl_system = None
            self.logger.warning("VL模型系统不可用，使用模拟实现")

    async def analyze_visual_style(self, image_path: str) -> Dict[str, Any]:
        """分析图片的视觉风格"""
        if self.vl_system and HAS_VL_SYSTEM:
            try:
                return await self.vl_system.analyze_visual_style(image_path)
            except Exception as e:
                self.logger.error(f"真实VL模型分析失败，回退到模拟实现: {e}")

        # 模拟实现作为后备
        return await self._mock_analyze_visual_style(image_path)

    async def compare_visual_consistency(self, image1_path: str, image2_path: str) -> Dict[str, Any]:
        """比较两张图片的视觉一致性"""
        if self.vl_system and HAS_VL_SYSTEM:
            try:
                return await self.vl_system.compare_visual_consistency(image1_path, image2_path)
            except Exception as e:
                self.logger.error(f"真实VL模型一致性比较失败，回退到模拟实现: {e}")

        # 模拟实现作为后备
        return await self._mock_compare_visual_consistency(image1_path, image2_path)

    async def analyze_content_match(self, image_path: str, description: str) -> Dict[str, Any]:
        """分析图片内容与描述的匹配度"""
        if self.vl_system and HAS_VL_SYSTEM:
            try:
                return await self.vl_system.analyze_content_match(image_path, description)
            except Exception as e:
                self.logger.error(f"真实VL模型内容匹配失败，回退到模拟实现: {e}")

        # 模拟实现作为后备
        return await self._mock_analyze_content_match(image_path, description)

    async def _mock_analyze_visual_style(self, image_path: str) -> Dict[str, Any]:
        """模拟风格分析"""
        style_analysis = {
            'dominant_colors': ['#2C3E50', '#3498DB', '#ECF0F1'],
            'lighting': 'natural',
            'composition': 'centered',
            'mood': 'professional',
            'style_tags': ['modern', 'clean', 'corporate'],
            'confidence': 0.85
        }
        self.logger.debug(f"Mock analyzed style for {image_path}")
        return style_analysis

    async def _mock_compare_visual_consistency(self, image1_path: str, image2_path: str) -> Dict[str, Any]:
        """模拟一致性比较"""
        consistency_result = {
            'overall_consistency': 0.78,
            'color_consistency': 0.82,
            'lighting_consistency': 0.75,
            'style_consistency': 0.80,
            'details': {
                'similar_elements': ['lighting', 'composition'],
                'different_elements': ['color_tone'],
                'recommendations': ['adjust color balance']
            }
        }
        return consistency_result

    async def _mock_analyze_content_match(self, image_path: str, description: str) -> Dict[str, Any]:
        """模拟内容匹配分析"""
        content_match = {
            'match_score': 0.72,
            'relevant_objects': ['product', 'background', 'lighting'],
            'missing_elements': ['specific_angle'],
            'extra_elements': ['watermark'],
            'suitability': 'high'
        }
        return content_match


class VideoAnalysisTools:
    """视频分析工具 - 基于YOLO和Whisper"""

    def __init__(self):
        self.logger = logger.getChild('VideoAnalysis')
        # 初始化VL系统
        if HAS_VL_SYSTEM:
            self.vl_system = IntegratedVLSystem()
            self.logger.info("使用VL视频分析系统")
        else:
            self.vl_system = None
            self.logger.warning("VL视频分析系统不可用")

        # 初始化Qwen混合理解系统
        if HAS_QWEN_SYSTEM:
            self.qwen_system = HybridVideoUnderstanding()
            self.logger.info("使用Qwen混合视频理解系统")
        else:
            self.qwen_system = None
            self.logger.warning("Qwen视频理解系统不可用")

    async def analyze_video_content(self, video_path: str,
                                  analysis_level: str = "balanced") -> Dict[str, Any]:
        """
        分析视频内容
        支持多种分析级别：
        - lightweight: YOLO+Qwen快速分析
        - balanced: VL基础分析 + Qwen理解（默认）
        - comprehensive: 全面分析包含关键帧深度理解
        """
        analysis_results = {}

        # 优先使用Qwen混合理解系统
        if self.qwen_system and HAS_QWEN_SYSTEM:
            try:
                qwen_result = await self.qwen_system.understand_video(video_path, analysis_level)
                analysis_results['qwen_understanding'] = qwen_result

                # 提取关键信息到标准格式
                standard_result = self._convert_qwen_to_standard_format(qwen_result)
                analysis_results.update(standard_result)

                self.logger.info(f"Qwen视频理解完成: {analysis_level} 级别")
                return analysis_results
            except Exception as e:
                self.logger.error(f"Qwen视频理解失败，尝试VL系统: {e}")

        # 回退到VL系统
        if self.vl_system and HAS_VL_SYSTEM:
            try:
                vl_result = await self.vl_system.analyze_video_content(video_path)
                analysis_results['vl_analysis'] = vl_result
                analysis_results.update(vl_result)

                self.logger.info("VL视频分析完成")
                return analysis_results
            except Exception as e:
                self.logger.error(f"VL视频分析失败，使用模拟实现: {e}")

        # 最终后备：模拟实现
        mock_result = await self._mock_analyze_video_content(video_path)
        analysis_results['mock_analysis'] = mock_result
        analysis_results.update(mock_result)

        return analysis_results

    def _convert_qwen_to_standard_format(self, qwen_result: Dict[str, Any]) -> Dict[str, Any]:
        """将Qwen理解结果转换为标准格式"""
        try:
            # 从Qwen结果中提取基础信息
            yolo_features = qwen_result.get('yolo_features', {})
            basic_info = yolo_features.get('basic_info', {})

            # 构建标准格式
            standard = {
                'duration': basic_info.get('duration', 0),
                'fps': basic_info.get('fps', 30),
                'resolution': basic_info.get('resolution', '1920x1080'),
                'processing_method': 'qwen_hybrid'
            }

            # 人脸检测信息（从YOLO特征转换）
            detected_objects = yolo_features.get('detected_objects', {})
            if 'person' in detected_objects:
                person_detections = detected_objects['person']
                standard['face_detection'] = {
                    'faces_detected': len(person_detections),
                    'face_timestamps': [
                        {
                            'start': det['timestamp'],
                            'end': det['timestamp'] + 2.0,  # 估算持续时间
                            'confidence': det['confidence']
                        }
                        for det in person_detections
                    ]
                }
            else:
                standard['face_detection'] = {
                    'faces_detected': 0,
                    'face_timestamps': []
                }

            # 音频分析（从理解摘要推断）
            understanding_summary = qwen_result.get('understanding_summary', {})
            standard['audio_analysis'] = {
                'has_speech': 'speech' in str(understanding_summary).lower(),
                'speech_segments': [],
                'speech_quality': {
                    'quality': 'good' if understanding_summary.get('overall_quality') == 'high' else 'medium',
                    'score': understanding_summary.get('recommendation_score', 0.5)
                },
                'language': 'zh',  # 默认中文
                'language_confidence': 0.8
            }

            # 内容变化率
            scene_changes = yolo_features.get('scene_changes', [])
            standard['content_change_rate'] = {
                'scene_changes': [change['timestamp'] for change in scene_changes],
                'motion_intensity': 'medium'
            }

            # 关键帧
            standard['key_frames'] = yolo_features.get('key_moments', [5.0, 15.0, 25.0])

            # 添加理解层级信息
            standard['understanding_level'] = qwen_result.get('understanding_level', 'balanced')
            standard['qwen_summary'] = understanding_summary

            return standard

        except Exception as e:
            self.logger.error(f"Qwen结果格式转换失败: {e}")
            return {
                'duration': 30.0,
                'fps': 30,
                'resolution': '1920x1080',
                'processing_method': 'qwen_hybrid_fallback',
                'error': str(e)
            }

    async def select_optimal_segment(self, video_path: str, required_duration: float,
                                   content_requirements: Dict[str, Any]) -> Dict[str, Any]:
        """
        选择最优的视频片段
        """
        try:
            analysis = await self.analyze_video_content(video_path)

            # 基于内容分析选择最佳片段
            best_segment = {
                'start_time': 8.0,
                'end_time': 8.0 + required_duration,
                'confidence': 0.87,
                'selection_reason': 'optimal_face_detection_and_audio',
                'quality_metrics': {
                    'visual_quality': 0.92,
                    'audio_quality': 0.88,
                    'content_relevance': 0.85
                }
            }

            return best_segment

        except Exception as e:
            self.logger.error(f"Segment selection failed: {e}")
            return {'error': str(e)}

    async def _mock_analyze_video_content(self, video_path: str) -> Dict[str, Any]:
        """模拟视频内容分析"""
        try:
            analysis_result = {
                'duration': 45.0,
                'fps': 30,
                'resolution': '1920x1080',
                'face_detection': {
                    'faces_detected': 2,
                    'face_timestamps': [
                        {'start': 5.0, 'end': 15.0, 'confidence': 0.95},
                        {'start': 20.0, 'end': 35.0, 'confidence': 0.88}
                    ]
                },
                'audio_analysis': {
                    'has_speech': True,
                    'speech_segments': [
                        {'start': 0.0, 'end': 10.0, 'text': '产品介绍开始'},
                        {'start': 15.0, 'end': 25.0, 'text': '功能演示'}
                    ],
                    'speech_quality': {
                        'quality': 'good',
                        'score': 0.80,
                        'avg_confidence': 0.85,
                        'total_segments': 2
                    },
                    'language': 'zh',
                    'language_confidence': 0.90
                },
                'content_change_rate': {
                    'scene_changes': [5.0, 15.0, 25.0, 35.0],
                    'motion_intensity': 'medium'
                },
                'key_frames': [1.0, 8.0, 16.0, 28.0, 40.0]
            }

            self.logger.debug(f"Mock analyzed video content for {video_path}")
            return analysis_result

        except Exception as e:
            self.logger.error(f"Mock video analysis failed for {video_path}: {e}")
            return {'error': str(e)}


class MaterialLibraryInterface:
    """素材库接口"""

    def __init__(self):
        self.logger = logger.getChild('MaterialLibrary')

    async def search_materials(self, tags: List[str], material_type: MaterialType = None,
                             duration_range: Tuple[float, float] = None) -> List[Dict[str, Any]]:
        """
        通过标签搜索素材库
        """
        try:
            # 模拟素材库搜索结果
            mock_results = [
                {
                    'material_id': f'mat_{i:03d}',
                    'title': f'Material {i}',
                    'tags': tags[:2] + [f'tag_{i}'],
                    'type': material_type.value if material_type else 'video',
                    'duration': 15.0 + i * 5,
                    'thumbnail_url': f'https://example.com/thumb_{i}.jpg',
                    'download_url': f'https://example.com/material_{i}.mp4',
                    'metadata': {
                        'resolution': '1920x1080',
                        'fps': 30,
                        'file_size': f'{10 + i}MB'
                    }
                }
                for i in range(min(len(tags) * 2, 10))
            ]

            self.logger.debug(f"Found {len(mock_results)} materials for tags: {tags}")
            return mock_results

        except Exception as e:
            self.logger.error(f"Material library search failed: {e}")
            return []

    async def download_material(self, material_id: str, download_url: str) -> str:
        """下载素材到本地"""
        try:
            # 模拟下载过程
            local_path = f"/tmp/materials/{material_id}.mp4"
            # 实际实现中会下载文件
            self.logger.info(f"Downloaded material {material_id} to {local_path}")
            return local_path

        except Exception as e:
            self.logger.error(f"Material download failed for {material_id}: {e}")
            return ""


class AIVideoGenerator:
    """AI视频生成器"""

    def __init__(self):
        self.logger = logger.getChild('AIGenerator')

    async def generate_storyboard(self, description: str, style: Dict[str, Any],
                                duration: float) -> List[Dict[str, Any]]:
        """
        生成分镜图
        将描述切分成5秒段，生成分镜图
        """
        try:
            # 计算需要的分镜数量
            segments_count = max(1, int(duration / 5.0))

            storyboard = []
            for i in range(segments_count):
                start_time = i * 5.0
                end_time = min((i + 1) * 5.0, duration)

                frame_description = f"{description} - 第{i+1}段"
                storyboard.append({
                    'frame_id': f'frame_{i:02d}',
                    'start_time': start_time,
                    'end_time': end_time,
                    'duration': end_time - start_time,
                    'description': frame_description,
                    'style_prompt': self._generate_style_prompt(style),
                    'image_path': f'/tmp/storyboard/frame_{i:02d}.jpg'
                })

            self.logger.info(f"Generated storyboard with {len(storyboard)} frames")
            return storyboard

        except Exception as e:
            self.logger.error(f"Storyboard generation failed: {e}")
            return []

    async def generate_video_from_storyboard(self, storyboard: List[Dict[str, Any]],
                                           consistency_elements: Dict[str, Any]) -> Dict[str, Any]:
        """
        从分镜图生成视频
        保证连续性和一致性
        """
        try:
            # 模拟AI视频生成过程
            generated_videos = []

            for frame in storyboard:
                video_segment = {
                    'segment_id': frame['frame_id'],
                    'video_path': f"/tmp/generated/{frame['frame_id']}.mp4",
                    'duration': frame['duration'],
                    'generation_params': {
                        'style': frame['style_prompt'],
                        'consistency_elements': consistency_elements,
                        'previous_frame': frame.get('previous_frame_path')
                    },
                    'quality_score': 0.85
                }
                generated_videos.append(video_segment)

            # 合并视频段
            final_video = {
                'video_path': '/tmp/generated/final_ai_video.mp4',
                'segments': generated_videos,
                'total_duration': sum(seg['duration'] for seg in generated_videos),
                'consistency_score': 0.82
            }

            self.logger.info(f"Generated AI video with {len(generated_videos)} segments")
            return final_video

        except Exception as e:
            self.logger.error(f"AI video generation failed: {e}")
            return {'error': str(e)}

    def _generate_style_prompt(self, style: Dict[str, Any]) -> str:
        """生成风格提示词"""
        # 这里会使用视频生成.md中的优化提示词
        base_prompt = "高质量产品宣传视频"

        if 'colors' in style:
            base_prompt += f", {style['colors']}色调"

        if 'mood' in style:
            base_prompt += f", {style['mood']}氛围"

        return base_prompt


class DigitalHumanGenerator:
    """数字人生成器"""

    def __init__(self):
        self.logger = logger.getChild('DigitalHuman')

    async def select_digital_human(self, requirements: Dict[str, Any]) -> Dict[str, Any]:
        """选择最适合的数字人播音员"""
        try:
            # 模拟数字人选择
            selected_human = {
                'human_id': 'dh_001',
                'name': '专业播音员A',
                'style': 'professional',
                'voice_type': 'standard',
                'appearance': 'business',
                'suitability_score': 0.92
            }

            self.logger.info(f"Selected digital human: {selected_human['name']}")
            return selected_human

        except Exception as e:
            self.logger.error(f"Digital human selection failed: {e}")
            return {'error': str(e)}

    async def generate_digital_human_video(self, script: str, human_config: Dict[str, Any],
                                         duration: float) -> Dict[str, Any]:
        """生成数字人口播视频"""
        try:
            # 模拟数字人视频生成
            video_result = {
                'video_path': '/tmp/digital_human/output.mp4',
                'duration': duration,
                'human_config': human_config,
                'audio_path': '/tmp/digital_human/audio.wav',
                'quality_metrics': {
                    'lip_sync_accuracy': 0.95,
                    'voice_quality': 0.90,
                    'visual_quality': 0.88
                }
            }

            self.logger.info(f"Generated digital human video: {duration}s")
            return video_result

        except Exception as e:
            self.logger.error(f"Digital human video generation failed: {e}")
            return {'error': str(e)}


class IntelligentMaterialMatcher:
    """智能素材匹配器 - 核心匹配逻辑"""

    def __init__(self):
        self.vl_model = VLModelInterface()
        self.video_analyzer = VideoAnalysisTools()
        self.material_library = MaterialLibraryInterface()
        self.ai_generator = AIVideoGenerator()
        self.digital_human = DigitalHumanGenerator()
        self.logger = logger.getChild('MaterialMatcher')

    async def match_materials_for_shots(self, shot_blocks: List[ShotBlock],
                                      user_style_reference: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        为分镜列表匹配素材 - 核心匹配函数

        Args:
            shot_blocks: 分镜列表
            user_style_reference: 用户提供的风格参考

        Returns:
            匹配结果和策略
        """
        self.logger.info(f"Starting material matching for {len(shot_blocks)} shots")

        # 1. 确定全局风格
        global_style = await self._determine_global_style(shot_blocks, user_style_reference)

        # 2. 分析每个分镜的素材需求
        shot_analysis = await self._analyze_shot_requirements(shot_blocks, global_style)

        # 3. 素材匹配策略决策
        matching_strategy = await self._decide_matching_strategy(shot_analysis, global_style)

        # 4. 执行素材匹配
        matching_results = await self._execute_material_matching(shot_blocks, matching_strategy, global_style)

        # 5. 全局一致性验证
        consistency_check = await self._verify_global_consistency(matching_results, global_style)

        # 6. 如果一致性不足，执行fallback策略
        if consistency_check['consistency_score'] < 0.7:
            matching_results = await self._apply_fallback_strategy(shot_blocks, global_style, consistency_check)

        final_result = {
            'global_style': global_style,
            'matching_strategy': matching_strategy,
            'shot_materials': matching_results,
            'consistency_check': consistency_check,
            'total_duration': sum(shot.duration for shot in shot_blocks),
            'performance_metrics': {
                'library_matches': len([r for r in matching_results if r.get('source') == 'library']),
                'ai_generated': len([r for r in matching_results if r.get('source') == 'ai']),
                'user_provided': len([r for r in matching_results if r.get('source') == 'user'])
            }
        }

        self.logger.info(f"Material matching completed with consistency score: {consistency_check['consistency_score']:.2f}")
        return final_result

    async def _determine_global_style(self, shot_blocks: List[ShotBlock],
                                    user_style_reference: Optional[Dict[str, Any]]) -> StyleProfile:
        """确定全局风格"""
        if user_style_reference:
            # 如果用户提供了风格参考，以此为准
            self.logger.info("Using user-provided style reference")

            # 分析用户提供的素材风格
            user_material_path = user_style_reference.get('material_path')
            if user_material_path:
                style_analysis = await self.vl_model.analyze_visual_style(user_material_path)

                return StyleProfile(
                    style_id=f"user_style_{hash(user_material_path)}",
                    dominant_colors=style_analysis.get('dominant_colors', []),
                    color_palette={},  # 会进一步分析
                    lighting_style=style_analysis.get('lighting', 'natural'),
                    composition_style=style_analysis.get('composition', 'centered'),
                    mood=style_analysis.get('mood', 'neutral'),
                    visual_elements=style_analysis.get('style_tags', []),
                    consistency_score=1.0  # 用户提供的作为基准
                )

        # 如果没有用户风格参考，根据分镜描述确定最佳风格
        self.logger.info("Determining style from shot descriptions")

        # 分析所有分镜描述，提取风格关键词
        style_keywords = []
        mood_keywords = []

        for shot in shot_blocks:
            style_keywords.extend(shot.content_keywords)
            # 从描述中提取情绪词
            description_lower = shot.description.lower()
            if any(word in description_lower for word in ['专业', '商务', 'professional']):
                mood_keywords.append('professional')
            if any(word in description_lower for word in ['现代', '科技', 'modern']):
                mood_keywords.append('modern')

        # 生成统一风格档案
        unified_style = StyleProfile(
            style_id=f"unified_style_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            dominant_colors=['#2C3E50', '#3498DB', '#ECF0F1'],  # 默认商务风格
            color_palette={'primary': 0.6, 'secondary': 0.3, 'accent': 0.1},
            lighting_style='professional',
            composition_style='balanced',
            mood='professional' if 'professional' in mood_keywords else 'modern',
            visual_elements=list(set(style_keywords)),
            consistency_score=0.8
        )

        return unified_style

    async def _analyze_shot_requirements(self, shot_blocks: List[ShotBlock],
                                       global_style: StyleProfile) -> List[Dict[str, Any]]:
        """分析每个分镜的素材需求"""
        shot_analysis = []

        for shot in shot_blocks:
            analysis = {
                'shot_id': shot.shot_id,
                'has_user_material': shot.user_material is not None,
                'required_duration': shot.duration,
                'content_complexity': len(shot.content_keywords),
                'style_requirements': shot.required_style,
                'preferred_material_type': self._determine_preferred_material_type(shot),
                'search_tags': self._generate_search_tags(shot, global_style)
            }
            shot_analysis.append(analysis)

        return shot_analysis

    async def _decide_matching_strategy(self, shot_analysis: List[Dict[str, Any]],
                                      global_style: StyleProfile) -> Dict[str, Any]:
        """决策素材匹配策略"""
        total_duration = sum(analysis['required_duration'] for analysis in shot_analysis)
        user_materials_count = sum(1 for analysis in shot_analysis if analysis['has_user_material'])

        strategy = {
            'primary_approach': 'library_first',  # library_first, ai_first, mixed
            'fallback_to_ai': True,
            'consistency_threshold': 0.7,
            'generation_strategy': GenerationStrategy.PURE_AI,
            'digital_human_threshold': 120.0,  # 超过2分钟考虑数字人
            'cost_optimization': total_duration > 120.0
        }

        # 根据时长调整策略
        if total_duration > 120.0:
            # 长视频优先成本考虑
            strategy['primary_approach'] = 'library_first'
            strategy['consistency_threshold'] = 0.6  # 降低一致性要求

            # 考虑数字人口播
            speech_heavy = any('口播' in analysis.get('content_type', '') for analysis in shot_analysis)
            if speech_heavy:
                strategy['generation_strategy'] = GenerationStrategy.DIGITAL_HUMAN

        elif user_materials_count > 0:
            # 有用户素材时优先保持一致性
            strategy['consistency_threshold'] = 0.8
            strategy['primary_approach'] = 'mixed'

        return strategy

    async def _execute_material_matching(self, shot_blocks: List[ShotBlock],
                                       strategy: Dict[str, Any],
                                       global_style: StyleProfile) -> List[Dict[str, Any]]:
        """执行素材匹配"""
        matching_results = []

        for shot in shot_blocks:
            if shot.user_material:
                # 用户提供素材直接使用
                result = {
                    'shot_id': shot.shot_id,
                    'source': 'user',
                    'material_path': shot.user_material['path'],
                    'material_type': MaterialType.VIDEO,
                    'confidence': 1.0,
                    'processing_required': shot.user_material.get('duration', 0) > shot.duration
                }

                # 如果需要剪辑
                if result['processing_required']:
                    segment = await self.video_analyzer.select_optimal_segment(
                        shot.user_material['path'], shot.duration, shot.visual_requirements
                    )
                    result['selected_segment'] = segment

            else:
                # 需要匹配素材
                if strategy['primary_approach'] == 'library_first':
                    result = await self._match_from_library(shot, global_style)

                    if not result or result['confidence'] < 0.6:
                        result = await self._generate_ai_material(shot, global_style, strategy)
                else:
                    result = await self._generate_ai_material(shot, global_style, strategy)

            matching_results.append(result)

        return matching_results

    async def _match_from_library(self, shot: ShotBlock, global_style: StyleProfile) -> Dict[str, Any]:
        """从素材库匹配"""
        search_tags = self._generate_search_tags(shot, global_style)

        # 搜索素材库
        materials = await self.material_library.search_materials(
            search_tags, MaterialType.VIDEO, (shot.duration * 0.8, shot.duration * 2.0)
        )

        if not materials:
            return {'confidence': 0.0, 'source': 'library', 'error': 'no_materials_found'}

        # 评估每个素材的匹配度
        best_material = None
        best_score = 0.0

        for material in materials:
            # 下载缩略图进行VL分析
            thumbnail_path = material.get('thumbnail_url')
            if thumbnail_path:
                style_match = await self.vl_model.analyze_visual_style(thumbnail_path)
                content_match = await self.vl_model.analyze_content_match(thumbnail_path, shot.description)

                combined_score = (style_match.get('confidence', 0) * 0.4 +
                                content_match.get('match_score', 0) * 0.6)

                if combined_score > best_score:
                    best_score = combined_score
                    best_material = material

        if best_material and best_score > 0.6:
            # 下载并处理最佳素材
            local_path = await self.material_library.download_material(
                best_material['material_id'], best_material['download_url']
            )

            return {
                'shot_id': shot.shot_id,
                'source': 'library',
                'material_path': local_path,
                'material_type': MaterialType.VIDEO,
                'confidence': best_score,
                'original_material': best_material,
                'processing_required': best_material.get('duration', 0) > shot.duration
            }

        return {'confidence': 0.0, 'source': 'library', 'error': 'no_suitable_materials'}

    async def _generate_ai_material(self, shot: ShotBlock, global_style: StyleProfile,
                                   strategy: Dict[str, Any]) -> Dict[str, Any]:
        """生成AI素材"""
        generation_strategy = strategy['generation_strategy']

        if generation_strategy == GenerationStrategy.DIGITAL_HUMAN:
            # 数字人口播
            human_config = await self.digital_human.select_digital_human({
                'style': global_style.mood,
                'content': shot.description
            })

            video_result = await self.digital_human.generate_digital_human_video(
                shot.description, human_config, shot.duration
            )

            return {
                'shot_id': shot.shot_id,
                'source': 'ai_digital_human',
                'material_path': video_result.get('video_path'),
                'material_type': MaterialType.DIGITAL_HUMAN,
                'confidence': 0.85,
                'generation_config': human_config
            }

        else:
            # 纯AI生成
            # 先生成分镜图
            storyboard = await self.ai_generator.generate_storyboard(
                shot.description, global_style.__dict__, shot.duration
            )

            # 从分镜图生成视频
            consistency_elements = {
                'style': global_style.__dict__,
                'previous_shots': []  # 这里会传入前面的分镜信息保持一致性
            }

            video_result = await self.ai_generator.generate_video_from_storyboard(
                storyboard, consistency_elements
            )

            return {
                'shot_id': shot.shot_id,
                'source': 'ai_generated',
                'material_path': video_result.get('video_path'),
                'material_type': MaterialType.AI_GENERATED,
                'confidence': 0.80,
                'storyboard': storyboard,
                'consistency_score': video_result.get('consistency_score', 0.8)
            }

    async def _verify_global_consistency(self, matching_results: List[Dict[str, Any]],
                                       global_style: StyleProfile) -> Dict[str, Any]:
        """验证全局一致性"""
        if len(matching_results) < 2:
            return {'consistency_score': 1.0, 'details': 'single_shot'}

        # 取前几个素材进行一致性分析
        sample_materials = [r for r in matching_results[:3] if r.get('material_path')]

        if len(sample_materials) < 2:
            return {'consistency_score': 0.5, 'details': 'insufficient_materials'}

        # 比较素材间的视觉一致性
        consistency_scores = []

        for i in range(len(sample_materials) - 1):
            mat1_path = sample_materials[i]['material_path']
            mat2_path = sample_materials[i + 1]['material_path']

            # 从视频中提取关键帧进行比较
            consistency = await self.vl_model.compare_visual_consistency(mat1_path, mat2_path)
            consistency_scores.append(consistency['overall_consistency'])

        average_consistency = sum(consistency_scores) / len(consistency_scores)

        return {
            'consistency_score': average_consistency,
            'individual_scores': consistency_scores,
            'recommendation': 'acceptable' if average_consistency > 0.7 else 'needs_improvement',
            'details': {
                'materials_analyzed': len(sample_materials),
                'comparisons_made': len(consistency_scores)
            }
        }

    async def _apply_fallback_strategy(self, shot_blocks: List[ShotBlock],
                                     global_style: StyleProfile,
                                     consistency_check: Dict[str, Any]) -> List[Dict[str, Any]]:
        """应用fallback策略 - 保证一致性"""
        self.logger.warning(f"Applying fallback strategy due to low consistency: {consistency_check['consistency_score']}")

        # Fallback策略：全部AI生成以保证一致性
        fallback_results = []

        # 为所有分镜生成统一的AI素材
        consistency_elements = {
            'global_style': global_style.__dict__,
            'character_consistency': True,
            'color_consistency': True,
            'lighting_consistency': True
        }

        for shot in shot_blocks:
            if shot.user_material:
                # 用户素材保留
                result = {
                    'shot_id': shot.shot_id,
                    'source': 'user_preserved',
                    'material_path': shot.user_material['path'],
                    'material_type': MaterialType.VIDEO,
                    'confidence': 1.0
                }
            else:
                # AI生成保证一致性
                storyboard = await self.ai_generator.generate_storyboard(
                    shot.description, global_style.__dict__, shot.duration
                )

                video_result = await self.ai_generator.generate_video_from_storyboard(
                    storyboard, consistency_elements
                )

                result = {
                    'shot_id': shot.shot_id,
                    'source': 'ai_fallback',
                    'material_path': video_result.get('video_path'),
                    'material_type': MaterialType.AI_GENERATED,
                    'confidence': 0.85,
                    'consistency_enforced': True
                }

            fallback_results.append(result)

        return fallback_results

    def _determine_preferred_material_type(self, shot: ShotBlock) -> MaterialType:
        """确定首选素材类型"""
        if shot.user_material:
            return MaterialType.VIDEO

        # 根据内容特征确定类型
        description_lower = shot.description.lower()

        if any(word in description_lower for word in ['说话', '介绍', '讲解', '口播']):
            return MaterialType.DIGITAL_HUMAN

        return MaterialType.VIDEO

    def _generate_search_tags(self, shot: ShotBlock, global_style: StyleProfile) -> List[str]:
        """生成搜索标签"""
        tags = []

        # 基于内容关键词
        tags.extend(shot.content_keywords)

        # 基于全局风格
        tags.extend(global_style.visual_elements)
        tags.append(global_style.mood)

        # 基于描述提取的关键词
        description_keywords = shot.description.split()[:5]  # 取前5个词
        tags.extend(description_keywords)

        return list(set(tags))  # 去重


# 创建全局实例
intelligent_material_matcher = IntelligentMaterialMatcher()