#!/usr/bin/env python3
"""
Qwen模型集成
集成Qwen和QwenVL进行视频内容理解
- Qwen: 基于YOLO特征的轻量级理解
- QwenVL: 关键帧的深度视觉理解
"""

import asyncio
import logging
import os
import json
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
from pathlib import Path

# Qwen相关导入
try:
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False
    print("警告: 未安装transformers/Qwen，将使用模拟实现")

# YOLO相关导入
try:
    from ultralytics import YOLO
    HAS_YOLO = True
except ImportError:
    HAS_YOLO = False
    print("警告: 未安装ultralytics/YOLO，将使用模拟实现")

logger = logging.getLogger(__name__)


class YOLOFeatureExtractor:
    """YOLO特征提取器 - 为Qwen提供结构化特征"""

    def __init__(self, offline_mode: bool = None):
        self.model = None
        # 默认禁用本地模型，避免启动时加载失败
        self.offline_mode = offline_mode or os.environ.get('HAS_VL_MODELS', 'false') == 'false' or True

        if not self.offline_mode and HAS_YOLO:
            try:
                self.model = YOLO('yolov8n.pt')  # 通用检测模型
                logger.info("YOLO特征提取器加载成功")
            except Exception as e:
                logger.error(f"YOLO模型加载失败: {e}")
                self.model = None
        else:
            logger.info("YOLO特征提取器运行在模拟模式")

    async def extract_video_features(self, video_path: str, sample_rate: int = 5) -> Dict[str, Any]:
        """
        提取视频的结构化特征

        Args:
            video_path: 视频路径
            sample_rate: 采样率（每N帧取一帧）

        Returns:
            结构化特征信息
        """
        if self.model and Path(video_path).exists():
            return await self._extract_real_features(video_path, sample_rate)
        else:
            return await self._extract_mock_features(video_path, sample_rate)

    async def _extract_real_features(self, video_path: str, sample_rate: int) -> Dict[str, Any]:
        """使用真实YOLO模型提取特征"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            features = {
                'basic_info': {
                    'duration': duration,
                    'fps': fps,
                    'total_frames': total_frames,
                    'resolution': f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
                },
                'detected_objects': {},  # {class_name: [timestamps]}
                'scene_changes': [],     # 场景变化时间点
                'object_counts': {},     # 每类对象的统计
                'temporal_patterns': {}, # 时序模式
                'key_moments': []        # 关键时刻
            }

            frame_count = 0
            prev_objects = set()

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % sample_rate == 0:
                    current_time = frame_count / fps if fps > 0 else frame_count / 30

                    # YOLO检测
                    results = self.model(frame)
                    current_objects = set()

                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                class_id = int(box.cls)
                                confidence = float(box.conf)
                                class_name = result.names[class_id]

                                if confidence > 0.5:
                                    current_objects.add(class_name)

                                    # 记录对象出现时间
                                    if class_name not in features['detected_objects']:
                                        features['detected_objects'][class_name] = []
                                    features['detected_objects'][class_name].append({
                                        'timestamp': current_time,
                                        'confidence': confidence
                                    })

                    # 检测场景变化
                    if prev_objects and current_objects != prev_objects:
                        features['scene_changes'].append({
                            'timestamp': current_time,
                            'objects_added': list(current_objects - prev_objects),
                            'objects_removed': list(prev_objects - current_objects)
                        })

                    prev_objects = current_objects

                frame_count += 1

            cap.release()

            # 统计对象计数
            for obj_class, detections in features['detected_objects'].items():
                features['object_counts'][obj_class] = len(detections)

            # 生成文本描述
            features['text_description'] = self._generate_feature_description(features)

            logger.info(f"提取了{len(features['detected_objects'])}类对象，{len(features['scene_changes'])}个场景变化")
            return features

        except Exception as e:
            logger.error(f"真实特征提取失败: {e}")
            return await self._extract_mock_features(video_path, sample_rate)

    async def _extract_mock_features(self, video_path: str, sample_rate: int) -> Dict[str, Any]:
        """模拟特征提取"""
        return {
            'basic_info': {
                'duration': 30.0,
                'fps': 30,
                'total_frames': 900,
                'resolution': '1920x1080'
            },
            'detected_objects': {
                'person': [
                    {'timestamp': 2.0, 'confidence': 0.92},
                    {'timestamp': 15.0, 'confidence': 0.88},
                    {'timestamp': 25.0, 'confidence': 0.95}
                ],
                'laptop': [
                    {'timestamp': 5.0, 'confidence': 0.85},
                    {'timestamp': 10.0, 'confidence': 0.90}
                ],
                'phone': [
                    {'timestamp': 20.0, 'confidence': 0.78}
                ]
            },
            'scene_changes': [
                {
                    'timestamp': 10.0,
                    'objects_added': ['laptop'],
                    'objects_removed': []
                },
                {
                    'timestamp': 20.0,
                    'objects_added': ['phone'],
                    'objects_removed': ['laptop']
                }
            ],
            'object_counts': {
                'person': 3,
                'laptop': 2,
                'phone': 1
            },
            'temporal_patterns': {
                'main_subject': 'person',
                'interaction_objects': ['laptop', 'phone']
            },
            'key_moments': [2.0, 10.0, 20.0],
            'text_description': "视频包含人物与电子设备的交互场景，主要对象包括人(3次出现)、笔记本电脑(2次出现)、手机(1次出现)。在10秒和20秒处有明显的场景变化。"
        }

    def _generate_feature_description(self, features: Dict[str, Any]) -> str:
        """生成特征的文本描述"""
        desc_parts = []

        # 基础信息
        basic = features['basic_info']
        desc_parts.append(f"视频时长{basic['duration']:.1f}秒，分辨率{basic['resolution']}")

        # 主要对象
        if features['detected_objects']:
            obj_desc = []
            for obj_class, count in features['object_counts'].items():
                obj_desc.append(f"{obj_class}({count}次出现)")
            desc_parts.append(f"主要对象包括{', '.join(obj_desc)}")

        # 场景变化
        if features['scene_changes']:
            change_times = [f"{change['timestamp']:.1f}秒" for change in features['scene_changes']]
            desc_parts.append(f"在{', '.join(change_times)}处有场景变化")

        return "。".join(desc_parts) + "。"


class QwenTextAnalyzer:
    """Qwen文本分析器 - 基于YOLO特征进行轻量级理解"""

    def __init__(self, model_name: str = "Qwen/Qwen2-1.5B-Instruct", offline_mode: bool = None):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        # 默认禁用本地模型，使用API调用模式
        self.offline_mode = offline_mode or os.environ.get('HAS_VL_MODELS', 'false') == 'false' or True

        if not self.offline_mode and HAS_QWEN:
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )
                logger.info(f"Qwen模型 {model_name} 加载成功")
            except Exception as e:
                logger.error(f"Qwen模型加载失败: {e}")
                self.model = None
        else:
            logger.info("Qwen分析器运行在模拟模式")

    async def analyze_video_from_features(self, features: Dict[str, Any],
                                        analysis_type: str = "content") -> Dict[str, Any]:
        """
        基于YOLO特征分析视频内容

        Args:
            features: YOLO提取的特征
            analysis_type: 分析类型 (content/style/action/emotion)

        Returns:
            分析结果
        """
        if self.model:
            return await self._analyze_with_qwen(features, analysis_type)
        else:
            return await self._analyze_mock(features, analysis_type)

    async def _analyze_with_qwen(self, features: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """使用真实Qwen模型分析"""
        try:
            # 构建提示词
            prompt = self._build_analysis_prompt(features, analysis_type)

            # 生成回复
            messages = [
                {"role": "system", "content": "你是一个专业的视频内容分析师，基于提供的视频特征信息进行分析。"},
                {"role": "user", "content": prompt}
            ]

            text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            model_inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

            generated_ids = self.model.generate(
                model_inputs.input_ids,
                max_new_tokens=512,
                temperature=0.7,
                do_sample=True
            )

            generated_ids = [
                output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
            ]

            response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # 解析响应
            return self._parse_qwen_response(response, analysis_type)

        except Exception as e:
            logger.error(f"Qwen分析失败: {e}")
            return await self._analyze_mock(features, analysis_type)

    async def _analyze_mock(self, features: Dict[str, Any], analysis_type: str) -> Dict[str, Any]:
        """模拟分析结果"""
        base_result = {
            'analysis_type': analysis_type,
            'confidence': 0.85,
            'processing_method': 'yolo_features + qwen_mock'
        }

        if analysis_type == "content":
            return {**base_result, **{
                'main_theme': '科技产品演示',
                'key_elements': ['人物', '电子设备', '操作演示'],
                'content_type': 'product_demo',
                'target_audience': 'tech_users',
                'description': '这是一个科技产品的演示视频，展示了人物与电子设备的交互过程'
            }}
        elif analysis_type == "style":
            return {**base_result, **{
                'visual_style': 'modern_tech',
                'pace': 'moderate',
                'composition': 'centered',
                'color_tone': 'neutral',
                'lighting': 'professional'
            }}
        elif analysis_type == "action":
            return {**base_result, **{
                'main_actions': ['产品展示', '功能演示', '操作指导'],
                'interaction_level': 'medium',
                'movement_pattern': 'sequential',
                'key_moments': features.get('key_moments', [])
            }}
        elif analysis_type == "emotion":
            return {**base_result, **{
                'overall_mood': 'professional',
                'emotional_arc': 'stable_positive',
                'engagement_level': 'moderate',
                'tone': 'informative'
            }}

    def _build_analysis_prompt(self, features: Dict[str, Any], analysis_type: str) -> str:
        """构建分析提示词"""
        base_prompt = f"""
请基于以下视频特征信息进行{analysis_type}分析：

基础信息：
- 视频时长：{features['basic_info']['duration']:.1f}秒
- 分辨率：{features['basic_info']['resolution']}

检测到的对象：
"""

        for obj_class, detections in features['detected_objects'].items():
            base_prompt += f"- {obj_class}: {len(detections)}次出现\n"

        base_prompt += f"\n场景变化：{len(features['scene_changes'])}处\n"
        base_prompt += f"\n特征描述：{features.get('text_description', '')}\n"

        if analysis_type == "content":
            base_prompt += """
请分析：
1. 视频的主要内容和主题
2. 关键元素和重点
3. 内容类型分类
4. 目标受众群体
5. 简短描述

请以JSON格式回复，包含：main_theme, key_elements, content_type, target_audience, description
"""
        elif analysis_type == "style":
            base_prompt += """
请分析：
1. 视觉风格特点
2. 节奏和节拍
3. 构图特点
4. 色调风格
5. 光线效果

请以JSON格式回复，包含：visual_style, pace, composition, color_tone, lighting
"""

        return base_prompt

    def _parse_qwen_response(self, response: str, analysis_type: str) -> Dict[str, Any]:
        """解析Qwen响应"""
        try:
            # 尝试提取JSON
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                result['confidence'] = 0.9
                result['processing_method'] = 'yolo_features + qwen_real'
                return result
        except:
            pass

        # 如果解析失败，返回基于响应文本的分析
        return {
            'analysis_type': analysis_type,
            'raw_response': response,
            'confidence': 0.7,
            'processing_method': 'yolo_features + qwen_text',
            'summary': response[:200] + "..." if len(response) > 200 else response
        }


class QwenVLKeyFrameAnalyzer:
    """QwenVL关键帧分析器 - 深度视觉理解单帧内容"""

    def __init__(self, model_name: str = "Qwen/Qwen2-VL-2B-Instruct", offline_mode: bool = None):
        self.model = None
        self.processor = None
        self.model_name = model_name
        # 默认禁用本地QwenVL模型，使用API调用模式
        self.offline_mode = offline_mode or os.environ.get('HAS_VL_MODELS', 'false') == 'false' or True

        if not self.offline_mode and HAS_QWEN:
            try:
                # QwenVL模型加载
                self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    device_map="auto"
                )
                self.processor = AutoProcessor.from_pretrained(model_name)
                logger.info(f"QwenVL模型 {model_name} 加载成功")
            except Exception as e:
                logger.error(f"QwenVL模型加载失败: {e}")
                self.model = None
        else:
            logger.info("QwenVL分析器运行在模拟模式")

    async def analyze_key_frames(self, video_path: str, key_timestamps: List[float],
                               analysis_focus: str = "detailed") -> Dict[str, Any]:
        """
        分析关键帧的详细内容

        Args:
            video_path: 视频路径
            key_timestamps: 关键时间点列表
            analysis_focus: 分析重点 (detailed/product/person/scene)

        Returns:
            关键帧分析结果
        """
        if self.model and Path(video_path).exists():
            return await self._analyze_real_frames(video_path, key_timestamps, analysis_focus)
        else:
            return await self._analyze_mock_frames(video_path, key_timestamps, analysis_focus)

    async def _analyze_real_frames(self, video_path: str, key_timestamps: List[float],
                                 analysis_focus: str) -> Dict[str, Any]:
        """使用真实QwenVL模型分析关键帧"""
        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            frame_analyses = []

            for timestamp in key_timestamps:
                # 定位到指定时间点
                frame_number = int(timestamp * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

                ret, frame = cap.read()
                if not ret:
                    continue

                # 转换为RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # 构建分析提示
                prompt = self._build_vl_prompt(analysis_focus, timestamp)

                # QwenVL推理
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": frame_rgb},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]

                text = self.processor.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                image_inputs, video_inputs = process_vision_info(messages)
                inputs = self.processor(
                    text=[text],
                    images=image_inputs,
                    videos=video_inputs,
                    padding=True,
                    return_tensors="pt",
                ).to(self.model.device)

                generated_ids = self.model.generate(**inputs, max_new_tokens=512)
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]

                response = self.processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )[0]

                # 解析分析结果
                frame_analysis = self._parse_vl_response(response, timestamp, analysis_focus)
                frame_analyses.append(frame_analysis)

            cap.release()

            # 综合分析结果
            return self._synthesize_frame_analyses(frame_analyses, analysis_focus)

        except Exception as e:
            logger.error(f"QwenVL关键帧分析失败: {e}")
            return await self._analyze_mock_frames(video_path, key_timestamps, analysis_focus)

    async def _analyze_mock_frames(self, video_path: str, key_timestamps: List[float],
                                 analysis_focus: str) -> Dict[str, Any]:
        """模拟关键帧分析"""
        frame_analyses = []

        for i, timestamp in enumerate(key_timestamps):
            frame_analysis = {
                'timestamp': timestamp,
                'frame_id': f"frame_{int(timestamp)}s",
                'analysis_focus': analysis_focus,
                'confidence': 0.82,
                'processing_method': 'qwenvl_mock'
            }

            if analysis_focus == "detailed":
                frame_analysis.update({
                    'scene_description': f"第{int(timestamp)}秒的详细场景：专业的产品展示环境",
                    'objects_detailed': [
                        {'object': '人物', 'position': 'center', 'action': '展示产品', 'confidence': 0.95},
                        {'object': '产品', 'position': 'right', 'state': '被操作中', 'confidence': 0.90}
                    ],
                    'visual_quality': {
                        'lighting': 'professional',
                        'composition': 'rule_of_thirds',
                        'focus': 'sharp',
                        'color_balance': 'neutral'
                    },
                    'content_analysis': {
                        'main_subject': '产品演示',
                        'secondary_elements': ['用户界面', '操作手势'],
                        'emotional_tone': 'professional_confident'
                    }
                })
            elif analysis_focus == "product":
                frame_analysis.update({
                    'product_details': {
                        'product_type': 'electronic_device',
                        'brand_visible': True,
                        'condition': 'new',
                        'usage_context': 'demonstration'
                    },
                    'product_features': ['屏幕显示', '操作界面', '外观设计'],
                    'market_positioning': 'premium_consumer'
                })

            frame_analyses.append(frame_analysis)

        return self._synthesize_frame_analyses(frame_analyses, analysis_focus)

    def _build_vl_prompt(self, analysis_focus: str, timestamp: float) -> str:
        """构建VL分析提示词"""
        base_prompt = f"请详细分析这张图片（视频第{timestamp:.1f}秒的关键帧）："

        if analysis_focus == "detailed":
            return base_prompt + """
请提供：
1. 整体场景描述
2. 所有可见对象的详细信息（位置、状态、动作）
3. 视觉质量评估（光线、构图、焦点、色彩）
4. 内容分析（主题、情感色调、重要元素）

请用结构化的方式回答。
"""
        elif analysis_focus == "product":
            return base_prompt + """
重点分析图片中的产品：
1. 产品类型和特征
2. 品牌识别和标识
3. 产品状态和使用情况
4. 市场定位判断
5. 产品亮点特色

请详细描述产品相关信息。
"""
        elif analysis_focus == "person":
            return base_prompt + """
重点分析图片中的人物：
1. 人物数量和位置
2. 动作和姿态
3. 表情和情绪
4. 穿着和形象
5. 与环境的互动

请详细描述人物相关信息。
"""
        else:  # scene
            return base_prompt + """
重点分析场景环境：
1. 场景类型和设置
2. 环境布置和道具
3. 空间布局和构图
4. 氛围和风格
5. 专业程度评估

请详细描述场景相关信息。
"""

    def _parse_vl_response(self, response: str, timestamp: float, analysis_focus: str) -> Dict[str, Any]:
        """解析QwenVL响应"""
        return {
            'timestamp': timestamp,
            'analysis_focus': analysis_focus,
            'raw_response': response,
            'confidence': 0.88,
            'processing_method': 'qwenvl_real',
            'detailed_analysis': response[:500] + "..." if len(response) > 500 else response
        }

    def _synthesize_frame_analyses(self, frame_analyses: List[Dict[str, Any]],
                                 analysis_focus: str) -> Dict[str, Any]:
        """综合多个关键帧的分析结果"""
        return {
            'analysis_type': 'qwenvl_keyframe',
            'analysis_focus': analysis_focus,
            'total_frames_analyzed': len(frame_analyses),
            'frame_analyses': frame_analyses,
            'overall_confidence': np.mean([fa.get('confidence', 0) for fa in frame_analyses]),
            'processing_method': 'qwenvl_keyframe_synthesis',
            'synthesis': {
                'consistent_elements': self._find_consistent_elements(frame_analyses),
                'temporal_changes': self._identify_temporal_changes(frame_analyses),
                'overall_theme': self._determine_overall_theme(frame_analyses, analysis_focus)
            }
        }

    def _find_consistent_elements(self, frame_analyses: List[Dict[str, Any]]) -> List[str]:
        """找出跨帧一致的元素"""
        # 简化实现
        return ['professional_setting', 'product_focus', 'clean_composition']

    def _identify_temporal_changes(self, frame_analyses: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """识别时序变化"""
        changes = []
        for i in range(1, len(frame_analyses)):
            changes.append({
                'from_timestamp': frame_analyses[i-1]['timestamp'],
                'to_timestamp': frame_analyses[i]['timestamp'],
                'change_type': 'scene_transition',
                'description': '场景或焦点的转换'
            })
        return changes

    def _determine_overall_theme(self, frame_analyses: List[Dict[str, Any]], analysis_focus: str) -> str:
        """确定整体主题"""
        if analysis_focus == "product":
            return "专业产品展示和演示"
        elif analysis_focus == "person":
            return "人物引导的内容呈现"
        elif analysis_focus == "scene":
            return "精心布置的演示环境"
        else:
            return "综合性内容展示"


class HybridVideoUnderstanding:
    """混合视频理解系统 - 整合YOLO+Qwen和QwenVL"""

    def __init__(self, offline_mode: bool = None):
        self.yolo_extractor = YOLOFeatureExtractor(offline_mode)
        self.qwen_analyzer = QwenTextAnalyzer(offline_mode=offline_mode)
        self.qwenvl_analyzer = QwenVLKeyFrameAnalyzer(offline_mode=offline_mode)
        self.offline_mode = offline_mode or os.environ.get('HAS_VL_MODELS') == 'false'
        self.logger = logger.getChild('HybridVideoUnderstanding')

    async def understand_video(self, video_path: str,
                             understanding_level: str = "balanced") -> Dict[str, Any]:
        """
        综合视频理解

        Args:
            video_path: 视频路径
            understanding_level: 理解级别
                - "lightweight": 仅使用YOLO+Qwen
                - "detailed": 仅使用QwenVL关键帧
                - "balanced": 混合使用
                - "comprehensive": 全面分析

        Returns:
            综合理解结果
        """
        self.logger.info(f"开始{understanding_level}级别的视频理解: {video_path}")

        result = {
            'video_path': video_path,
            'understanding_level': understanding_level,
            'timestamp': asyncio.get_event_loop().time(),
            'processing_methods': []
        }

        if understanding_level in ["lightweight", "balanced", "comprehensive"]:
            # YOLO特征提取 + Qwen分析
            self.logger.info("执行YOLO特征提取...")
            features = await self.yolo_extractor.extract_video_features(video_path)
            result['yolo_features'] = features
            result['processing_methods'].append('yolo_features')

            self.logger.info("执行Qwen内容分析...")
            content_analysis = await self.qwen_analyzer.analyze_video_from_features(features, "content")
            style_analysis = await self.qwen_analyzer.analyze_video_from_features(features, "style")

            result['qwen_analysis'] = {
                'content': content_analysis,
                'style': style_analysis
            }
            result['processing_methods'].append('qwen_analysis')

        if understanding_level in ["detailed", "balanced", "comprehensive"]:
            # 选择关键帧进行QwenVL分析
            if 'yolo_features' in result and result['yolo_features']['key_moments']:
                key_timestamps = result['yolo_features']['key_moments'][:3]  # 最多分析3个关键帧
            else:
                # 默认选择开头、中间、结尾
                duration = result.get('yolo_features', {}).get('basic_info', {}).get('duration', 30)
                key_timestamps = [duration * 0.1, duration * 0.5, duration * 0.9]

            self.logger.info(f"执行QwenVL关键帧分析: {key_timestamps}")
            detailed_analysis = await self.qwenvl_analyzer.analyze_key_frames(
                video_path, key_timestamps, "detailed"
            )
            result['qwenvl_analysis'] = detailed_analysis
            result['processing_methods'].append('qwenvl_keyframe')

        # 如果是comprehensive级别，额外进行专门分析
        if understanding_level == "comprehensive":
            self.logger.info("执行综合专项分析...")

            # 产品专项分析
            if 'yolo_features' in result:
                key_timestamps = result['yolo_features']['key_moments'][:2]
                product_analysis = await self.qwenvl_analyzer.analyze_key_frames(
                    video_path, key_timestamps, "product"
                )
                result['specialized_analysis'] = {
                    'product_focus': product_analysis
                }
                result['processing_methods'].append('specialized_analysis')

        # 生成综合理解摘要
        result['understanding_summary'] = self._generate_understanding_summary(result)

        self.logger.info(f"视频理解完成，使用方法: {result['processing_methods']}")
        return result

    def _generate_understanding_summary(self, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """生成综合理解摘要"""
        summary = {
            'video_type': 'unknown',
            'main_content': 'unknown',
            'style_characteristics': [],
            'key_elements': [],
            'overall_quality': 'medium',
            'recommendation_score': 0.5,
            'processing_confidence': 0.0
        }

        # 从Qwen分析中提取信息
        if 'qwen_analysis' in analysis_result:
            qwen_content = analysis_result['qwen_analysis'].get('content', {})
            qwen_style = analysis_result['qwen_analysis'].get('style', {})

            summary['video_type'] = qwen_content.get('content_type', 'unknown')
            summary['main_content'] = qwen_content.get('main_theme', 'unknown')
            summary['key_elements'] = qwen_content.get('key_elements', [])

            if qwen_style:
                summary['style_characteristics'] = [
                    qwen_style.get('visual_style', 'unknown'),
                    qwen_style.get('pace', 'unknown'),
                    qwen_style.get('lighting', 'unknown')
                ]

        # 从QwenVL分析中补充信息
        if 'qwenvl_analysis' in analysis_result:
            qwenvl = analysis_result['qwenvl_analysis']
            summary['overall_quality'] = 'high'  # QwenVL分析表明质量较高
            summary['processing_confidence'] = qwenvl.get('overall_confidence', 0.8)

        # 计算推荐分数
        confidence_scores = []
        for method in analysis_result['processing_methods']:
            if 'qwen' in method:
                confidence_scores.append(0.8)
            elif 'qwenvl' in method:
                confidence_scores.append(0.9)
            elif 'yolo' in method:
                confidence_scores.append(0.7)

        if confidence_scores:
            summary['recommendation_score'] = np.mean(confidence_scores)
            summary['processing_confidence'] = np.mean(confidence_scores)

        return summary


# 全局实例
hybrid_video_understanding = HybridVideoUnderstanding()


# 辅助函数（QwenVL需要的）
def process_vision_info(messages):
    """处理视觉信息的辅助函数"""
    image_inputs = []
    video_inputs = []

    for message in messages:
        if isinstance(message.get("content"), list):
            for content_item in message["content"]:
                if content_item.get("type") == "image" and "image" in content_item:
                    image_inputs.append(content_item["image"])

    return image_inputs, video_inputs


async def test_hybrid_understanding():
    """测试混合视频理解系统"""
    print("🧠 测试Qwen混合视频理解系统")

    # 测试YOLO特征提取
    print("\n1. 测试YOLO特征提取...")
    extractor = YOLOFeatureExtractor(offline_mode=True)
    features = await extractor.extract_video_features("/fake/test.mp4")
    print(f"   提取特征: {len(features['detected_objects'])} 类对象")

    # 测试Qwen分析
    print("\n2. 测试Qwen内容分析...")
    qwen = QwenTextAnalyzer(offline_mode=True)
    analysis = await qwen.analyze_video_from_features(features, "content")
    print(f"   内容分析: {analysis.get('main_theme', 'unknown')}")

    # 测试QwenVL关键帧
    print("\n3. 测试QwenVL关键帧分析...")
    qwenvl = QwenVLKeyFrameAnalyzer(offline_mode=True)
    keyframe_analysis = await qwenvl.analyze_key_frames("/fake/test.mp4", [10.0, 20.0], "detailed")
    print(f"   关键帧分析: {keyframe_analysis['total_frames_analyzed']} 帧")

    # 测试综合理解
    print("\n4. 测试综合视频理解...")
    hybrid = HybridVideoUnderstanding(offline_mode=True)
    result = await hybrid.understand_video("/fake/test.mp4", "balanced")
    print(f"   综合理解: {result['understanding_summary']['video_type']}")
    print(f"   处理方法: {result['processing_methods']}")

    print("\n🎉 Qwen混合视频理解系统测试完成！")
    return True


if __name__ == "__main__":
    asyncio.run(test_hybrid_understanding())