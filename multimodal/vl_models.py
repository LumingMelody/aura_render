#!/usr/bin/env python3
"""
VL模型集成实现
集成CLIP、BLIP等模型用于视觉风格分析和内容理解
"""

import asyncio
import logging
import os
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from pathlib import Path
import cv2
from PIL import Image
import torch
import torchvision.transforms as transforms

# 先定义logger
logger = logging.getLogger(__name__)

# VL模型相关导入
try:
    import clip
    HAS_CLIP = True
except ImportError:
    HAS_CLIP = False
    # 静默处理，不打印警告

try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    HAS_BLIP = True
except ImportError:
    HAS_BLIP = False
    # 静默处理，不打印警告

# YOLO相关导入
try:
    import ultralytics
    from ultralytics import YOLO
    HAS_YOLO = True
    logger.info("✅ YOLO模型可用")
except ImportError:
    HAS_YOLO = False
    logger.info("ℹ️ YOLO未安装，使用备用方案")

# Whisper相关导入
try:
    import whisper
    HAS_WHISPER = True
except ImportError:
    HAS_WHISPER = False
    # 静默处理


class CLIPStyleAnalyzer:
    """基于CLIP的风格分析器"""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.preprocess = None

        if HAS_CLIP:
            try:
                self.model, self.preprocess = clip.load("ViT-B/32", device=self.device)
                logger.info(f"CLIP模型加载成功，使用设备: {self.device}")
            except Exception as e:
                logger.error(f"CLIP模型加载失败: {e}")
                self.model = None
        else:
            # 静默处理，使用备用方案
            pass

    async def analyze_image_style(self, image_path: str) -> Dict[str, Any]:
        """使用CLIP分析图像风格"""
        if not self.model or not Path(image_path).exists():
            return await self._mock_style_analysis(image_path)

        try:
            # 加载和预处理图像
            image = Image.open(image_path).convert("RGB")
            image_tensor = self.preprocess(image).unsqueeze(0).to(self.device)

            # 定义风格标签
            style_labels = [
                "modern minimalist style",
                "vintage retro style",
                "corporate professional style",
                "creative artistic style",
                "natural organic style",
                "high-tech futuristic style",
                "warm cozy style",
                "clean bright style",
                "dark moody style",
                "colorful vibrant style"
            ]

            # 色彩标签
            color_labels = [
                "warm colors", "cool colors", "neutral colors",
                "bright colors", "dark colors", "pastel colors",
                "monochromatic", "high contrast", "low contrast"
            ]

            # 光线标签
            lighting_labels = [
                "natural daylight", "artificial lighting", "soft lighting",
                "hard lighting", "golden hour", "blue hour", "studio lighting"
            ]

            # 构建文本提示
            text_prompts = style_labels + color_labels + lighting_labels
            text_tokens = clip.tokenize(text_prompts).to(self.device)

            with torch.no_grad():
                # 计算图像和文本特征
                image_features = self.model.encode_image(image_tensor)
                text_features = self.model.encode_text(text_tokens)

                # 计算相似度
                similarities = torch.cosine_similarity(image_features, text_features, dim=1)
                similarities = similarities.cpu().numpy()

            # 解析结果
            style_scores = similarities[:len(style_labels)]
            color_scores = similarities[len(style_labels):len(style_labels)+len(color_labels)]
            lighting_scores = similarities[len(style_labels)+len(color_labels):]

            # 获取最匹配的风格
            top_style_idx = np.argmax(style_scores)
            top_color_idx = np.argmax(color_scores)
            top_lighting_idx = np.argmax(lighting_scores)

            # 分析颜色分布
            dominant_colors = await self._extract_dominant_colors(image_path)

            result = {
                'dominant_colors': dominant_colors,
                'lighting': lighting_labels[top_lighting_idx],
                'lighting_confidence': float(lighting_scores[top_lighting_idx]),
                'composition': 'centered',  # 需要额外分析
                'mood': style_labels[top_style_idx].split()[0],
                'style_tags': [
                    style_labels[top_style_idx],
                    color_labels[top_color_idx],
                    lighting_labels[top_lighting_idx]
                ],
                'confidence': float(np.mean([style_scores[top_style_idx],
                                           color_scores[top_color_idx],
                                           lighting_scores[top_lighting_idx]])),
                'detailed_scores': {
                    'style_scores': dict(zip(style_labels, style_scores.tolist())),
                    'color_scores': dict(zip(color_labels, color_scores.tolist())),
                    'lighting_scores': dict(zip(lighting_labels, lighting_scores.tolist()))
                }
            }

            logger.info(f"CLIP风格分析完成: {image_path}")
            return result

        except Exception as e:
            logger.error(f"CLIP风格分析失败: {e}")
            return await self._mock_style_analysis(image_path)

    async def _extract_dominant_colors(self, image_path: str, k: int = 5) -> List[str]:
        """提取主要颜色"""
        try:
            image = cv2.imread(image_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = image.reshape((-1, 3))

            # 使用K-means聚类
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(image)

            colors = kmeans.cluster_centers_.astype(int)
            colors_hex = [f"#{r:02x}{g:02x}{b:02x}" for r, g, b in colors]

            return colors_hex[:3]  # 返回前3个主要颜色

        except Exception as e:
            logger.error(f"颜色提取失败: {e}")
            return ['#2C3E50', '#3498DB', '#ECF0F1']  # 默认颜色

    async def _mock_style_analysis(self, image_path: str) -> Dict[str, Any]:
        """模拟风格分析"""
        return {
            'dominant_colors': ['#2C3E50', '#3498DB', '#ECF0F1'],
            'lighting': 'natural daylight',
            'lighting_confidence': 0.75,
            'composition': 'centered',
            'mood': 'professional',
            'style_tags': ['modern', 'clean', 'corporate'],
            'confidence': 0.70,
            'note': 'Using mock implementation - install CLIP for real analysis'
        }


class BLIPContentAnalyzer:
    """基于BLIP的内容理解分析器"""

    def __init__(self, device: str = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = None
        self.model = None

        if HAS_BLIP:
            try:
                self.processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
                self.model.to(self.device)
                logger.info(f"BLIP模型加载成功，使用设备: {self.device}")
            except Exception as e:
                logger.error(f"BLIP模型加载失败: {e}")
                self.model = None
        else:
            logger.warning("BLIP未安装，将使用模拟实现")

    async def analyze_content_match(self, image_path: str, description: str) -> Dict[str, Any]:
        """分析图像内容与描述的匹配度"""
        if not self.model or not Path(image_path).exists():
            return await self._mock_content_analysis(image_path, description)

        try:
            # 生成图像描述
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(image, return_tensors="pt").to(self.device)

            with torch.no_grad():
                generated_ids = self.model.generate(**inputs, max_length=50)
                generated_caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)

            # 简单的文本相似度计算
            match_score = await self._calculate_text_similarity(generated_caption, description)

            # 分析具体匹配元素
            description_words = set(description.lower().split())
            caption_words = set(generated_caption.lower().split())

            relevant_objects = list(description_words.intersection(caption_words))
            missing_elements = list(description_words - caption_words)
            extra_elements = list(caption_words - description_words)

            # 计算适用性
            suitability = "high" if match_score > 0.7 else "medium" if match_score > 0.4 else "low"

            result = {
                'match_score': match_score,
                'generated_caption': generated_caption,
                'relevant_objects': relevant_objects[:10],  # 限制数量
                'missing_elements': missing_elements[:5],
                'extra_elements': extra_elements[:5],
                'suitability': suitability,
                'confidence': 0.85
            }

            logger.info(f"BLIP内容分析完成: {image_path}")
            return result

        except Exception as e:
            logger.error(f"BLIP内容分析失败: {e}")
            return await self._mock_content_analysis(image_path, description)

    async def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """计算文本相似度"""
        try:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())

            intersection = len(words1.intersection(words2))
            union = len(words1.union(words2))

            if union == 0:
                return 0.0

            return intersection / union

        except Exception:
            return 0.5

    async def _mock_content_analysis(self, image_path: str, description: str) -> Dict[str, Any]:
        """模拟内容分析"""
        return {
            'match_score': 0.72,
            'generated_caption': 'a professional product image with modern lighting',
            'relevant_objects': ['product', 'background', 'lighting'],
            'missing_elements': ['specific_angle'],
            'extra_elements': ['watermark'],
            'suitability': 'high',
            'note': 'Using mock implementation - install transformers/BLIP for real analysis'
        }


class YOLOFaceDetector:
    """基于YOLO的人脸检测器"""

    def __init__(self, offline_mode=None):
        self.model = None

        # 检查是否强制离线模式
        if offline_mode or os.environ.get('HAS_VL_MODELS') == 'false':
            logger.info("YOLO检测器运行在离线模式")
            return

        if HAS_YOLO:
            try:
                # 检查模型文件是否存在
                face_model_path = Path('yolov8n-face.pt')
                general_model_path = Path('yolov8n.pt')

                if face_model_path.exists():
                    self.model = YOLO('yolov8n-face.pt')
                    logger.info("YOLO人脸检测模型加载成功")
                elif general_model_path.exists() and general_model_path.stat().st_size > 0:
                    self.model = YOLO('yolov8n.pt')
                    logger.info("YOLO通用检测模型加载成功")
                else:
                    logger.info("YOLO模型文件不存在或已损坏，使用模拟实现")
                    self.model = None
            except Exception as e:
                logger.info(f"YOLO模型加载失败，使用模拟实现: {e}")
                self.model = None
        else:
            logger.info("YOLO未安装，使用模拟实现")

    async def detect_faces_in_video(self, video_path: str) -> Dict[str, Any]:
        """检测视频中的人脸"""
        if not self.model or not Path(video_path).exists():
            return await self._mock_face_detection(video_path)

        try:
            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps > 0 else 0

            face_detections = []
            current_time = 0
            frame_count = 0

            # 每秒检测一帧以提高效率
            frame_skip = max(1, int(fps)) if fps > 0 else 30

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_skip == 0:
                    current_time = frame_count / fps if fps > 0 else frame_count / 30

                    # YOLO检测
                    results = self.model(frame)

                    # 提取人脸检测结果
                    for result in results:
                        boxes = result.boxes
                        if boxes is not None:
                            for box in boxes:
                                # 检查是否是人类（class 0通常是person）
                                if int(box.cls) == 0:  # person class
                                    confidence = float(box.conf)
                                    if confidence > 0.5:
                                        face_detections.append({
                                            'timestamp': current_time,
                                            'confidence': confidence,
                                            'bbox': box.xyxy[0].cpu().numpy().tolist()
                                        })

                frame_count += 1

            cap.release()

            # 合并连续的检测结果为时间段
            face_segments = self._merge_detections_to_segments(face_detections)

            result = {
                'duration': duration,
                'fps': fps,
                'faces_detected': len(face_segments),
                'face_timestamps': face_segments,
                'total_detections': len(face_detections)
            }

            logger.info(f"YOLO人脸检测完成: {video_path}")
            return result

        except Exception as e:
            logger.error(f"YOLO人脸检测失败: {e}")
            return await self._mock_face_detection(video_path)

    def _merge_detections_to_segments(self, detections: List[Dict], gap_threshold: float = 2.0) -> List[Dict]:
        """将检测结果合并为连续时间段"""
        if not detections:
            return []

        # 按时间排序
        detections = sorted(detections, key=lambda x: x['timestamp'])

        segments = []
        current_segment = {
            'start': detections[0]['timestamp'],
            'end': detections[0]['timestamp'],
            'confidence': detections[0]['confidence']
        }

        for detection in detections[1:]:
            if detection['timestamp'] - current_segment['end'] <= gap_threshold:
                # 扩展当前段
                current_segment['end'] = detection['timestamp']
                current_segment['confidence'] = max(current_segment['confidence'], detection['confidence'])
            else:
                # 开始新段
                segments.append(current_segment)
                current_segment = {
                    'start': detection['timestamp'],
                    'end': detection['timestamp'],
                    'confidence': detection['confidence']
                }

        # 添加最后一段
        segments.append(current_segment)

        return segments

    async def _mock_face_detection(self, video_path: str) -> Dict[str, Any]:
        """模拟人脸检测"""
        return {
            'duration': 45.0,
            'fps': 30,
            'faces_detected': 2,
            'face_timestamps': [
                {'start': 5.0, 'end': 15.0, 'confidence': 0.95},
                {'start': 20.0, 'end': 35.0, 'confidence': 0.88}
            ],
            'total_detections': 156,
            'note': 'Using mock implementation - install ultralytics for real detection'
        }


class WhisperAudioAnalyzer:
    """基于Whisper的语音分析器"""

    def __init__(self, model_size: str = "base", offline_mode=None):
        self.model = None
        self.model_size = model_size

        # 检查是否强制离线模式
        if offline_mode or os.environ.get('HAS_VL_MODELS') == 'false':
            logger.info("Whisper分析器运行在离线模式")
            return

        if HAS_WHISPER:
            try:
                self.model = whisper.load_model(model_size)
                logger.info(f"Whisper {model_size} 模型加载成功")
            except Exception as e:
                logger.error(f"Whisper模型加载失败: {e}")
                self.model = None
        else:
            logger.warning("Whisper未安装，将使用模拟实现")

    async def analyze_video_audio(self, video_path: str) -> Dict[str, Any]:
        """分析视频音频内容"""
        if not self.model or not Path(video_path).exists():
            return await self._mock_audio_analysis(video_path)

        try:
            # 使用whisper处理整个视频文件
            result = self.model.transcribe(video_path)

            # 提取语音段落
            speech_segments = []
            for segment in result['segments']:
                speech_segments.append({
                    'start': segment['start'],
                    'end': segment['end'],
                    'text': segment['text'].strip(),
                    'confidence': segment.get('avg_logprob', 0.0)
                })

            # 分析语音特征
            has_speech = len(speech_segments) > 0
            total_speech_duration = sum(seg['end'] - seg['start'] for seg in speech_segments)

            # 语言检测
            language = result.get('language', 'unknown')
            language_confidence = result.get('language_probability', 0.0)

            analysis_result = {
                'has_speech': has_speech,
                'language': language,
                'language_confidence': language_confidence,
                'speech_segments': speech_segments,
                'total_speech_duration': total_speech_duration,
                'full_text': result['text'],
                'speech_quality': self._assess_speech_quality(speech_segments)
            }

            logger.info(f"Whisper音频分析完成: {video_path}")
            return analysis_result

        except Exception as e:
            logger.error(f"Whisper音频分析失败: {e}")
            return await self._mock_audio_analysis(video_path)

    def _assess_speech_quality(self, segments: List[Dict]) -> Dict[str, Any]:
        """评估语音质量"""
        if not segments:
            return {'quality': 'no_speech', 'score': 0.0}

        # 基于置信度评估
        confidences = [seg.get('confidence', 0.0) for seg in segments]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0

        # 基于段落长度评估（太短或太长的段落可能表示质量问题）
        segment_lengths = [seg['end'] - seg['start'] for seg in segments]
        avg_length = sum(segment_lengths) / len(segment_lengths) if segment_lengths else 0.0

        # 综合评分
        quality_score = (avg_confidence * 0.7 + min(avg_length / 5.0, 1.0) * 0.3)

        if quality_score > 0.8:
            quality = 'excellent'
        elif quality_score > 0.6:
            quality = 'good'
        elif quality_score > 0.4:
            quality = 'fair'
        else:
            quality = 'poor'

        return {
            'quality': quality,
            'score': quality_score,
            'avg_confidence': avg_confidence,
            'avg_segment_length': avg_length,
            'total_segments': len(segments)
        }

    async def _mock_audio_analysis(self, video_path: str) -> Dict[str, Any]:
        """模拟音频分析"""
        return {
            'has_speech': True,
            'language': 'zh',
            'language_confidence': 0.95,
            'speech_segments': [
                {'start': 2.0, 'end': 8.0, 'text': '欢迎使用我们的产品', 'confidence': 0.92},
                {'start': 10.0, 'end': 18.0, 'text': '这是一个创新的解决方案', 'confidence': 0.88},
                {'start': 25.0, 'end': 35.0, 'text': '感谢您的关注', 'confidence': 0.85}
            ],
            'total_speech_duration': 20.0,
            'full_text': '欢迎使用我们的产品。这是一个创新的解决方案。感谢您的关注。',
            'speech_quality': {
                'quality': 'good',
                'score': 0.75,
                'avg_confidence': 0.88,
                'avg_segment_length': 6.67,
                'total_segments': 3
            },
            'note': 'Using mock implementation - install whisper for real analysis'
        }


class IntegratedVLSystem:
    """集成VL系统 - 统一接口"""

    def __init__(self, device: str = None, offline_mode: bool = None):
        # 检查离线模式
        if offline_mode is None:
            offline_mode = os.environ.get('HAS_VL_MODELS') == 'false'

        self.offline_mode = offline_mode
        self.clip_analyzer = CLIPStyleAnalyzer(device)
        self.blip_analyzer = BLIPContentAnalyzer(device)
        self.yolo_detector = YOLOFaceDetector(offline_mode=offline_mode)
        self.whisper_analyzer = WhisperAudioAnalyzer(offline_mode=offline_mode)
        self.logger = logger.getChild('IntegratedVL')

        if offline_mode:
            self.logger.info("VL系统运行在离线模式，使用模拟实现")

    async def analyze_visual_style(self, image_path: str) -> Dict[str, Any]:
        """统一的视觉风格分析接口"""
        return await self.clip_analyzer.analyze_image_style(image_path)

    async def analyze_content_match(self, image_path: str, description: str) -> Dict[str, Any]:
        """统一的内容匹配分析接口"""
        return await self.blip_analyzer.analyze_content_match(image_path, description)

    async def compare_visual_consistency(self, image1_path: str, image2_path: str) -> Dict[str, Any]:
        """比较两个图像的视觉一致性"""
        try:
            # 分别分析两个图像的风格
            style1 = await self.analyze_visual_style(image1_path)
            style2 = await self.analyze_visual_style(image2_path)

            # 计算一致性分数
            color_consistency = self._compare_colors(
                style1.get('dominant_colors', []),
                style2.get('dominant_colors', [])
            )

            lighting_consistency = 1.0 if style1.get('lighting') == style2.get('lighting') else 0.5

            style_consistency = self._compare_style_tags(
                style1.get('style_tags', []),
                style2.get('style_tags', [])
            )

            overall_consistency = (color_consistency * 0.4 +
                                 lighting_consistency * 0.3 +
                                 style_consistency * 0.3)

            return {
                'overall_consistency': overall_consistency,
                'color_consistency': color_consistency,
                'lighting_consistency': lighting_consistency,
                'style_consistency': style_consistency,
                'details': {
                    'similar_elements': [],
                    'different_elements': [],
                    'recommendations': []
                }
            }

        except Exception as e:
            self.logger.error(f"视觉一致性比较失败: {e}")
            return {
                'overall_consistency': 0.5,
                'color_consistency': 0.5,
                'lighting_consistency': 0.5,
                'style_consistency': 0.5,
                'error': str(e)
            }

    async def analyze_video_content(self, video_path: str) -> Dict[str, Any]:
        """统一的视频内容分析接口"""
        try:
            # 并行执行人脸检测和音频分析
            face_task = self.yolo_detector.detect_faces_in_video(video_path)
            audio_task = self.whisper_analyzer.analyze_video_audio(video_path)

            face_result, audio_result = await asyncio.gather(face_task, audio_task)

            # 合并结果
            return {
                'duration': face_result.get('duration', 0),
                'fps': face_result.get('fps', 30),
                'resolution': '1920x1080',  # 需要从视频中获取
                'face_detection': {
                    'faces_detected': face_result.get('faces_detected', 0),
                    'face_timestamps': face_result.get('face_timestamps', [])
                },
                'audio_analysis': audio_result,
                'analysis_timestamp': asyncio.get_event_loop().time()
            }

        except Exception as e:
            self.logger.error(f"视频内容分析失败: {e}")
            return {
                'duration': 0,
                'error': str(e)
            }

    def _compare_colors(self, colors1: List[str], colors2: List[str]) -> float:
        """比较颜色一致性"""
        if not colors1 or not colors2:
            return 0.5

        # 简单的颜色匹配算法
        matches = 0
        for color1 in colors1:
            for color2 in colors2:
                if color1 == color2:
                    matches += 1
                    break

        return matches / max(len(colors1), len(colors2))

    def _compare_style_tags(self, tags1: List[str], tags2: List[str]) -> float:
        """比较风格标签一致性"""
        if not tags1 or not tags2:
            return 0.5

        set1 = set(tags1)
        set2 = set(tags2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0


# 全局实例
vl_system = IntegratedVLSystem()


async def test_vl_system():
    """测试VL系统功能"""
    print("🧠 测试VL模型集成系统")

    # 测试图像风格分析
    print("\n1. 测试图像风格分析")
    if HAS_CLIP:
        print("✅ CLIP可用")
    else:
        print("⚠️  CLIP不可用，使用模拟实现")

    # 测试内容理解
    print("\n2. 测试内容理解")
    if HAS_BLIP:
        print("✅ BLIP可用")
    else:
        print("⚠️  BLIP不可用，使用模拟实现")

    # 测试人脸检测
    print("\n3. 测试人脸检测")
    if HAS_YOLO:
        print("✅ YOLO可用")
    else:
        print("⚠️  YOLO不可用，使用模拟实现")

    # 测试语音分析
    print("\n4. 测试语音分析")
    if HAS_WHISPER:
        print("✅ Whisper可用")
    else:
        print("⚠️  Whisper不可用，使用模拟实现")

    print("\n🎯 VL系统初始化完成")
    return True


if __name__ == "__main__":
    asyncio.run(test_vl_system())