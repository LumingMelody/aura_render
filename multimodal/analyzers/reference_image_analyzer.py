"""
Reference Image Analyzer
参考图片分析器 - 提取风格、情绪和产品特征
"""
import os
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
import asyncio
import json
from datetime import datetime
import math
import requests
import tempfile
from urllib.parse import urlparse

try:
    from PIL import Image, ImageStat, ImageFilter
    import numpy as np
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    Image = None

try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    cv2 = None

logger = logging.getLogger(__name__)


class ImageStyleAnalyzer:
    """图像风格指导分析器"""

    def __init__(self):
        self.logger = logger.getChild('ImageStyleAnalyzer')

    async def analyze(self, image_path: str) -> Dict[str, Any]:
        """分析图像风格特征"""
        if not PIL_AVAILABLE:
            self.logger.warning("PIL不可用，使用fallback分析")
            return await self._fallback_style_analysis(image_path)

        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._analyze_style_sync, image_path
            )
        except Exception as e:
            self.logger.error(f"图像风格分析失败: {e}")
            return await self._fallback_style_analysis(image_path)

    def _analyze_style_sync(self, image_path: str) -> Dict[str, Any]:
        """同步风格分析"""
        with Image.open(image_path) as img:
            # 转换为RGB模式
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 1. 色调分析
            color_analysis = self._analyze_colors(img)

            # 2. 构图分析
            composition_analysis = self._analyze_composition(img)

            # 3. 滤镜效果分析
            filter_analysis = self._analyze_filters(img)

            # 4. 整体风格评估
            style_score = self._calculate_style_score(color_analysis, composition_analysis)

            return {
                "color_palette": color_analysis,
                "composition": composition_analysis,
                "filter_effects": filter_analysis,
                "style_score": style_score,
                "analysis_method": "pil_analysis",
                "image_dimensions": img.size
            }

    def _analyze_colors(self, img: Image) -> Dict[str, Any]:
        """分析图像色调"""
        try:
            # 获取图像统计信息
            stat = ImageStat.Stat(img)

            # RGB平均值
            mean_colors = stat.mean

            # 计算色温
            color_temperature = self._estimate_color_temperature(mean_colors)

            # 计算饱和度
            hsv_img = img.convert('HSV')
            hsv_stat = ImageStat.Stat(hsv_img)
            saturation = hsv_stat.mean[1] / 255.0

            # 计算亮度
            brightness = sum(mean_colors) / (3 * 255.0)

            # 主导色分析
            dominant_colors = self._extract_dominant_colors(img)

            return {
                "mean_rgb": mean_colors,
                "color_temperature": color_temperature,
                "saturation": saturation,
                "brightness": brightness,
                "dominant_colors": dominant_colors,
                "color_harmony": self._assess_color_harmony(dominant_colors)
            }
        except Exception as e:
            self.logger.error(f"色调分析失败: {e}")
            return {"error": str(e)}

    def _analyze_composition(self, img: Image) -> Dict[str, Any]:
        """分析图像构图"""
        try:
            width, height = img.size
            aspect_ratio = width / height

            # 构图类型判断
            if abs(aspect_ratio - 1.0) < 0.1:
                composition_type = "square"
            elif aspect_ratio > 1.5:
                composition_type = "landscape"
            elif aspect_ratio < 0.7:
                composition_type = "portrait"
            else:
                composition_type = "standard"

            # 简化的焦点分析（基于中心区域亮度）
            center_region = self._get_center_region_stats(img)

            # 对称性分析
            symmetry_score = self._analyze_symmetry(img)

            return {
                "aspect_ratio": aspect_ratio,
                "composition_type": composition_type,
                "center_focus": center_region,
                "symmetry_score": symmetry_score,
                "image_size": img.size
            }
        except Exception as e:
            self.logger.error(f"构图分析失败: {e}")
            return {"error": str(e)}

    def _analyze_filters(self, img: Image) -> Dict[str, Any]:
        """分析滤镜效果"""
        try:
            # 检测可能的滤镜效果
            filters_detected = []

            # 检测模糊效果
            blur_level = self._detect_blur_level(img)
            if blur_level > 0.3:
                filters_detected.append("blur")

            # 检测锐化效果
            sharpness_level = self._detect_sharpness_level(img)
            if sharpness_level > 0.7:
                filters_detected.append("sharpen")

            # 检测对比度调整
            contrast_level = self._detect_contrast_level(img)
            if contrast_level > 0.8:
                filters_detected.append("high_contrast")
            elif contrast_level < 0.3:
                filters_detected.append("low_contrast")

            return {
                "detected_filters": filters_detected,
                "blur_level": blur_level,
                "sharpness_level": sharpness_level,
                "contrast_level": contrast_level,
                "filter_strength": len(filters_detected) / 4.0  # 标准化
            }
        except Exception as e:
            self.logger.error(f"滤镜分析失败: {e}")
            return {"error": str(e)}

    def _estimate_color_temperature(self, rgb: List[float]) -> str:
        """估算色温"""
        r, g, b = rgb
        if b > r * 1.1:
            return "cool"
        elif r > b * 1.1:
            return "warm"
        else:
            return "neutral"

    def _extract_dominant_colors(self, img: Image, num_colors: int = 5) -> List[Tuple[int, int, int]]:
        """提取主导色"""
        try:
            # 缩小图像以提高性能
            img_small = img.resize((100, 100))

            # 转换为numpy数组
            pixels = np.array(img_small)
            pixels = pixels.reshape(-1, 3)

            # 简化的K-means聚类（使用固定采样点）
            sample_size = min(1000, len(pixels))
            sample_indices = np.random.choice(len(pixels), sample_size, replace=False)
            sample_pixels = pixels[sample_indices]

            # 简化的颜色聚类
            dominant_colors = []
            for i in range(min(num_colors, len(sample_pixels) // 100)):
                start_idx = i * (len(sample_pixels) // num_colors)
                end_idx = start_idx + (len(sample_pixels) // num_colors)
                color_group = sample_pixels[start_idx:end_idx]
                avg_color = np.mean(color_group, axis=0)
                dominant_colors.append(tuple(map(int, avg_color)))

            return dominant_colors[:num_colors]
        except Exception as e:
            self.logger.error(f"主导色提取失败: {e}")
            return [(128, 128, 128)]  # 默认灰色

    def _assess_color_harmony(self, colors: List[Tuple[int, int, int]]) -> str:
        """评估色彩和谐度"""
        if len(colors) < 2:
            return "monochromatic"

        # 计算颜色之间的距离
        color_distances = []
        for i in range(len(colors)):
            for j in range(i + 1, len(colors)):
                distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(colors[i], colors[j])))
                color_distances.append(distance)

        avg_distance = sum(color_distances) / len(color_distances)

        if avg_distance < 50:
            return "analogous"
        elif avg_distance > 150:
            return "complementary"
        else:
            return "triadic"

    def _get_center_region_stats(self, img: Image) -> Dict[str, Any]:
        """获取中心区域统计信息"""
        try:
            width, height = img.size
            center_x, center_y = width // 2, height // 2
            region_size = min(width, height) // 4

            # 裁剪中心区域
            left = max(0, center_x - region_size)
            top = max(0, center_y - region_size)
            right = min(width, center_x + region_size)
            bottom = min(height, center_y + region_size)

            center_region = img.crop((left, top, right, bottom))
            stat = ImageStat.Stat(center_region)

            return {
                "mean_brightness": sum(stat.mean) / (3 * 255.0),
                "region_size": (right - left, bottom - top)
            }
        except Exception as e:
            return {"error": str(e)}

    def _analyze_symmetry(self, img: Image) -> float:
        """分析对称性（简化实现）"""
        try:
            # 转换为灰度
            gray = img.convert('L')
            width, height = gray.size

            # 垂直对称性检查
            left_half = gray.crop((0, 0, width // 2, height))
            right_half = gray.crop((width // 2, 0, width, height))
            right_half_flipped = right_half.transpose(Image.FLIP_LEFT_RIGHT)

            # 计算相似度（简化）
            left_stat = ImageStat.Stat(left_half)
            right_stat = ImageStat.Stat(right_half_flipped)

            similarity = 1.0 - abs(left_stat.mean[0] - right_stat.mean[0]) / 255.0
            return max(0.0, similarity)
        except Exception as e:
            return 0.5  # 默认值

    def _detect_blur_level(self, img: Image) -> float:
        """检测模糊程度"""
        try:
            # 简化的边缘检测
            gray = img.convert('L')
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges)

            # 边缘强度越低，模糊程度越高
            blur_level = 1.0 - (edge_stat.mean[0] / 255.0)
            return max(0.0, min(1.0, blur_level))
        except:
            return 0.0

    def _detect_sharpness_level(self, img: Image) -> float:
        """检测锐化程度"""
        try:
            # 应用锐化滤镜并比较
            sharpened = img.filter(ImageFilter.SHARPEN)

            # 简化的锐化检测
            original_stat = ImageStat.Stat(img.convert('L'))
            sharpened_stat = ImageStat.Stat(sharpened.convert('L'))

            sharpness = abs(sharpened_stat.stddev[0] - original_stat.stddev[0]) / 255.0
            return max(0.0, min(1.0, sharpness))
        except:
            return 0.5

    def _detect_contrast_level(self, img: Image) -> float:
        """检测对比度水平"""
        try:
            stat = ImageStat.Stat(img.convert('L'))
            # 使用标准差作为对比度指标
            contrast = stat.stddev[0] / 128.0  # 标准化
            return max(0.0, min(1.0, contrast))
        except:
            return 0.5

    def _calculate_style_score(self, color_analysis: Dict, composition_analysis: Dict) -> float:
        """计算整体风格评分"""
        try:
            color_score = color_analysis.get("saturation", 0.5)
            composition_score = composition_analysis.get("symmetry_score", 0.5)

            overall_score = (color_score + composition_score) / 2
            return max(0.0, min(1.0, overall_score))
        except:
            return 0.7

    async def _fallback_style_analysis(self, image_path: str) -> Dict[str, Any]:
        """Fallback风格分析"""
        file_size = os.path.getsize(image_path)
        filename = os.path.basename(image_path).lower()

        # 基于文件名的启发式分析
        if "bright" in filename or "light" in filename:
            brightness = "high"
        elif "dark" in filename or "night" in filename:
            brightness = "low"
        else:
            brightness = "medium"

        return {
            "brightness_estimate": brightness,
            "file_size_kb": file_size / 1024,
            "analysis_method": "fallback_heuristic",
            "confidence": 0.4
        }


class MoodBoardExtractor:
    """情绪板解析器"""

    def __init__(self):
        self.logger = logger.getChild('MoodBoardExtractor')

        # 情绪色彩映射
        self.mood_color_mapping = {
            "energetic": [(255, 100, 100), (255, 200, 0), (100, 255, 100)],
            "calm": [(100, 150, 255), (150, 200, 255), (200, 255, 200)],
            "elegant": [(50, 50, 50), (200, 200, 200), (100, 100, 150)],
            "warm": [(255, 150, 100), (255, 200, 150), (200, 150, 100)],
            "cool": [(100, 200, 255), (150, 255, 200), (200, 200, 255)]
        }

    async def extract_mood(self, image_path: str) -> Dict[str, Any]:
        """提取情绪特征"""
        if not PIL_AVAILABLE:
            return await self._fallback_mood_analysis(image_path)

        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._extract_mood_sync, image_path
            )
        except Exception as e:
            self.logger.error(f"情绪提取失败: {e}")
            return await self._fallback_mood_analysis(image_path)

    def _extract_mood_sync(self, image_path: str) -> Dict[str, Any]:
        """同步情绪提取"""
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 1. 情感色彩分析
            emotional_colors = self._analyze_emotional_colors(img)

            # 2. 氛围营造分析
            atmosphere = self._analyze_atmosphere(img)

            # 3. 情绪强度评估
            mood_intensity = self._calculate_mood_intensity(emotional_colors, atmosphere)

            return {
                "emotional_colors": emotional_colors,
                "atmosphere": atmosphere,
                "mood_intensity": mood_intensity,
                "primary_mood": self._determine_primary_mood(emotional_colors),
                "analysis_method": "mood_extraction"
            }

    def _analyze_emotional_colors(self, img: Image) -> Dict[str, Any]:
        """分析情感色彩"""
        try:
            stat = ImageStat.Stat(img)
            mean_colors = stat.mean

            # 转换为HSV获取饱和度和亮度信息
            hsv_img = img.convert('HSV')
            hsv_stat = ImageStat.Stat(hsv_img)

            hue = hsv_stat.mean[0]
            saturation = hsv_stat.mean[1] / 255.0
            value = hsv_stat.mean[2] / 255.0

            # 情感色彩分类
            if hue < 60:  # 红色-黄色
                emotional_category = "warm_energetic"
            elif hue < 180:  # 黄色-青色
                emotional_category = "cool_calm"
            elif hue < 300:  # 青色-紫色
                emotional_category = "cool_mysterious"
            else:  # 紫色-红色
                emotional_category = "warm_passionate"

            return {
                "hue": hue,
                "saturation": saturation,
                "value": value,
                "emotional_category": emotional_category,
                "warmth_level": self._calculate_warmth(mean_colors)
            }
        except Exception as e:
            return {"error": str(e)}

    def _analyze_atmosphere(self, img: Image) -> Dict[str, Any]:
        """分析氛围营造"""
        try:
            # 亮度分布分析
            gray = img.convert('L')
            histogram = gray.histogram()

            # 计算亮度分布特征
            dark_pixels = sum(histogram[:85])  # 0-85 dark
            mid_pixels = sum(histogram[85:170])  # 85-170 mid
            bright_pixels = sum(histogram[170:256])  # 170-255 bright

            total_pixels = sum(histogram)

            dark_ratio = dark_pixels / total_pixels
            bright_ratio = bright_pixels / total_pixels

            # 氛围分类
            if dark_ratio > 0.6:
                atmosphere_type = "mysterious"
            elif bright_ratio > 0.6:
                atmosphere_type = "cheerful"
            elif dark_ratio > 0.4 and bright_ratio < 0.3:
                atmosphere_type = "moody"
            else:
                atmosphere_type = "balanced"

            return {
                "dark_ratio": dark_ratio,
                "bright_ratio": bright_ratio,
                "atmosphere_type": atmosphere_type,
                "contrast_level": abs(dark_ratio - bright_ratio)
            }
        except Exception as e:
            return {"error": str(e)}

    def _calculate_warmth(self, rgb_colors: List[float]) -> float:
        """计算色彩温暖度"""
        r, g, b = rgb_colors
        # 红色和黄色成分高表示温暖
        warmth = (r + g/2 - b) / 255.0
        return max(0.0, min(1.0, warmth))

    def _calculate_mood_intensity(self, emotional_colors: Dict, atmosphere: Dict) -> float:
        """计算情绪强度"""
        try:
            saturation = emotional_colors.get("saturation", 0.5)
            contrast = atmosphere.get("contrast_level", 0.5)

            intensity = (saturation + contrast) / 2
            return max(0.0, min(1.0, intensity))
        except:
            return 0.5

    def _determine_primary_mood(self, emotional_colors: Dict) -> str:
        """确定主要情绪"""
        try:
            category = emotional_colors.get("emotional_category", "balanced")
            saturation = emotional_colors.get("saturation", 0.5)
            value = emotional_colors.get("value", 0.5)

            if category == "warm_energetic" and saturation > 0.7:
                return "energetic"
            elif category == "cool_calm" and value > 0.6:
                return "calm"
            elif category == "cool_mysterious" and value < 0.4:
                return "mysterious"
            elif category == "warm_passionate" and saturation > 0.6:
                return "passionate"
            else:
                return "neutral"
        except:
            return "neutral"

    async def _fallback_mood_analysis(self, image_path: str) -> Dict[str, Any]:
        """Fallback情绪分析"""
        filename = os.path.basename(image_path).lower()

        # 基于文件名的情绪推断
        if any(word in filename for word in ["happy", "bright", "joy"]):
            mood = "energetic"
        elif any(word in filename for word in ["calm", "peace", "soft"]):
            mood = "calm"
        elif any(word in filename for word in ["dark", "mystery", "deep"]):
            mood = "mysterious"
        else:
            mood = "neutral"

        return {
            "primary_mood": mood,
            "analysis_method": "fallback_filename",
            "confidence": 0.3
        }


class ProductImageDetector:
    """产品图智能识别器"""

    def __init__(self):
        self.logger = logger.getChild('ProductImageDetector')

    async def detect_and_analyze(self, image_path: str) -> Dict[str, Any]:
        """检测并分析产品图"""
        if not PIL_AVAILABLE:
            return await self._fallback_product_analysis(image_path)

        try:
            return await asyncio.get_event_loop().run_in_executor(
                None, self._detect_product_sync, image_path
            )
        except Exception as e:
            self.logger.error(f"产品检测失败: {e}")
            return await self._fallback_product_analysis(image_path)

    def _detect_product_sync(self, image_path: str) -> Dict[str, Any]:
        """同步产品检测"""
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # 1. 主体检测
            subject_detection = self._detect_main_subject(img)

            # 2. 背景分析
            background_analysis = self._analyze_background(img)

            # 3. 产品类型推断
            product_type = self._infer_product_type(img, subject_detection)

            # 4. 质量评估
            quality_assessment = self._assess_image_quality(img)

            return {
                "subject_detection": subject_detection,
                "background_analysis": background_analysis,
                "product_type": product_type,
                "quality_assessment": quality_assessment,
                "analysis_method": "product_detection"
            }

    def _detect_main_subject(self, img: Image) -> Dict[str, Any]:
        """检测主要主体"""
        try:
            width, height = img.size

            # 中心区域分析
            center_region = self._get_center_analysis(img)

            # 边缘检测（简化）
            edges = img.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges.convert('L'))

            # 主体位置推断
            subject_position = self._estimate_subject_position(img)

            return {
                "center_focus": center_region,
                "edge_density": edge_stat.mean[0] / 255.0,
                "estimated_position": subject_position,
                "subject_confidence": center_region.get("focus_strength", 0.5)
            }
        except Exception as e:
            return {"error": str(e)}

    def _analyze_background(self, img: Image) -> Dict[str, Any]:
        """分析背景特征"""
        try:
            # 边缘区域分析
            width, height = img.size
            border_width = min(width, height) // 10

            # 提取边缘区域
            top_border = img.crop((0, 0, width, border_width))
            bottom_border = img.crop((0, height - border_width, width, height))
            left_border = img.crop((0, 0, border_width, height))
            right_border = img.crop((width - border_width, 0, width, height))

            # 分析边缘一致性
            borders = [top_border, bottom_border, left_border, right_border]
            border_stats = [ImageStat.Stat(border) for border in borders]

            # 计算背景一致性
            consistency = self._calculate_background_consistency(border_stats)

            # 背景类型判断
            if consistency > 0.8:
                bg_type = "solid"
            elif consistency > 0.5:
                bg_type = "gradient"
            else:
                bg_type = "complex"

            return {
                "background_type": bg_type,
                "consistency_score": consistency,
                "is_clean_background": consistency > 0.7
            }
        except Exception as e:
            return {"error": str(e)}

    def _infer_product_type(self, img: Image, subject_info: Dict) -> Dict[str, Any]:
        """推断产品类型"""
        try:
            width, height = img.size
            aspect_ratio = width / height

            # 基于宽高比的产品类型推断
            if abs(aspect_ratio - 1.0) < 0.2:
                likely_type = "square_product"  # 可能是包装、盒子等
            elif aspect_ratio > 1.5:
                likely_type = "horizontal_product"  # 可能是电子产品、工具等
            elif aspect_ratio < 0.7:
                likely_type = "vertical_product"  # 可能是瓶装、书籍等
            else:
                likely_type = "standard_product"

            # 基于颜色特征的进一步推断
            stat = ImageStat.Stat(img)
            mean_brightness = sum(stat.mean) / (3 * 255.0)

            if mean_brightness > 0.8:
                product_style = "clean_minimal"
            elif mean_brightness < 0.3:
                product_style = "dark_premium"
            else:
                product_style = "balanced"

            return {
                "likely_type": likely_type,
                "product_style": product_style,
                "aspect_ratio": aspect_ratio,
                "confidence": subject_info.get("subject_confidence", 0.5)
            }
        except Exception as e:
            return {"error": str(e)}

    def _assess_image_quality(self, img: Image) -> Dict[str, Any]:
        """评估图像质量"""
        try:
            width, height = img.size
            resolution_score = min(1.0, (width * height) / (1920 * 1080))

            # 锐度评估
            sharpness = self._calculate_sharpness(img)

            # 噪点评估（简化）
            noise_level = self._estimate_noise_level(img)

            # 整体质量评分
            quality_score = (resolution_score + sharpness + (1 - noise_level)) / 3

            return {
                "resolution_score": resolution_score,
                "sharpness_score": sharpness,
                "noise_level": noise_level,
                "overall_quality": quality_score,
                "image_dimensions": (width, height)
            }
        except Exception as e:
            return {"error": str(e)}

    def _get_center_analysis(self, img: Image) -> Dict[str, Any]:
        """获取中心区域分析"""
        try:
            width, height = img.size
            center_size = min(width, height) // 3

            center_x, center_y = width // 2, height // 2
            left = center_x - center_size // 2
            top = center_y - center_size // 2
            right = center_x + center_size // 2
            bottom = center_y + center_size // 2

            center_region = img.crop((left, top, right, bottom))
            center_stat = ImageStat.Stat(center_region)

            # 计算中心焦点强度
            center_brightness = sum(center_stat.mean) / (3 * 255.0)
            focus_strength = min(1.0, center_brightness * 1.2)

            return {
                "center_brightness": center_brightness,
                "focus_strength": focus_strength,
                "region_size": (right - left, bottom - top)
            }
        except Exception as e:
            return {"error": str(e)}

    def _estimate_subject_position(self, img: Image) -> str:
        """估算主体位置"""
        # 简化的位置估算
        width, height = img.size

        # 分析不同区域的活跃度
        regions = {
            "center": img.crop((width//4, height//4, 3*width//4, 3*height//4)),
            "left": img.crop((0, height//4, width//2, 3*height//4)),
            "right": img.crop((width//2, height//4, width, 3*height//4)),
        }

        max_activity = 0
        best_position = "center"

        for position, region in regions.items():
            stat = ImageStat.Stat(region.convert('L'))
            activity = stat.stddev[0]  # 使用标准差作为活跃度指标

            if activity > max_activity:
                max_activity = activity
                best_position = position

        return best_position

    def _calculate_background_consistency(self, border_stats: List) -> float:
        """计算背景一致性"""
        try:
            # 计算各边框的亮度差异
            brightnesses = []
            for stat in border_stats:
                brightness = sum(stat.mean) / (3 * 255.0)
                brightnesses.append(brightness)

            # 计算标准差，标准差越小越一致
            if len(brightnesses) < 2:
                return 0.5

            mean_brightness = sum(brightnesses) / len(brightnesses)
            variance = sum((b - mean_brightness) ** 2 for b in brightnesses) / len(brightnesses)
            std_dev = math.sqrt(variance)

            # 转换为一致性评分（0-1）
            consistency = max(0.0, 1.0 - std_dev * 2)
            return consistency
        except:
            return 0.5

    def _calculate_sharpness(self, img: Image) -> float:
        """计算图像锐度"""
        try:
            # 使用边缘检测评估锐度
            gray = img.convert('L')
            edges = gray.filter(ImageFilter.FIND_EDGES)
            edge_stat = ImageStat.Stat(edges)

            sharpness = edge_stat.mean[0] / 255.0
            return max(0.0, min(1.0, sharpness))
        except:
            return 0.5

    def _estimate_noise_level(self, img: Image) -> float:
        """估算噪点水平"""
        try:
            # 简化的噪点估算
            small_img = img.resize((100, 100))
            blurred = small_img.filter(ImageFilter.BLUR)

            # 计算原图和模糊图的差异
            original_stat = ImageStat.Stat(small_img)
            blurred_stat = ImageStat.Stat(blurred)

            noise = abs(original_stat.stddev[0] - blurred_stat.stddev[0]) / 255.0
            return max(0.0, min(1.0, noise))
        except:
            return 0.3

    async def _fallback_product_analysis(self, image_path: str) -> Dict[str, Any]:
        """Fallback产品分析"""
        filename = os.path.basename(image_path).lower()
        file_size = os.path.getsize(image_path)

        # 基于文件名的产品类型推断
        if any(word in filename for word in ["product", "item", "goods"]):
            product_type = "general_product"
        elif any(word in filename for word in ["phone", "laptop", "device"]):
            product_type = "electronics"
        elif any(word in filename for word in ["bottle", "package", "box"]):
            product_type = "packaged_goods"
        else:
            product_type = "unknown"

        return {
            "product_type": product_type,
            "file_size_mb": file_size / (1024 * 1024),
            "analysis_method": "fallback_heuristic",
            "confidence": 0.3
        }


class ReferenceImageAnalyzer:
    """参考图片分析器主类"""

    def __init__(self):
        self.logger = logger.getChild('ReferenceImageAnalyzer')
        self.style_analyzer = ImageStyleAnalyzer()
        self.mood_extractor = MoodBoardExtractor()
        self.product_detector = ProductImageDetector()

    def _download_image_from_url(self, url: str) -> str:
        """
        从URL下载图片到临时文件

        Args:
            url: 图片URL

        Returns:
            下载的临时文件路径
        """
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()

            # 获取文件扩展名
            parsed_url = urlparse(url)
            file_extension = os.path.splitext(parsed_url.path)[1] or '.jpg'

            # 创建临时文件
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp_file:
                tmp_file.write(response.content)
                temp_path = tmp_file.name

            self.logger.info(f"图片已下载到临时文件: {temp_path}")
            return temp_path

        except Exception as e:
            self.logger.error(f"下载图片失败: {url}, 错误: {e}")
            raise

    async def analyze_reference_image(
        self,
        image_path: str,
        image_type: str,
        weight: float = 1.0
    ) -> Dict[str, Any]:
        """
        分析参考图片

        Args:
            image_path: 图片文件路径或URL
            image_type: 分析类型 ('style_guide', 'mood_board', 'main_product', 'detail_shot', etc.)
            weight: 分析权重 (0.0-1.0)

        Returns:
            分析结果字典
        """
        # 判断是URL还是本地文件路径
        is_url = image_path.startswith('http://') or image_path.startswith('https://')
        temp_file_path = None

        if is_url:
            # 下载图片到临时文件
            image_path = self._download_image_from_url(image_path)
            temp_file_path = image_path
        elif not os.path.exists(image_path):
            raise FileNotFoundError(f"图片文件不存在: {image_path}")

        self.logger.info(f"开始分析参考图片: {image_path} (类型: {image_type}, 权重: {weight})")

        start_time = datetime.now()

        try:
            analysis_result = {}

            if image_type == "style_guide":
                analysis_result = await self.style_analyzer.analyze(image_path)
            elif image_type == "mood_board":
                analysis_result = await self.mood_extractor.extract_mood(image_path)
            elif image_type in ["main_product", "detail_shot"]:
                analysis_result = await self.product_detector.detect_and_analyze(image_path)
            else:
                # 综合分析
                style_task = self.style_analyzer.analyze(image_path)
                mood_task = self.mood_extractor.extract_mood(image_path)

                style_result, mood_result = await asyncio.gather(
                    style_task, mood_task, return_exceptions=True
                )

                analysis_result = {
                    "style_analysis": style_result if not isinstance(style_result, Exception) else {"error": str(style_result)},
                    "mood_analysis": mood_result if not isinstance(mood_result, Exception) else {"error": str(mood_result)}
                }

            analysis_time = (datetime.now() - start_time).total_seconds()

            result = {
                "image_path": image_path,
                "analysis_type": image_type,
                "analysis_weight": weight,
                "analysis_result": analysis_result,
                "analysis_metadata": {
                    "analysis_time": analysis_time,
                    "timestamp": datetime.now().isoformat(),
                    "pil_available": PIL_AVAILABLE,
                    "opencv_available": OPENCV_AVAILABLE,
                    "success": True
                }
            }

            self.logger.info(f"图片分析完成: {image_path} (耗时: {analysis_time:.2f}s)")
            return result

        except Exception as e:
            self.logger.error(f"图片分析失败: {e}")
            return {
                "image_path": image_path,
                "analysis_type": image_type,
                "analysis_weight": weight,
                "error": str(e),
                "analysis_metadata": {
                    "analysis_time": (datetime.now() - start_time).total_seconds(),
                    "timestamp": datetime.now().isoformat(),
                    "pil_available": PIL_AVAILABLE,
                    "opencv_available": OPENCV_AVAILABLE,
                    "success": False
                }
            }
        finally:
            # 清理临时文件
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.unlink(temp_file_path)
                    self.logger.info(f"已清理临时文件: {temp_file_path}")
                except Exception as e:
                    self.logger.warning(f"清理临时文件失败: {temp_file_path}, 错误: {e}")

    async def batch_analyze_images(
        self,
        image_references: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        批量分析多个参考图片

        Args:
            image_references: 图片引用列表，格式为 [{"url": "path", "type": "style_guide", "weight": 0.8}, ...]

        Returns:
            分析结果列表
        """
        self.logger.info(f"开始批量分析 {len(image_references)} 个参考图片")

        tasks = [
            self.analyze_reference_image(
                image_ref["url"],
                image_ref.get("type", "style_guide"),
                image_ref.get("weight", 1.0)
            )
            for image_ref in image_references
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"图片 {i} 分析失败: {result}")
                processed_results.append({
                    "error": str(result),
                    "image_index": i,
                    "success": False
                })
            else:
                processed_results.append(result)

        self.logger.info(f"批量分析完成: {len(processed_results)} 个结果")
        return processed_results


# 用于测试的主函数
async def main():
    """测试ReferenceImageAnalyzer"""
    analyzer = ReferenceImageAnalyzer()

    # 测试单个图片分析
    test_image = "/path/to/test/image.jpg"  # 替换为实际测试图片路径

    if os.path.exists(test_image):
        result = await analyzer.analyze_reference_image(
            test_image,
            "style_guide",
            0.8
        )
        print("分析结果:")
        print(json.dumps(result, indent=2, ensure_ascii=False))
    else:
        print(f"测试图片文件不存在: {test_image}")
        print("请提供有效的图片文件路径进行测试")


if __name__ == "__main__":
    asyncio.run(main())