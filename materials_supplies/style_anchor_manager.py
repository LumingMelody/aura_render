"""
风格锚点管理器 - 全局风格一致性控制核心
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
from enum import Enum
import asyncio
from concurrent.futures import ThreadPoolExecutor

from llm.qwen import QwenLLM


class StyleType(Enum):
    """视觉风格类型"""
    CINEMATIC = "cinematic"  # 电影感，暗调，镜头光晕
    REALISTIC = "realistic"  # 写实，自然光，真实场景
    ANIME = "anime"  # 二次元，卡通，夸张表情
    DOCUMENTARY = "documentary"  # 纪实，手持感，低饱和
    ADVERTISEMENT = "advertisement"  # 广告风，高光，产品聚焦
    CYBERPUNK = "cyberpunk"  # 赛博朋克，霓虹灯，科技感
    WATERCOLOR = "watercolor"  # 水彩，手绘，柔和边缘


@dataclass
class StyleVector:
    """风格向量化表示"""
    style_type: StyleType
    color_palette: List[str]  # 主色调
    saturation: float  # 饱和度 0-1
    brightness: float  # 明度 0-1
    contrast: float  # 对比度 0-1
    texture_complexity: float  # 纹理复杂度 0-1
    motion_intensity: float  # 动态强度 0-1
    camera_stability: float  # 镜头稳定性 0-1

    def to_vector(self) -> np.ndarray:
        """转换为可计算的向量"""
        return np.array([
            self.saturation,
            self.brightness,
            self.contrast,
            self.texture_complexity,
            self.motion_intensity,
            self.camera_stability
        ])

    def distance(self, other: 'StyleVector') -> float:
        """计算风格距离"""
        if self.style_type != other.style_type:
            base_distance = 0.5  # 不同类型的基础距离
        else:
            base_distance = 0.0

        # 计算向量距离
        vector_distance = np.linalg.norm(self.to_vector() - other.to_vector())
        return base_distance + vector_distance * 0.5


@dataclass
class GlobalElements:
    """全局统一元素"""
    main_character: Optional[Dict[str, Any]] = None  # 主角形象
    brand_logo: Optional[str] = None  # 品牌标识
    product_reference: Optional[Dict[str, Any]] = None  # 产品参考
    color_scheme: Optional[List[str]] = None  # 配色方案
    font_style: Optional[str] = None  # 字体风格


class StyleAnchorManager:
    """风格锚点管理器"""

    def __init__(self):
        self.qwen = QwenLLM()
        self.style_anchor: Optional[StyleVector] = None
        self.global_elements = GlobalElements()
        self.user_reference_materials: List[Dict] = []
        self.style_compatibility_threshold = 0.3  # 风格兼容性阈值

    async def establish_style_anchor(self, shots: List[Dict]) -> StyleVector:
        """
        建立全局风格锚点
        优先级：用户素材风格 > 素材库主流风格 > AI推断风格
        """
        # Step 1: 检查用户提供的素材
        user_materials = self._extract_user_materials(shots)
        if user_materials:
            style = await self._analyze_user_material_style(user_materials)
            if style:
                self.style_anchor = style
                return style

        # Step 2: 分析素材库主流风格
        library_styles = await self._analyze_library_tendency(shots)
        if library_styles:
            style = self._select_dominant_style(library_styles)
            if style:
                self.style_anchor = style
                return style

        # Step 3: AI推断风格
        style = await self._infer_style_from_descriptions(shots)
        self.style_anchor = style
        return style

    def _extract_user_materials(self, shots: List[Dict]) -> List[Dict]:
        """提取用户上传的素材"""
        user_materials = []
        for shot in shots:
            if shot.get("asset_status") == "matched":
                asset = shot.get("scheduled_asset", {})
                if asset.get("source") == "user_upload":
                    user_materials.append({
                        "shot": shot,
                        "asset": asset,
                        "thumbnail": asset.get("thumbnail"),
                        "url": asset.get("url")
                    })
                    self.user_reference_materials.append(asset)
        return user_materials

    async def _analyze_user_material_style(self, materials: List[Dict]) -> Optional[StyleVector]:
        """
        分析用户素材的视觉风格，提取视觉DNA
        """
        if not materials:
            return None

        # 使用VL模型分析每个素材
        style_vectors = []
        for material in materials:
            thumbnail = material.get("thumbnail")
            if not thumbnail:
                continue

            # VL分析提取详细特征
            features = await self._extract_visual_features_vl(thumbnail)
            if features:
                style_vectors.append(features)

        if not style_vectors:
            return None

        # 聚合多个素材的风格特征
        return self._aggregate_style_vectors(style_vectors)

    async def _extract_visual_features_vl(self, image_url: str) -> Optional[StyleVector]:
        """
        使用VL模型提取视觉特征
        """
        prompt = """
        请详细分析这张图片的视觉风格特征，包括：
        1. 风格类型（cinematic/realistic/anime/documentary/advertisement/cyberpunk/watercolor）
        2. 主色调（列出3-5个主要颜色的hex值）
        3. 饱和度（0-1）
        4. 明度（0-1）
        5. 对比度（0-1）
        6. 纹理复杂度（0-1）
        7. 动态感（0-1）
        8. 镜头稳定性（0-1）

        请以JSON格式输出。
        """

        try:
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=2)
            response = await loop.run_in_executor(
                executor,
                lambda: self.qwen.generate(
                    prompt=prompt,
                    images=[image_url],
                    max_retries=3
                )
            )

            if response:
                # 解析返回的特征
                import json
                features = json.loads(response)

                return StyleVector(
                    style_type=StyleType(features.get("style_type", "realistic")),
                    color_palette=features.get("color_palette", []),
                    saturation=features.get("saturation", 0.5),
                    brightness=features.get("brightness", 0.5),
                    contrast=features.get("contrast", 0.5),
                    texture_complexity=features.get("texture_complexity", 0.5),
                    motion_intensity=features.get("motion_intensity", 0.3),
                    camera_stability=features.get("camera_stability", 0.7)
                )
        except Exception as e:
            print(f"[StyleAnchor] VL分析失败: {e}")
            return None

    def _aggregate_style_vectors(self, vectors: List[StyleVector]) -> StyleVector:
        """聚合多个风格向量，找出主导风格"""
        if not vectors:
            return None

        # 找出最常见的风格类型
        style_types = [v.style_type for v in vectors]
        from collections import Counter
        most_common_type = Counter(style_types).most_common(1)[0][0]

        # 计算平均特征值
        avg_saturation = np.mean([v.saturation for v in vectors])
        avg_brightness = np.mean([v.brightness for v in vectors])
        avg_contrast = np.mean([v.contrast for v in vectors])
        avg_texture = np.mean([v.texture_complexity for v in vectors])
        avg_motion = np.mean([v.motion_intensity for v in vectors])
        avg_stability = np.mean([v.camera_stability for v in vectors])

        # 合并颜色调色板
        all_colors = []
        for v in vectors:
            all_colors.extend(v.color_palette)
        # 取最常见的5个颜色
        color_palette = list(Counter(all_colors).most_common(5))
        color_palette = [c[0] for c in color_palette]

        return StyleVector(
            style_type=most_common_type,
            color_palette=color_palette,
            saturation=float(avg_saturation),
            brightness=float(avg_brightness),
            contrast=float(avg_contrast),
            texture_complexity=float(avg_texture),
            motion_intensity=float(avg_motion),
            camera_stability=float(avg_stability)
        )

    async def _analyze_library_tendency(self, shots: List[Dict]) -> List[StyleVector]:
        """分析素材库的风格倾向"""
        library_styles = []

        for shot in shots:
            if shot.get("asset_status") == "matched":
                asset = shot.get("scheduled_asset", {})
                if asset.get("source") == "library":
                    # 如果素材库有风格标签
                    style_tag = asset.get("style")
                    if style_tag:
                        try:
                            library_styles.append(
                                StyleVector(
                                    style_type=StyleType(style_tag.lower()),
                                    color_palette=[],
                                    saturation=0.5,
                                    brightness=0.5,
                                    contrast=0.5,
                                    texture_complexity=0.5,
                                    motion_intensity=0.3,
                                    camera_stability=0.7
                                )
                            )
                        except:
                            pass

        return library_styles

    def _select_dominant_style(self, styles: List[StyleVector]) -> Optional[StyleVector]:
        """选择主导风格"""
        if not styles:
            return None

        # 找出最常见的风格
        style_types = [s.style_type for s in styles]
        from collections import Counter
        most_common = Counter(style_types).most_common(1)[0]

        # 如果出现频率超过50%，使用这个风格
        if most_common[1] / len(styles) > 0.5:
            # 返回这个类型的平均风格向量
            matching_styles = [s for s in styles if s.style_type == most_common[0]]
            return self._aggregate_style_vectors(matching_styles)

        return None

    async def _infer_style_from_descriptions(self, shots: List[Dict]) -> StyleVector:
        """从分镜描述推断整体风格"""
        descriptions = []
        for shot in shots[:10]:  # 最多取前10个避免prompt过长
            desc = shot.get("description", "")
            if desc:
                descriptions.append(desc)

        if not descriptions:
            # 默认风格
            return StyleVector(
                style_type=StyleType.REALISTIC,
                color_palette=[],
                saturation=0.5,
                brightness=0.5,
                contrast=0.5,
                texture_complexity=0.5,
                motion_intensity=0.3,
                camera_stability=0.7
            )

        prompt = f"""
        分析以下视频分镜描述，推断最适合的视觉风格：

        分镜描述：
        {chr(10).join(f"- {d}" for d in descriptions)}

        请选择最匹配的风格类型：
        - cinematic: 电影感，暗调，镜头光晕
        - realistic: 写实，自然光，真实场景
        - anime: 二次元，卡通，夸张表情
        - documentary: 纪实，手持感，低饱和
        - advertisement: 广告风，高光，产品聚焦
        - cyberpunk: 赛博朋克，霓虹灯，科技感
        - watercolor: 水彩，手绘，柔和边缘

        只输出风格名称。
        """

        try:
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            response = await loop.run_in_executor(
                executor,
                lambda: self.qwen.generate(prompt=prompt, max_retries=3)
            )

            if response:
                style_name = response.strip().lower()
                try:
                    style_type = StyleType(style_name)
                except:
                    style_type = StyleType.REALISTIC
            else:
                style_type = StyleType.REALISTIC
        except:
            style_type = StyleType.REALISTIC

        # 返回推断的风格
        return self._get_style_preset(style_type)

    def _get_style_preset(self, style_type: StyleType) -> StyleVector:
        """获取预设的风格参数"""
        presets = {
            StyleType.CINEMATIC: StyleVector(
                style_type=StyleType.CINEMATIC,
                color_palette=["#1a1a2e", "#0f3460", "#533483"],
                saturation=0.6,
                brightness=0.3,
                contrast=0.8,
                texture_complexity=0.7,
                motion_intensity=0.4,
                camera_stability=0.9
            ),
            StyleType.REALISTIC: StyleVector(
                style_type=StyleType.REALISTIC,
                color_palette=["#f5f5f5", "#e8e8e8", "#333333"],
                saturation=0.5,
                brightness=0.6,
                contrast=0.5,
                texture_complexity=0.6,
                motion_intensity=0.3,
                camera_stability=0.8
            ),
            StyleType.ANIME: StyleVector(
                style_type=StyleType.ANIME,
                color_palette=["#ff6b6b", "#4ecdc4", "#ffe66d"],
                saturation=0.9,
                brightness=0.7,
                contrast=0.7,
                texture_complexity=0.3,
                motion_intensity=0.6,
                camera_stability=0.7
            ),
            StyleType.DOCUMENTARY: StyleVector(
                style_type=StyleType.DOCUMENTARY,
                color_palette=["#8b8680", "#a0a0a0", "#606060"],
                saturation=0.3,
                brightness=0.5,
                contrast=0.4,
                texture_complexity=0.8,
                motion_intensity=0.5,
                camera_stability=0.4
            ),
            StyleType.ADVERTISEMENT: StyleVector(
                style_type=StyleType.ADVERTISEMENT,
                color_palette=["#ffffff", "#ff0000", "#000000"],
                saturation=0.8,
                brightness=0.8,
                contrast=0.9,
                texture_complexity=0.2,
                motion_intensity=0.4,
                camera_stability=0.95
            ),
            StyleType.CYBERPUNK: StyleVector(
                style_type=StyleType.CYBERPUNK,
                color_palette=["#ff00ff", "#00ffff", "#1a0033"],
                saturation=1.0,
                brightness=0.4,
                contrast=0.9,
                texture_complexity=0.9,
                motion_intensity=0.7,
                camera_stability=0.6
            ),
            StyleType.WATERCOLOR: StyleVector(
                style_type=StyleType.WATERCOLOR,
                color_palette=["#ffd6cc", "#c9e4ca", "#a8dadc"],
                saturation=0.4,
                brightness=0.8,
                contrast=0.3,
                texture_complexity=0.5,
                motion_intensity=0.2,
                camera_stability=0.9
            )
        }

        return presets.get(style_type, presets[StyleType.REALISTIC])

    def check_style_compatibility(self, candidate_style: StyleVector) -> Tuple[bool, float]:
        """
        检查候选素材与锚点风格的兼容性
        返回：(是否兼容, 兼容度分数)
        """
        if not self.style_anchor:
            return True, 1.0

        distance = self.style_anchor.distance(candidate_style)
        compatibility_score = 1.0 - min(distance, 1.0)
        is_compatible = distance < self.style_compatibility_threshold

        return is_compatible, compatibility_score

    def extract_global_elements(self, shots: List[Dict]):
        """提取全局统一元素（人物、产品、品牌等）"""
        for shot in shots:
            description = shot.get("description", "").lower()

            # 提取主角
            if any(word in description for word in ["主角", "主人公", "主播", "模特"]):
                if not self.global_elements.main_character:
                    self.global_elements.main_character = {
                        "description": shot.get("character_description"),
                        "reference_image": shot.get("character_image"),
                        "shot_id": shot.get("shot_id")
                    }

            # 提取产品
            if any(word in description for word in ["产品", "商品", "物品"]):
                if not self.global_elements.product_reference:
                    self.global_elements.product_reference = {
                        "description": shot.get("product_description"),
                        "reference_image": shot.get("product_image"),
                        "shot_id": shot.get("shot_id")
                    }

            # 提取品牌元素
            if shot.get("brand_logo"):
                self.global_elements.brand_logo = shot.get("brand_logo")

        # 从用户素材中提取
        for material in self.user_reference_materials:
            if material.get("has_person") and not self.global_elements.main_character:
                self.global_elements.main_character = {
                    "reference_image": material.get("thumbnail"),
                    "from_user_material": True
                }

            if material.get("has_product") and not self.global_elements.product_reference:
                self.global_elements.product_reference = {
                    "reference_image": material.get("thumbnail"),
                    "from_user_material": True
                }

    def get_style_prompt(self) -> str:
        """获取用于AI生成的风格提示词"""
        if not self.style_anchor:
            return ""

        style_prompts = {
            StyleType.CINEMATIC: "cinematic lighting, film grain, anamorphic lens, moody atmosphere",
            StyleType.REALISTIC: "photorealistic, natural lighting, high detail, 8k resolution",
            StyleType.ANIME: "anime style, cel shading, vibrant colors, expressive characters",
            StyleType.DOCUMENTARY: "documentary style, handheld camera, natural colors, authentic",
            StyleType.ADVERTISEMENT: "commercial photography, studio lighting, product focus, clean",
            StyleType.CYBERPUNK: "cyberpunk aesthetic, neon lights, futuristic, high tech low life",
            StyleType.WATERCOLOR: "watercolor painting, soft edges, artistic, hand-painted"
        }

        base_prompt = style_prompts.get(self.style_anchor.style_type, "")

        # 添加颜色信息
        if self.style_anchor.color_palette:
            colors = ", ".join(self.style_anchor.color_palette[:3])
            base_prompt += f", color palette: {colors}"

        # 添加特征修饰
        if self.style_anchor.saturation > 0.7:
            base_prompt += ", vibrant colors"
        elif self.style_anchor.saturation < 0.3:
            base_prompt += ", desaturated"

        if self.style_anchor.contrast > 0.7:
            base_prompt += ", high contrast"

        return base_prompt

    def get_consistency_requirements(self) -> Dict[str, Any]:
        """获取一致性要求"""
        return {
            "style_type": self.style_anchor.style_type.value if self.style_anchor else None,
            "style_vector": self.style_anchor.to_vector().tolist() if self.style_anchor else None,
            "global_elements": {
                "main_character": self.global_elements.main_character,
                "product": self.global_elements.product_reference,
                "brand": self.global_elements.brand_logo,
                "colors": self.global_elements.color_scheme
            },
            "compatibility_threshold": self.style_compatibility_threshold,
            "style_prompt": self.get_style_prompt()
        }