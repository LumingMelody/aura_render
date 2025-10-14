"""
Material Taxonomy and Classification System
素材分类和标签体系 - 支持多维度智能分类和标签管理
"""
from typing import Dict, List, Set, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from datetime import datetime


class MediaType(Enum):
    """媒体类型枚举"""
    VIDEO = "video"
    AUDIO = "audio"
    IMAGE = "image"
    TEXT = "text"
    FONT = "font"
    TEMPLATE = "template"


class ContentCategory(Enum):
    """内容分类枚举"""
    # 视觉素材类别
    NATURE = "nature"           # 自然风光
    TECHNOLOGY = "technology"   # 科技产品
    BUSINESS = "business"      # 商业办公
    LIFESTYLE = "lifestyle"    # 生活方式
    EDUCATION = "education"    # 教育培训
    ENTERTAINMENT = "entertainment"  # 娱乐休闲
    HEALTHCARE = "healthcare"  # 医疗健康
    FOOD = "food"             # 食物料理
    TRAVEL = "travel"         # 旅行探索
    SPORTS = "sports"         # 运动健身

    # 音频类别
    BACKGROUND_MUSIC = "bgm"   # 背景音乐
    VOICE_OVER = "voiceover"  # 配音旁白
    SOUND_EFFECTS = "sfx"     # 音效
    AMBIENT = "ambient"       # 环境音

    # 文本类别
    TITLE = "title"           # 标题文字
    SUBTITLE = "subtitle"     # 字幕文字
    CAPTION = "caption"       # 说明文字
    WATERMARK = "watermark"   # 水印文字


class StyleTag(Enum):
    """视觉风格标签"""
    CINEMATIC = "cinematic"         # 电影级
    REALISTIC = "realistic"         # 写实风格
    ANIME = "anime"                # 动漫风格
    DOCUMENTARY = "documentary"     # 纪录片风格
    ADVERTISEMENT = "advertisement" # 广告风格
    CYBERPUNK = "cyberpunk"        # 赛博朋克
    WATERCOLOR = "watercolor"      # 水彩风格
    MINIMALIST = "minimalist"      # 极简风格
    VINTAGE = "vintage"            # 复古风格
    MODERN = "modern"              # 现代风格


class QualityLevel(Enum):
    """质量等级"""
    LOW = "low"           # 低质量 (720p以下, <5MB)
    STANDARD = "standard" # 标准质量 (1080p, 5-20MB)
    HIGH = "high"        # 高质量 (1080p+, 20-100MB)
    PREMIUM = "premium"   # 顶级质量 (4K+, >100MB)


class UsageRights(Enum):
    """使用权限"""
    FREE = "free"                    # 免费使用
    ATTRIBUTION = "attribution"       # 需署名
    COMMERCIAL = "commercial"         # 商业授权
    EDITORIAL = "editorial"           # 编辑使用
    RESTRICTED = "restricted"         # 受限使用


@dataclass
class MaterialTag:
    """素材标签"""
    key: str
    value: str
    category: str = "general"
    confidence: float = 1.0
    source: str = "manual"  # manual, ai_generated, extracted

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "value": self.value,
            "category": self.category,
            "confidence": self.confidence,
            "source": self.source
        }


@dataclass
class MaterialMetadata:
    """素材完整元数据"""
    # 基础信息
    material_id: str
    filename: str
    media_type: MediaType
    file_size: int
    primary_category: ContentCategory
    duration: Optional[float] = None  # 对视频/音频有效
    dimensions: Optional[Tuple[int, int]] = None  # 对图片/视频有效

    # 内容分类
    secondary_categories: List[ContentCategory] = field(default_factory=list)

    # 风格和质量
    style_tags: List[StyleTag] = field(default_factory=list)
    quality_level: QualityLevel = QualityLevel.STANDARD

    # 标签系统
    tags: List[MaterialTag] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)

    # 权限和来源
    usage_rights: UsageRights = UsageRights.FREE
    source: str = "unknown"
    provider: str = "local"

    # 统计信息
    view_count: int = 0
    download_count: int = 0
    rating: float = 0.0

    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典格式"""
        return {
            "material_id": self.material_id,
            "filename": self.filename,
            "media_type": self.media_type.value,
            "file_size": self.file_size,
            "duration": self.duration,
            "dimensions": self.dimensions,
            "primary_category": self.primary_category.value,
            "secondary_categories": [cat.value for cat in self.secondary_categories],
            "style_tags": [tag.value for tag in self.style_tags],
            "quality_level": self.quality_level.value,
            "tags": [tag.to_dict() for tag in self.tags],
            "keywords": self.keywords,
            "usage_rights": self.usage_rights.value,
            "source": self.source,
            "provider": self.provider,
            "view_count": self.view_count,
            "download_count": self.download_count,
            "rating": self.rating,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat()
        }


class MaterialTaxonomy:
    """素材分类体系管理器"""

    def __init__(self):
        self.category_hierarchy = self._build_category_hierarchy()
        self.style_mapping = self._build_style_mapping()
        self.keyword_categories = self._build_keyword_categories()

    def _build_category_hierarchy(self) -> Dict[str, List[str]]:
        """构建分类层级关系"""
        return {
            "visual_content": [
                ContentCategory.NATURE.value,
                ContentCategory.TECHNOLOGY.value,
                ContentCategory.BUSINESS.value,
                ContentCategory.LIFESTYLE.value,
                ContentCategory.EDUCATION.value,
                ContentCategory.ENTERTAINMENT.value,
                ContentCategory.HEALTHCARE.value,
                ContentCategory.FOOD.value,
                ContentCategory.TRAVEL.value,
                ContentCategory.SPORTS.value
            ],
            "audio_content": [
                ContentCategory.BACKGROUND_MUSIC.value,
                ContentCategory.VOICE_OVER.value,
                ContentCategory.SOUND_EFFECTS.value,
                ContentCategory.AMBIENT.value
            ],
            "text_content": [
                ContentCategory.TITLE.value,
                ContentCategory.SUBTITLE.value,
                ContentCategory.CAPTION.value,
                ContentCategory.WATERMARK.value
            ]
        }

    def _build_style_mapping(self) -> Dict[str, List[str]]:
        """构建风格映射关系"""
        return {
            StyleTag.CINEMATIC.value: ["电影", "大片", "史诗", "戏剧性", "镜头感"],
            StyleTag.REALISTIC.value: ["写实", "真实", "自然", "纪实", "原生"],
            StyleTag.ANIME.value: ["动漫", "二次元", "卡通", "动画", "漫画"],
            StyleTag.DOCUMENTARY.value: ["纪录片", "documentary", "真实记录", "客观"],
            StyleTag.ADVERTISEMENT.value: ["广告", "商业", "推广", "营销", "产品"],
            StyleTag.CYBERPUNK.value: ["赛博朋克", "科幻", "未来", "霓虹", "科技"],
            StyleTag.WATERCOLOR.value: ["水彩", "手绘", "艺术", "柔和", "绘画"],
            StyleTag.MINIMALIST.value: ["极简", "简约", "干净", "简洁", "纯净"],
            StyleTag.VINTAGE.value: ["复古", "怀旧", "古典", "老式", "经典"],
            StyleTag.MODERN.value: ["现代", "时尚", "当代", "潮流", "都市"]
        }

    def _build_keyword_categories(self) -> Dict[str, List[str]]:
        """构建关键词分类映射"""
        return {
            ContentCategory.NATURE.value: [
                "自然", "风景", "山", "海", "森林", "天空", "云", "日出", "日落",
                "花", "树", "草", "河流", "湖泊", "动物", "鸟", "风", "雨"
            ],
            ContentCategory.TECHNOLOGY.value: [
                "科技", "电脑", "手机", "AI", "机器人", "代码", "数据", "网络",
                "芯片", "电子", "创新", "未来", "智能", "编程", "软件"
            ],
            ContentCategory.BUSINESS.value: [
                "商务", "办公", "会议", "团队", "合作", "成功", "增长", "数据",
                "图表", "分析", "策略", "领导", "管理", "销售", "市场"
            ],
            ContentCategory.LIFESTYLE.value: [
                "生活", "家庭", "朋友", "聚会", "休闲", "购物", "咖啡", "读书",
                "音乐", "艺术", "时尚", "美容", "健康", "快乐", "温馨"
            ],
            ContentCategory.EDUCATION.value: [
                "教育", "学习", "学校", "老师", "学生", "书本", "知识", "研究",
                "实验", "科学", "数学", "语言", "培训", "技能", "成长"
            ]
        }

    def classify_material(self, description: str, filename: str = "",
                         existing_tags: List[str] = None) -> Dict[str, Any]:
        """
        智能分类素材

        Args:
            description: 素材描述
            filename: 文件名
            existing_tags: 已有标签

        Returns:
            分类结果
        """
        existing_tags = existing_tags or []
        text_content = f"{description} {filename} {' '.join(existing_tags)}".lower()

        # 分类分析
        primary_category, category_confidence = self._classify_content_category(text_content)
        style_tags = self._identify_style_tags(text_content)
        quality_level = self._assess_quality_level(filename, existing_tags)

        # 提取关键词
        keywords = self._extract_keywords(text_content)

        # 生成标签
        tags = self._generate_tags(text_content, primary_category, style_tags)

        return {
            "primary_category": primary_category,
            "category_confidence": category_confidence,
            "style_tags": style_tags,
            "quality_level": quality_level,
            "keywords": keywords,
            "generated_tags": tags,
            "classification_metadata": {
                "processed_text": text_content,
                "classification_timestamp": datetime.now().isoformat()
            }
        }

    def _classify_content_category(self, text: str) -> Tuple[ContentCategory, float]:
        """分类内容类别"""
        category_scores = {}

        for category, keywords in self.keyword_categories.items():
            score = 0
            for keyword in keywords:
                if keyword.lower() in text:
                    score += 1

            # 归一化分数
            if keywords:
                category_scores[category] = score / len(keywords)

        if not category_scores:
            return ContentCategory.LIFESTYLE, 0.1

        best_category = max(category_scores.items(), key=lambda x: x[1])
        category_enum = ContentCategory(best_category[0])
        confidence = best_category[1]

        return category_enum, confidence

    def _identify_style_tags(self, text: str) -> List[StyleTag]:
        """识别风格标签"""
        identified_styles = []

        for style, keywords in self.style_mapping.items():
            for keyword in keywords:
                if keyword.lower() in text:
                    style_enum = StyleTag(style)
                    if style_enum not in identified_styles:
                        identified_styles.append(style_enum)
                    break

        return identified_styles

    def _assess_quality_level(self, filename: str, tags: List[str]) -> QualityLevel:
        """评估质量等级"""
        text = f"{filename} {' '.join(tags)}".lower()

        # 基于关键词判断
        if any(keyword in text for keyword in ["4k", "ultra", "premium", "high-res"]):
            return QualityLevel.PREMIUM
        elif any(keyword in text for keyword in ["hd", "1080p", "high"]):
            return QualityLevel.HIGH
        elif any(keyword in text for keyword in ["low", "compressed", "small"]):
            return QualityLevel.LOW
        else:
            return QualityLevel.STANDARD

    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简单的关键词提取 - 在实际应用中可以用更sophisticated的方法
        words = re.findall(r'\b\w+\b', text.lower())

        # 过滤停用词
        stop_words = {
            "的", "在", "是", "有", "和", "与", "或", "等", "了", "着", "过",
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for"
        }

        keywords = [word for word in words if len(word) > 2 and word not in stop_words]

        # 去重并限制数量
        return list(set(keywords))[:20]

    def _generate_tags(self, text: str, category: ContentCategory,
                      style_tags: List[StyleTag]) -> List[MaterialTag]:
        """生成标签"""
        tags = []

        # 分类标签
        tags.append(MaterialTag(
            key="category",
            value=category.value,
            category="content",
            confidence=0.9,
            source="ai_generated"
        ))

        # 风格标签
        for style in style_tags:
            tags.append(MaterialTag(
                key="style",
                value=style.value,
                category="visual",
                confidence=0.8,
                source="ai_generated"
            ))

        # 自动检测的特殊标签
        if "人" in text or "person" in text:
            tags.append(MaterialTag("content", "people", "subject", 0.7, "ai_generated"))

        if "产品" in text or "product" in text:
            tags.append(MaterialTag("content", "product", "subject", 0.7, "ai_generated"))

        return tags


class MaterialTagManager:
    """素材标签管理器"""

    def __init__(self):
        self.taxonomy = MaterialTaxonomy()
        self.tag_statistics = {}

    def add_material_tags(self, material_id: str, metadata: MaterialMetadata,
                         description: str = "") -> MaterialMetadata:
        """为素材添加智能标签"""

        # 执行智能分类
        classification_result = self.taxonomy.classify_material(
            description=description,
            filename=metadata.filename,
            existing_tags=metadata.keywords
        )

        # 更新元数据
        metadata.primary_category = ContentCategory(classification_result["primary_category"])
        metadata.style_tags = classification_result["style_tags"]
        metadata.quality_level = classification_result["quality_level"]
        metadata.keywords.extend(classification_result["keywords"])
        metadata.tags.extend(classification_result["generated_tags"])

        # 去重关键词
        metadata.keywords = list(set(metadata.keywords))

        # 更新时间戳
        metadata.updated_at = datetime.now()

        # 更新标签统计
        self._update_tag_statistics(metadata.tags)

        return metadata

    def search_by_tags(self, materials: List[MaterialMetadata],
                      tag_queries: List[str],
                      category_filter: Optional[ContentCategory] = None,
                      style_filter: Optional[StyleTag] = None) -> List[Tuple[MaterialMetadata, float]]:
        """基于标签搜索素材"""
        results = []

        for material in materials:
            score = self._calculate_match_score(
                material, tag_queries, category_filter, style_filter
            )

            if score > 0:
                results.append((material, score))

        # 按分数排序
        results.sort(key=lambda x: x[1], reverse=True)
        return results

    def _calculate_match_score(self, material: MaterialMetadata,
                              tag_queries: List[str],
                              category_filter: Optional[ContentCategory],
                              style_filter: Optional[StyleTag]) -> float:
        """计算匹配分数"""
        score = 0.0

        # 分类匹配
        if category_filter:
            if material.primary_category == category_filter:
                score += 2.0
            elif category_filter in material.secondary_categories:
                score += 1.0

        # 风格匹配
        if style_filter:
            if style_filter in material.style_tags:
                score += 1.5

        # 标签匹配
        all_tags = [tag.value for tag in material.tags] + material.keywords
        for query in tag_queries:
            query_lower = query.lower()
            for tag in all_tags:
                if query_lower in tag.lower():
                    score += 1.0
                    break

        return score

    def _update_tag_statistics(self, tags: List[MaterialTag]):
        """更新标签统计"""
        for tag in tags:
            key = f"{tag.key}:{tag.value}"
            if key not in self.tag_statistics:
                self.tag_statistics[key] = 0
            self.tag_statistics[key] += 1

    def get_popular_tags(self, limit: int = 20) -> List[Tuple[str, int]]:
        """获取热门标签"""
        sorted_tags = sorted(
            self.tag_statistics.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_tags[:limit]

    def get_category_distribution(self, materials: List[MaterialMetadata]) -> Dict[str, int]:
        """获取分类分布"""
        distribution = {}
        for material in materials:
            category = material.primary_category.value
            distribution[category] = distribution.get(category, 0) + 1
        return distribution


# 使用示例和测试
def test_material_taxonomy():
    """测试素材分类系统"""
    taxonomy = MaterialTaxonomy()
    tag_manager = MaterialTagManager()

    # 测试素材
    test_cases = [
        {
            "description": "美丽的日出山景，cinematic风格拍摄",
            "filename": "mountain_sunrise_4k.mp4"
        },
        {
            "description": "现代办公室商务会议场景",
            "filename": "business_meeting_hd.jpg"
        },
        {
            "description": "科技感十足的AI机器人动画",
            "filename": "ai_robot_animation.gif"
        }
    ]

    results = []
    for case in test_cases:
        result = taxonomy.classify_material(
            case["description"],
            case["filename"]
        )
        results.append({
            "input": case,
            "classification": result
        })

        print(f"素材: {case['filename']}")
        print(f"分类: {result['primary_category'].value}")
        print(f"风格: {[style.value for style in result['style_tags']]}")
        print(f"质量: {result['quality_level'].value}")
        print(f"关键词: {result['keywords'][:5]}")
        print("-" * 50)

    return results


if __name__ == "__main__":
    test_material_taxonomy()