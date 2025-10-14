#!/usr/bin/env python3
"""
Mock素材API - 模拟外部素材库接口
等有了真实的Java API后，只需要替换这个模块
"""

import random
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from enum import Enum

class MaterialType(Enum):
    VIDEO = "video"
    IMAGE = "image"
    AUDIO = "audio"

class MaterialCategory(Enum):
    TECH = "科技"
    CITY = "城市"
    BUSINESS = "商业"
    INNOVATION = "创新"
    FUTURE = "未来"
    NATURE = "自然"

@dataclass
class MaterialInfo:
    """素材信息"""
    id: str
    title: str
    url: str
    type: MaterialType
    category: MaterialCategory
    tags: List[str]
    duration: Optional[float] = None  # 视频/音频时长(秒)
    resolution: Optional[str] = None  # 视频/图片分辨率
    file_size: Optional[int] = None   # 文件大小(bytes)
    description: str = ""

class MockMaterialsAPI:
    """Mock素材API - 提供示例素材数据"""

    def __init__(self):
        self.materials = self._generate_mock_materials()

    def _generate_mock_materials(self) -> List[MaterialInfo]:
        """生成Mock素材数据"""
        materials = []

        # 科技类视频素材
        tech_videos = [
            MaterialInfo(
                id="tech_video_001",
                title="城市数字化转型",
                url="https://sample-videos.com/zip/10/mp4/SampleVideo_1280x720_1mb.mp4",
                type=MaterialType.VIDEO,
                category=MaterialCategory.TECH,
                tags=["科技", "数字化", "城市", "未来"],
                duration=30.0,
                resolution="1280x720",
                file_size=1024000,
                description="展示城市数字化转型的科技视频"
            ),
            MaterialInfo(
                id="tech_video_002",
                title="AI人工智能概念",
                url="https://sample-videos.com/zip/10/mp4/SampleVideo_1920x1080_2mb.mp4",
                type=MaterialType.VIDEO,
                category=MaterialCategory.TECH,
                tags=["AI", "人工智能", "科技", "创新"],
                duration=25.0,
                resolution="1920x1080",
                file_size=2048000,
                description="AI人工智能概念展示视频"
            ),
            MaterialInfo(
                id="tech_video_003",
                title="智能城市夜景",
                url="https://sample-videos.com/zip/10/mp4/SampleVideo_1920x1080_1mb.mp4",
                type=MaterialType.VIDEO,
                category=MaterialCategory.CITY,
                tags=["城市", "夜景", "智能", "科技"],
                duration=20.0,
                resolution="1920x1080",
                file_size=1024000,
                description="现代智能城市夜景视频"
            )
        ]

        # 城市类图片素材
        city_images = [
            MaterialInfo(
                id="city_img_001",
                title="现代化城市天际线",
                url="https://images.unsplash.com/photo-1449824913935-59a10b8d2000",
                type=MaterialType.IMAGE,
                category=MaterialCategory.CITY,
                tags=["城市", "天际线", "现代化", "建筑"],
                resolution="1920x1080",
                file_size=512000,
                description="现代化城市天际线高清图片"
            ),
            MaterialInfo(
                id="city_img_002",
                title="科技园区鸟瞰图",
                url="https://images.unsplash.com/photo-1486406146926-c627a92ad1ab",
                type=MaterialType.IMAGE,
                category=MaterialCategory.TECH,
                tags=["科技园", "鸟瞰", "建筑", "科技"],
                resolution="1920x1080",
                file_size=768000,
                description="科技园区鸟瞰图高清图片"
            )
        ]

        # 商业类音频素材
        business_audios = [
            MaterialInfo(
                id="bgm_001",
                title="科技感BGM",
                url="https://www.soundjay.com/misc/sounds/bell-ringing-05.wav",
                type=MaterialType.AUDIO,
                category=MaterialCategory.TECH,
                tags=["BGM", "科技", "背景音乐", "现代"],
                duration=120.0,
                file_size=2048000,
                description="适合科技视频的背景音乐"
            ),
            MaterialInfo(
                id="bgm_002",
                title="商业宣传BGM",
                url="https://www.soundjay.com/misc/sounds/bell-ringing-04.wav",
                type=MaterialType.AUDIO,
                category=MaterialCategory.BUSINESS,
                tags=["BGM", "商业", "宣传", "激励"],
                duration=90.0,
                file_size=1536000,
                description="适合商业宣传的激励音乐"
            )
        ]

        materials.extend(tech_videos)
        materials.extend(city_images)
        materials.extend(business_audios)

        return materials

    def search_materials(self,
                        keywords: List[str] = None,
                        material_type: MaterialType = None,
                        category: MaterialCategory = None,
                        duration_range: tuple = None,
                        limit: int = 10) -> List[MaterialInfo]:
        """搜索素材"""
        results = self.materials.copy()

        # 按关键词过滤
        if keywords:
            filtered = []
            for material in results:
                # 检查标签和标题是否包含关键词
                material_text = " ".join(material.tags + [material.title, material.description]).lower()
                if any(keyword.lower() in material_text for keyword in keywords):
                    filtered.append(material)
            results = filtered

        # 按类型过滤
        if material_type:
            results = [m for m in results if m.type == material_type]

        # 按分类过滤
        if category:
            results = [m for m in results if m.category == category]

        # 按时长过滤(仅对视频/音频)
        if duration_range and len(duration_range) == 2:
            min_dur, max_dur = duration_range
            results = [m for m in results
                      if m.duration and min_dur <= m.duration <= max_dur]

        # 随机排序以模拟真实API的多样性
        random.shuffle(results)

        return results[:limit]

    def get_material_by_id(self, material_id: str) -> Optional[MaterialInfo]:
        """根据ID获取素材"""
        for material in self.materials:
            if material.id == material_id:
                return material
        return None

    def get_recommended_materials(self,
                                 video_type: str,
                                 emotions: Dict[str, float],
                                 keywords: List[str],
                                 duration: int) -> Dict[str, List[MaterialInfo]]:
        """根据VGP分析结果推荐素材"""

        # 根据视频类型和情感推荐不同类型的素材
        recommendations = {
            "videos": [],
            "images": [],
            "bgm": [],
            "sfx": []
        }

        # 推荐视频素材
        video_keywords = keywords + [video_type]
        recommendations["videos"] = self.search_materials(
            keywords=video_keywords,
            material_type=MaterialType.VIDEO,
            duration_range=(5, duration),
            limit=5
        )

        # 推荐图片素材
        recommendations["images"] = self.search_materials(
            keywords=video_keywords,
            material_type=MaterialType.IMAGE,
            limit=10
        )

        # 推荐BGM
        if "激昂" in emotions or "励志" in emotions:
            bgm_category = MaterialCategory.BUSINESS
        else:
            bgm_category = MaterialCategory.TECH

        recommendations["bgm"] = self.search_materials(
            material_type=MaterialType.AUDIO,
            category=bgm_category,
            duration_range=(duration, 300),
            limit=3
        )

        return recommendations

# 全局实例
mock_materials_api = MockMaterialsAPI()

# 为了兼容现有代码，提供简化接口
def search_materials(keywords: List[str], material_type: str = None, limit: int = 10) -> List[Dict[str, Any]]:
    """简化的搜索接口 - 返回字典格式以兼容现有代码"""
    type_mapping = {
        "video": MaterialType.VIDEO,
        "image": MaterialType.IMAGE,
        "audio": MaterialType.AUDIO
    }

    search_type = type_mapping.get(material_type) if material_type else None
    materials = mock_materials_api.search_materials(
        keywords=keywords,
        material_type=search_type,
        limit=limit
    )

    # 转换为字典格式
    return [
        {
            "id": m.id,
            "title": m.title,
            "url": m.url,
            "type": m.type.value,
            "category": m.category.value,
            "tags": m.tags,
            "duration": m.duration,
            "resolution": m.resolution,
            "file_size": m.file_size,
            "description": m.description
        }
        for m in materials
    ]

def get_recommended_materials_for_vgp(video_type: str, emotions: Dict[str, float],
                                     keywords: List[str], duration: int) -> Dict[str, List[Dict[str, Any]]]:
    """为VGP分析结果推荐素材"""
    recommendations = mock_materials_api.get_recommended_materials(
        video_type=video_type,
        emotions=emotions,
        keywords=keywords,
        duration=duration
    )

    # 转换为字典格式
    result = {}
    for key, materials in recommendations.items():
        result[key] = [
            {
                "id": m.id,
                "title": m.title,
                "url": m.url,
                "type": m.type.value,
                "category": m.category.value,
                "tags": m.tags,
                "duration": m.duration,
                "resolution": m.resolution,
                "file_size": m.file_size,
                "description": m.description
            }
            for m in materials
        ]

    return result

if __name__ == "__main__":
    # 测试API
    api = MockMaterialsAPI()

    print("=== 搜索科技相关视频 ===")
    tech_videos = api.search_materials(keywords=["科技"], material_type=MaterialType.VIDEO)
    for video in tech_videos:
        print(f"- {video.title}: {video.url}")

    print("\n=== 获取VGP推荐素材 ===")
    recommendations = api.get_recommended_materials(
        video_type="商业类",
        emotions={"励志": 0.8, "科技": 0.6},
        keywords=["科技", "创新"],
        duration=30
    )

    for category, materials in recommendations.items():
        print(f"\n{category.upper()}:")
        for material in materials:
            print(f"  - {material.title}")