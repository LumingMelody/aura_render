"""
Initial Material Library Setup
初始素材库建设 - 建立基础素材库并集成免费素材源
"""
import asyncio
import json
import os
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path
import tempfile

from .material_download_manager import MaterialDownloadManager, MaterialStorage, DownloadRequest
from .material_taxonomy import MediaType, ContentCategory, StyleTag, QualityLevel, UsageRights


@dataclass
class MaterialSource:
    """素材源定义"""
    name: str
    base_url: str
    api_key: Optional[str] = None
    rate_limit: int = 100  # requests per hour
    supported_types: List[MediaType] = None
    quality_levels: List[QualityLevel] = None


class InitialMaterialLibrary:
    """初始素材库构建器"""

    def __init__(self, download_manager: MaterialDownloadManager):
        self.download_manager = download_manager
        self.material_sources = self._init_material_sources()

    def _init_material_sources(self) -> List[MaterialSource]:
        """初始化素材源配置"""
        return [
            # 免费视频源
            MaterialSource(
                name="pixabay_videos",
                base_url="https://pixabay.com/api/videos/",
                supported_types=[MediaType.VIDEO],
                quality_levels=[QualityLevel.STANDARD, QualityLevel.HIGH]
            ),

            # 免费图片源
            MaterialSource(
                name="unsplash",
                base_url="https://api.unsplash.com/",
                supported_types=[MediaType.IMAGE],
                quality_levels=[QualityLevel.HIGH, QualityLevel.PREMIUM]
            ),

            # 免费音频源
            MaterialSource(
                name="freesound",
                base_url="https://freesound.org/apiv2/",
                supported_types=[MediaType.AUDIO],
                quality_levels=[QualityLevel.STANDARD, QualityLevel.HIGH]
            ),

            # Pexels视频
            MaterialSource(
                name="pexels_videos",
                base_url="https://api.pexels.com/videos/",
                supported_types=[MediaType.VIDEO],
                quality_levels=[QualityLevel.HIGH, QualityLevel.PREMIUM]
            )
        ]

    async def build_initial_library(self) -> Dict[str, Any]:
        """构建初始素材库"""
        print("🚀 开始构建初始素材库...")

        # 构建基础素材集合
        material_collections = {
            "nature_videos": await self._build_nature_video_collection(),
            "business_videos": await self._build_business_video_collection(),
            "technology_videos": await self._build_technology_video_collection(),
            "background_music": await self._build_background_music_collection(),
            "sound_effects": await self._build_sound_effects_collection(),
            "stock_images": await self._build_stock_image_collection(),
        }

        # 统计构建结果
        total_materials = 0
        successful_downloads = 0

        for collection_name, results in material_collections.items():
            if results:
                collection_total = len(results)
                collection_success = sum(1 for r in results if r and r.success)
                total_materials += collection_total
                successful_downloads += collection_success

                print(f"✅ {collection_name}: {collection_success}/{collection_total} 成功")
            else:
                print(f"⚠️ {collection_name}: 构建失败")

        # 返回构建报告
        build_report = {
            "total_materials": total_materials,
            "successful_downloads": successful_downloads,
            "success_rate": (successful_downloads / max(1, total_materials)) * 100,
            "collections": {
                name: len(results) if results else 0
                for name, results in material_collections.items()
            },
            "storage_stats": self.download_manager.get_download_stats()
        }

        print(f"🎉 初始素材库构建完成！")
        print(f"   总素材数: {total_materials}")
        print(f"   成功下载: {successful_downloads}")
        print(f"   成功率: {build_report['success_rate']:.1f}%")

        return build_report

    async def _build_nature_video_collection(self) -> List[DownloadRequest]:
        """构建自然风光视频集合"""
        nature_keywords = [
            "mountain landscape", "ocean waves", "forest trees", "sunset sky",
            "river flowing", "clouds moving", "flowers blooming", "rain drops",
            "snow falling", "birds flying", "wind grass", "beach waves"
        ]

        requests = []
        for i, keyword in enumerate(nature_keywords):
            requests.append(DownloadRequest(
                url=f"https://mock-api.pixabay.com/videos/{keyword.replace(' ', '_')}.mp4",
                material_id=f"nature_video_{i+1:03d}",
                expected_type=MediaType.VIDEO,
                priority=2,
                metadata={
                    "description": f"自然风光视频: {keyword}",
                    "primary_category": ContentCategory.NATURE.value,
                    "style_tags": [StyleTag.REALISTIC.value, StyleTag.CINEMATIC.value],
                    "quality_level": QualityLevel.HIGH.value,
                    "keywords": keyword.split(),
                    "duration": 10.0,
                    "usage_rights": UsageRights.FREE.value
                }
            ))

        # 模拟下载（实际应用中会调用真实API）
        return await self._mock_download_batch(requests)

    async def _build_business_video_collection(self) -> List[DownloadRequest]:
        """构建商务视频集合"""
        business_keywords = [
            "office meeting", "teamwork collaboration", "business handshake",
            "data analysis", "presentation slides", "corporate building",
            "keyboard typing", "phone call", "document signing",
            "networking event", "success celebration", "growth chart"
        ]

        requests = []
        for i, keyword in enumerate(business_keywords):
            requests.append(DownloadRequest(
                url=f"https://mock-api.pexels.com/videos/{keyword.replace(' ', '_')}.mp4",
                material_id=f"business_video_{i+1:03d}",
                expected_type=MediaType.VIDEO,
                priority=3,
                metadata={
                    "description": f"商务场景视频: {keyword}",
                    "primary_category": ContentCategory.BUSINESS.value,
                    "style_tags": [StyleTag.MODERN.value, StyleTag.ADVERTISEMENT.value],
                    "quality_level": QualityLevel.HIGH.value,
                    "keywords": keyword.split(),
                    "duration": 8.0,
                    "usage_rights": UsageRights.FREE.value
                }
            ))

        return await self._mock_download_batch(requests)

    async def _build_technology_video_collection(self) -> List[DownloadRequest]:
        """构建科技视频集合"""
        tech_keywords = [
            "computer coding", "robot automation", "ai artificial intelligence",
            "data visualization", "circuit board", "smartphone technology",
            "virtual reality", "network connections", "cybersecurity",
            "cloud computing", "blockchain", "machine learning"
        ]

        requests = []
        for i, keyword in enumerate(tech_keywords):
            requests.append(DownloadRequest(
                url=f"https://mock-api.pixabay.com/videos/{keyword.replace(' ', '_')}.mp4",
                material_id=f"tech_video_{i+1:03d}",
                expected_type=MediaType.VIDEO,
                priority=2,
                metadata={
                    "description": f"科技场景视频: {keyword}",
                    "primary_category": ContentCategory.TECHNOLOGY.value,
                    "style_tags": [StyleTag.CYBERPUNK.value, StyleTag.MODERN.value],
                    "quality_level": QualityLevel.HIGH.value,
                    "keywords": keyword.split(),
                    "duration": 6.0,
                    "usage_rights": UsageRights.FREE.value
                }
            ))

        return await self._mock_download_batch(requests)

    async def _build_background_music_collection(self) -> List[DownloadRequest]:
        """构建背景音乐集合"""
        music_styles = [
            ("corporate_upbeat", "积极向上的企业音乐"),
            ("ambient_calm", "平静的环境音乐"),
            ("tech_electronic", "科技感电子音乐"),
            ("cinematic_epic", "史诗感电影音乐"),
            ("acoustic_warm", "温暖的原声音乐"),
            ("jazz_smooth", "柔滑的爵士音乐"),
            ("classical_elegant", "优雅的古典音乐"),
            ("pop_energetic", "充满活力的流行音乐")
        ]

        requests = []
        for i, (style_id, description) in enumerate(music_styles):
            requests.append(DownloadRequest(
                url=f"https://mock-api.freesound.org/sounds/{style_id}.mp3",
                material_id=f"bgm_{i+1:03d}",
                expected_type=MediaType.AUDIO,
                priority=2,
                metadata={
                    "description": description,
                    "primary_category": ContentCategory.BACKGROUND_MUSIC.value,
                    "quality_level": QualityLevel.STANDARD.value,
                    "keywords": style_id.split('_'),
                    "duration": 120.0,  # 2分钟
                    "usage_rights": UsageRights.FREE.value
                }
            ))

        return await self._mock_download_batch(requests)

    async def _build_sound_effects_collection(self) -> List[DownloadRequest]:
        """构建音效集合"""
        sfx_types = [
            ("click_button", "按钮点击音效"),
            ("notification_bell", "通知铃声"),
            ("success_chime", "成功提示音"),
            ("error_buzz", "错误提示音"),
            ("transition_swoosh", "转场音效"),
            ("typing_keyboard", "键盘打字音"),
            ("phone_ring", "电话铃声"),
            ("applause_crowd", "掌声音效"),
            ("door_open", "开门音效"),
            ("water_drop", "水滴音效"),
            ("wind_breeze", "微风音效"),
            ("footstep_walk", "脚步声音效")
        ]

        requests = []
        for i, (sfx_id, description) in enumerate(sfx_types):
            requests.append(DownloadRequest(
                url=f"https://mock-api.freesound.org/sounds/{sfx_id}.wav",
                material_id=f"sfx_{i+1:03d}",
                expected_type=MediaType.AUDIO,
                priority=1,
                metadata={
                    "description": description,
                    "primary_category": ContentCategory.SOUND_EFFECTS.value,
                    "quality_level": QualityLevel.STANDARD.value,
                    "keywords": sfx_id.split('_'),
                    "duration": 2.0,  # 2秒
                    "usage_rights": UsageRights.FREE.value
                }
            ))

        return await self._mock_download_batch(requests)

    async def _build_stock_image_collection(self) -> List[DownloadRequest]:
        """构建库存图片集合"""
        image_categories = [
            ("business_team", "商务团队合作"),
            ("nature_landscape", "自然风景"),
            ("technology_devices", "科技设备"),
            ("lifestyle_home", "居家生活方式"),
            ("food_cooking", "美食烹饪"),
            ("travel_adventure", "旅行探险"),
            ("education_learning", "教育学习"),
            ("healthcare_medical", "医疗健康"),
            ("sports_fitness", "运动健身"),
            ("art_creative", "艺术创意"),
            ("city_architecture", "城市建筑"),
            ("abstract_pattern", "抽象图案")
        ]

        requests = []
        for i, (img_id, description) in enumerate(image_categories):
            requests.append(DownloadRequest(
                url=f"https://mock-api.unsplash.com/photos/{img_id}.jpg",
                material_id=f"image_{i+1:03d}",
                expected_type=MediaType.IMAGE,
                priority=1,
                metadata={
                    "description": description,
                    "primary_category": ContentCategory(img_id.split('_')[0]).value
                    if img_id.split('_')[0] in [c.value for c in ContentCategory] else ContentCategory.LIFESTYLE.value,
                    "quality_level": QualityLevel.HIGH.value,
                    "keywords": img_id.split('_'),
                    "dimensions": (1920, 1080),
                    "usage_rights": UsageRights.FREE.value
                }
            ))

        return await self._mock_download_batch(requests)

    async def _mock_download_batch(self, requests: List[DownloadRequest]) -> List:
        """模拟批量下载（用于测试）"""
        # 在实际应用中，这里会调用真实的下载管理器
        # return await self.download_manager.batch_download(requests)

        # 模拟下载结果
        print(f"  📦 模拟下载 {len(requests)} 个素材...")

        # 创建模拟文件
        mock_results = []
        for request in requests:
            # 创建临时模拟文件
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mock')
            temp_file.write(b"Mock material content")
            temp_file.close()

            # 模拟成功结果
            from .material_download_manager import DownloadResult
            result = DownloadResult(
                success=True,
                material_id=request.material_id,
                local_path=temp_file.name,
                file_size=len(b"Mock material content"),
                content_type="application/octet-stream"
            )
            mock_results.append(result)

            # 创建并保存元数据
            from .material_taxonomy import MaterialMetadata, ContentCategory

            # 确保primary_category是ContentCategory枚举
            primary_category_value = request.metadata.get('primary_category', ContentCategory.LIFESTYLE.value)
            if isinstance(primary_category_value, str):
                # 将字符串转换为枚举
                primary_category = ContentCategory(primary_category_value)
            else:
                primary_category = primary_category_value

            # 获取质量等级
            quality_level_value = request.metadata.get('quality_level', QualityLevel.STANDARD.value)
            if isinstance(quality_level_value, str):
                quality_level = QualityLevel(quality_level_value)
            else:
                quality_level = quality_level_value

            # 获取使用权限
            usage_rights_value = request.metadata.get('usage_rights', UsageRights.FREE.value)
            if isinstance(usage_rights_value, str):
                usage_rights = UsageRights(usage_rights_value)
            else:
                usage_rights = usage_rights_value

            metadata = MaterialMetadata(
                material_id=request.material_id,
                filename=f"{request.material_id}.mock",
                media_type=request.expected_type,
                file_size=result.file_size,
                primary_category=primary_category,
                duration=request.metadata.get('duration'),
                dimensions=request.metadata.get('dimensions'),
                quality_level=quality_level,
                keywords=request.metadata.get('keywords', []),
                usage_rights=usage_rights
            )

            # 保存到存储系统
            self.download_manager.storage.save_material(
                request.material_id,
                temp_file.name,
                metadata,
                request.url
            )

        return mock_results

    def create_material_catalog(self) -> Dict[str, Any]:
        """创建素材目录"""
        storage = self.download_manager.storage

        # 获取所有素材
        all_materials = storage.list_materials(limit=1000)

        # 按类别分组
        catalog = {
            "categories": {},
            "styles": {},
            "quality_levels": {},
            "total_count": len(all_materials),
            "catalog_updated": "2025-01-15T10:00:00Z"
        }

        # 统计分类
        for material in all_materials:
            if 'parsed_metadata' in material:
                metadata = material['parsed_metadata']

                # 按主类别分组
                primary_cat = metadata.get('primary_category', 'unknown')
                if primary_cat not in catalog["categories"]:
                    catalog["categories"][primary_cat] = {
                        "count": 0,
                        "materials": []
                    }
                catalog["categories"][primary_cat]["count"] += 1
                catalog["categories"][primary_cat]["materials"].append({
                    "material_id": material['material_id'],
                    "filename": material['filename'],
                    "description": metadata.get('description', ''),
                    "file_size": material['file_size'],
                    "created_at": material['created_at']
                })

                # 按风格分组
                style_tags = metadata.get('style_tags', [])
                for style in style_tags:
                    if style not in catalog["styles"]:
                        catalog["styles"][style] = {"count": 0, "materials": []}
                    catalog["styles"][style]["count"] += 1
                    catalog["styles"][style]["materials"].append(material['material_id'])

                # 按质量等级分组
                quality = metadata.get('quality_level', 'standard')
                if quality not in catalog["quality_levels"]:
                    catalog["quality_levels"][quality] = {"count": 0}
                catalog["quality_levels"][quality]["count"] += 1

        return catalog

    def generate_library_report(self) -> Dict[str, Any]:
        """生成素材库报告"""
        catalog = self.create_material_catalog()
        stats = self.download_manager.get_download_stats()

        return {
            "library_overview": {
                "total_materials": catalog["total_count"],
                "categories_count": len(catalog["categories"]),
                "styles_count": len(catalog["styles"]),
                "quality_levels": list(catalog["quality_levels"].keys())
            },
            "category_breakdown": catalog["categories"],
            "style_distribution": {
                style: data["count"]
                for style, data in catalog["styles"].items()
            },
            "quality_distribution": catalog["quality_levels"],
            "storage_information": stats["storage"],
            "download_statistics": {
                "total_downloads": stats["total_downloads"],
                "successful_downloads": stats["successful_downloads"],
                "success_rate": stats["success_rate"],
                "total_bytes": stats["total_bytes"]
            },
            "report_generated": "2025-01-15T10:00:00Z"
        }


async def setup_initial_library():
    """设置初始素材库的主函数"""
    print("🎬 Aura Render 初始素材库设置")
    print("=" * 50)

    # 初始化存储和下载管理器
    storage = MaterialStorage("/tmp/aura_render_outputs/materials")
    download_manager = MaterialDownloadManager(storage, max_concurrent=3)

    # 创建初始素材库构建器
    library_builder = InitialMaterialLibrary(download_manager)

    # 构建素材库
    build_report = await library_builder.build_initial_library()

    # 生成报告
    library_report = library_builder.generate_library_report()

    # 保存报告
    report_path = storage.base_path / "library_report.json"
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(library_report, f, indent=2, ensure_ascii=False)

    print(f"\n📊 素材库报告已保存: {report_path}")
    print("\n🎉 初始素材库设置完成！")

    return {
        "build_report": build_report,
        "library_report": library_report,
        "storage_path": str(storage.base_path)
    }


if __name__ == "__main__":
    asyncio.run(setup_initial_library())