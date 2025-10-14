"""
Material Download and Management System
素材下载和管理系统 - 支持多源下载、缓存管理和智能存储
"""
import os
import asyncio
import aiohttp
import hashlib
import json
import shutil
from typing import Dict, List, Optional, Union, Any, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timedelta
from urllib.parse import urlparse
import tempfile
import mimetypes
from concurrent.futures import ThreadPoolExecutor
import sqlite3
import logging

from .material_taxonomy import MaterialMetadata, MediaType, MaterialTag, MaterialTagManager


@dataclass
class DownloadRequest:
    """下载请求"""
    url: str
    material_id: str
    expected_type: MediaType
    metadata: Optional[Dict[str, Any]] = None
    priority: int = 1  # 1=low, 2=normal, 3=high
    max_retries: int = 3
    timeout: int = 30
    expected_size: Optional[int] = None


@dataclass
class DownloadResult:
    """下载结果"""
    success: bool
    material_id: str
    local_path: Optional[str] = None
    file_size: int = 0
    content_type: Optional[str] = None
    download_time: float = 0.0
    error_message: Optional[str] = None
    checksum: Optional[str] = None


class MaterialStorage:
    """素材存储管理"""

    def __init__(self, base_path: str = "/tmp/aura_render_outputs/materials"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)

        # 创建分类目录
        self.directories = {
            MediaType.VIDEO: self.base_path / "videos",
            MediaType.AUDIO: self.base_path / "audio",
            MediaType.IMAGE: self.base_path / "images",
            MediaType.TEXT: self.base_path / "text",
            MediaType.FONT: self.base_path / "fonts",
            MediaType.TEMPLATE: self.base_path / "templates"
        }

        for directory in self.directories.values():
            directory.mkdir(parents=True, exist_ok=True)

        # 初始化数据库
        self.db_path = self.base_path / "materials.db"
        self._init_database()

        # 初始化日志
        self.logger = logging.getLogger(__name__)

    def _init_database(self):
        """初始化SQLite数据库"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS materials (
                    material_id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    local_path TEXT NOT NULL,
                    original_url TEXT,
                    media_type TEXT NOT NULL,
                    file_size INTEGER NOT NULL,
                    checksum TEXT,
                    content_type TEXT,
                    metadata TEXT,  -- JSON格式
                    created_at TEXT NOT NULL,
                    accessed_at TEXT NOT NULL,
                    download_count INTEGER DEFAULT 0
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_media_type ON materials(media_type)
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_created_at ON materials(created_at)
            """)

    def save_material(self, material_id: str, file_path: str, metadata: MaterialMetadata,
                     original_url: str = None, checksum: str = None) -> bool:
        """保存素材到存储系统"""
        try:
            # 确定目标路径
            target_dir = self.directories[metadata.media_type]

            # 生成安全的文件名
            safe_filename = self._generate_safe_filename(
                metadata.filename, metadata.media_type
            )
            target_path = target_dir / safe_filename

            # 复制文件
            shutil.copy2(file_path, target_path)

            # 保存到数据库
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO materials
                    (material_id, filename, local_path, original_url, media_type,
                     file_size, checksum, content_type, metadata, created_at, accessed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    material_id,
                    safe_filename,
                    str(target_path),
                    original_url,
                    metadata.media_type.value,
                    metadata.file_size,
                    checksum,
                    mimetypes.guess_type(safe_filename)[0],
                    json.dumps(metadata.to_dict()),
                    datetime.now().isoformat(),
                    datetime.now().isoformat()
                ))

            self.logger.info(f"Material saved: {material_id} -> {target_path}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to save material {material_id}: {e}")
            return False

    def get_material_path(self, material_id: str) -> Optional[str]:
        """获取素材本地路径"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT local_path FROM materials WHERE material_id = ?",
                (material_id,)
            )
            result = cursor.fetchone()

            if result and os.path.exists(result[0]):
                # 更新访问时间
                conn.execute(
                    "UPDATE materials SET accessed_at = ?, download_count = download_count + 1 WHERE material_id = ?",
                    (datetime.now().isoformat(), material_id)
                )
                return result[0]

        return None

    def get_material_metadata(self, material_id: str) -> Optional[Dict[str, Any]]:
        """获取素材元数据"""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                "SELECT * FROM materials WHERE material_id = ?",
                (material_id,)
            )

            result = cursor.fetchone()
            if result:
                columns = [description[0] for description in cursor.description]
                material_dict = dict(zip(columns, result))

                # 解析JSON元数据
                if material_dict['metadata']:
                    material_dict['parsed_metadata'] = json.loads(material_dict['metadata'])

                return material_dict

        return None

    def list_materials(self, media_type: Optional[MediaType] = None,
                      limit: int = 100) -> List[Dict[str, Any]]:
        """列出素材"""
        query = "SELECT * FROM materials"
        params = []

        if media_type:
            query += " WHERE media_type = ?"
            params.append(media_type.value)

        query += " ORDER BY created_at DESC LIMIT ?"
        params.append(limit)

        materials = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(query, params)
            columns = [description[0] for description in cursor.description]

            for row in cursor.fetchall():
                material_dict = dict(zip(columns, row))
                if material_dict['metadata']:
                    material_dict['parsed_metadata'] = json.loads(material_dict['metadata'])
                materials.append(material_dict)

        return materials

    def delete_material(self, material_id: str) -> bool:
        """删除素材"""
        try:
            # 获取文件路径
            material_path = self.get_material_path(material_id)

            # 删除物理文件
            if material_path and os.path.exists(material_path):
                os.remove(material_path)

            # 从数据库删除
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("DELETE FROM materials WHERE material_id = ?", (material_id,))

            self.logger.info(f"Material deleted: {material_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to delete material {material_id}: {e}")
            return False

    def cleanup_old_materials(self, days: int = 30) -> int:
        """清理旧素材"""
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()

        with sqlite3.connect(self.db_path) as conn:
            # 获取要删除的材料
            cursor = conn.execute(
                "SELECT material_id, local_path FROM materials WHERE accessed_at < ?",
                (cutoff_date,)
            )

            deleted_count = 0
            for material_id, local_path in cursor.fetchall():
                if os.path.exists(local_path):
                    os.remove(local_path)
                    deleted_count += 1

            # 从数据库删除
            conn.execute("DELETE FROM materials WHERE accessed_at < ?", (cutoff_date,))

        self.logger.info(f"Cleaned up {deleted_count} old materials")
        return deleted_count

    def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计"""
        stats = {
            "total_materials": 0,
            "total_size": 0,
            "by_type": {},
            "storage_path": str(self.base_path)
        }

        with sqlite3.connect(self.db_path) as conn:
            # 总数统计
            cursor = conn.execute("SELECT COUNT(*), SUM(file_size) FROM materials")
            total_count, total_size = cursor.fetchone()
            stats["total_materials"] = total_count or 0
            stats["total_size"] = total_size or 0

            # 按类型统计
            cursor = conn.execute("""
                SELECT media_type, COUNT(*), SUM(file_size)
                FROM materials
                GROUP BY media_type
            """)

            for media_type, count, size in cursor.fetchall():
                stats["by_type"][media_type] = {
                    "count": count,
                    "total_size": size or 0
                }

        return stats

    def _generate_safe_filename(self, original_name: str, media_type: MediaType) -> str:
        """生成安全的文件名"""
        # 清理文件名
        safe_name = "".join(c for c in original_name if c.isalnum() or c in ".-_")

        # 添加时间戳避免冲突
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # 根据媒体类型添加扩展名
        name_parts = safe_name.rsplit('.', 1)
        if len(name_parts) == 2:
            base_name, ext = name_parts
            return f"{base_name}_{timestamp}.{ext}"
        else:
            # 如果没有扩展名，根据类型添加默认扩展名
            extensions = {
                MediaType.VIDEO: "mp4",
                MediaType.AUDIO: "mp3",
                MediaType.IMAGE: "jpg",
                MediaType.TEXT: "txt",
                MediaType.FONT: "ttf",
                MediaType.TEMPLATE: "json"
            }
            ext = extensions.get(media_type, "bin")
            return f"{safe_name}_{timestamp}.{ext}"


class MaterialDownloadManager:
    """素材下载管理器"""

    def __init__(self, storage: MaterialStorage, max_concurrent: int = 5):
        self.storage = storage
        self.max_concurrent = max_concurrent
        self.download_queue: asyncio.Queue = asyncio.Queue()
        self.active_downloads: Dict[str, DownloadRequest] = {}
        self.tag_manager = MaterialTagManager()

        # 下载统计
        self.stats = {
            "total_downloads": 0,
            "successful_downloads": 0,
            "failed_downloads": 0,
            "total_bytes": 0
        }

        self.logger = logging.getLogger(__name__)

    async def download_material(self, request: DownloadRequest) -> DownloadResult:
        """下载单个素材"""
        start_time = asyncio.get_event_loop().time()

        try:
            # 检查是否已存在
            existing_path = self.storage.get_material_path(request.material_id)
            if existing_path:
                self.logger.info(f"Material already exists: {request.material_id}")
                return DownloadResult(
                    success=True,
                    material_id=request.material_id,
                    local_path=existing_path,
                    file_size=os.path.getsize(existing_path)
                )

            # 执行下载
            result = await self._download_file(request)

            # 更新统计
            download_time = asyncio.get_event_loop().time() - start_time
            result.download_time = download_time

            if result.success:
                self.stats["successful_downloads"] += 1
                self.stats["total_bytes"] += result.file_size
            else:
                self.stats["failed_downloads"] += 1

            self.stats["total_downloads"] += 1

            return result

        except Exception as e:
            self.logger.error(f"Download failed for {request.material_id}: {e}")
            return DownloadResult(
                success=False,
                material_id=request.material_id,
                error_message=str(e),
                download_time=asyncio.get_event_loop().time() - start_time
            )

    async def _download_file(self, request: DownloadRequest) -> DownloadResult:
        """执行文件下载"""
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=request.timeout)
        ) as session:

            for attempt in range(request.max_retries):
                try:
                    async with session.get(request.url) as response:
                        if response.status == 200:
                            # 创建临时文件
                            temp_file = tempfile.NamedTemporaryFile(delete=False)
                            temp_path = temp_file.name

                            # 下载内容
                            file_size = 0
                            content_hash = hashlib.md5()

                            async for chunk in response.content.iter_chunked(8192):
                                temp_file.write(chunk)
                                file_size += len(chunk)
                                content_hash.update(chunk)

                            temp_file.close()

                            # 验证文件大小
                            if request.expected_size and file_size != request.expected_size:
                                os.unlink(temp_path)
                                raise ValueError(f"File size mismatch: expected {request.expected_size}, got {file_size}")

                            # 创建元数据
                            metadata = self._create_metadata(request, response, file_size)

                            # 保存到存储系统
                            checksum = content_hash.hexdigest()
                            success = self.storage.save_material(
                                request.material_id,
                                temp_path,
                                metadata,
                                request.url,
                                checksum
                            )

                            # 清理临时文件
                            os.unlink(temp_path)

                            if success:
                                final_path = self.storage.get_material_path(request.material_id)
                                return DownloadResult(
                                    success=True,
                                    material_id=request.material_id,
                                    local_path=final_path,
                                    file_size=file_size,
                                    content_type=response.content_type,
                                    checksum=checksum
                                )
                            else:
                                raise RuntimeError("Failed to save material to storage")

                        else:
                            raise aiohttp.ClientResponseError(
                                None, None, status=response.status,
                                message=f"HTTP {response.status}"
                            )

                except asyncio.TimeoutError:
                    if attempt == request.max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)  # 指数退避

                except Exception as e:
                    if attempt == request.max_retries - 1:
                        raise
                    await asyncio.sleep(2 ** attempt)

    def _create_metadata(self, request: DownloadRequest, response,
                        file_size: int) -> MaterialMetadata:
        """创建素材元数据"""
        # 从URL提取文件名
        parsed_url = urlparse(request.url)
        filename = os.path.basename(parsed_url.path) or f"{request.material_id}.bin"

        # 从响应头获取额外信息
        content_type = response.content_type or "application/octet-stream"

        # 创建基础元数据
        from .material_taxonomy import ContentCategory, QualityLevel, UsageRights

        metadata = MaterialMetadata(
            material_id=request.material_id,
            filename=filename,
            media_type=request.expected_type,
            file_size=file_size,
            primary_category=ContentCategory.LIFESTYLE,  # 默认分类，后续可通过AI分析
            quality_level=QualityLevel.STANDARD,
            usage_rights=UsageRights.FREE,
            source=request.url,
            provider="downloaded"
        )

        # 如果有额外元数据，合并
        if request.metadata:
            for key, value in request.metadata.items():
                if hasattr(metadata, key):
                    setattr(metadata, key, value)

        return metadata

    async def batch_download(self, requests: List[DownloadRequest]) -> List[DownloadResult]:
        """批量下载素材"""
        self.logger.info(f"Starting batch download of {len(requests)} materials")

        # 按优先级排序
        requests.sort(key=lambda x: x.priority, reverse=True)

        # 创建信号量限制并发
        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def download_with_semaphore(req):
            async with semaphore:
                return await self.download_material(req)

        # 并发下载
        tasks = [download_with_semaphore(req) for req in requests]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 处理异常结果
        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                final_results.append(DownloadResult(
                    success=False,
                    material_id=requests[i].material_id,
                    error_message=str(result)
                ))
            else:
                final_results.append(result)

        successful = sum(1 for r in final_results if r.success)
        self.logger.info(f"Batch download completed: {successful}/{len(requests)} successful")

        return final_results

    def get_download_stats(self) -> Dict[str, Any]:
        """获取下载统计"""
        storage_stats = self.storage.get_storage_stats()

        return {
            **self.stats,
            "success_rate": (
                self.stats["successful_downloads"] / max(1, self.stats["total_downloads"])
            ) * 100,
            "storage": storage_stats
        }

    async def verify_downloads(self, material_ids: List[str]) -> Dict[str, bool]:
        """验证下载是否完整"""
        results = {}

        for material_id in material_ids:
            try:
                path = self.storage.get_material_path(material_id)
                if path and os.path.exists(path):
                    # 检查文件完整性
                    metadata = self.storage.get_material_metadata(material_id)
                    if metadata:
                        actual_size = os.path.getsize(path)
                        expected_size = metadata.get('file_size', 0)
                        results[material_id] = (actual_size == expected_size)
                    else:
                        results[material_id] = True  # 文件存在但缺少元数据
                else:
                    results[material_id] = False
            except Exception as e:
                self.logger.error(f"Failed to verify {material_id}: {e}")
                results[material_id] = False

        return results


# 使用示例
async def test_download_system():
    """测试下载系统"""
    # 初始化系统
    storage = MaterialStorage()
    download_manager = MaterialDownloadManager(storage)

    # 创建测试下载请求
    test_requests = [
        DownloadRequest(
            url="https://example.com/video1.mp4",
            material_id="test_video_1",
            expected_type=MediaType.VIDEO,
            priority=3,
            metadata={"description": "测试视频素材"}
        ),
        DownloadRequest(
            url="https://example.com/audio1.mp3",
            material_id="test_audio_1",
            expected_type=MediaType.AUDIO,
            priority=2,
            metadata={"description": "测试音频素材"}
        )
    ]

    # 执行批量下载
    results = await download_manager.batch_download(test_requests)

    # 打印结果
    for result in results:
        print(f"Material: {result.material_id}")
        print(f"Success: {result.success}")
        if result.success:
            print(f"Path: {result.local_path}")
            print(f"Size: {result.file_size} bytes")
        else:
            print(f"Error: {result.error_message}")
        print("-" * 40)

    # 显示统计
    stats = download_manager.get_download_stats()
    print("Download Statistics:")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    asyncio.run(test_download_system())