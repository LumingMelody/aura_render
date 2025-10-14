"""
Material Management API
素材管理接口 - 提供RESTful API用于素材的上传、删除、更新和查询
"""
import os
import asyncio
import json
import hashlib
import mimetypes
import tempfile
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path
import uuid
from dataclasses import asdict

from fastapi import FastAPI, UploadFile, File, HTTPException, Query, Form, Depends
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import uvicorn

from .material_taxonomy import (
    MaterialMetadata, MediaType, ContentCategory, StyleTag,
    QualityLevel, UsageRights, MaterialTagManager
)
from .material_download_manager import MaterialStorage, MaterialDownloadManager
from .enhanced_video_matcher import EnhancedVideoMatcher, MatchingContext


# API模型定义
class MaterialUploadRequest(BaseModel):
    """素材上传请求"""
    description: str = Field(..., description="素材描述")
    category: str = Field(default="lifestyle", description="主分类")
    tags: List[str] = Field(default_factory=list, description="标签列表")
    style: Optional[str] = Field(default=None, description="风格标签")
    quality: str = Field(default="standard", description="质量等级")
    usage_rights: str = Field(default="free", description="使用权限")


class MaterialUpdateRequest(BaseModel):
    """素材更新请求"""
    description: Optional[str] = None
    category: Optional[str] = None
    tags: Optional[List[str]] = None
    style: Optional[str] = None
    quality: Optional[str] = None
    usage_rights: Optional[str] = None


class MaterialSearchRequest(BaseModel):
    """素材搜索请求"""
    description: str = Field(..., description="搜索描述")
    duration: Optional[float] = Field(default=None, description="期望时长")
    category: Optional[str] = Field(default=None, description="分类过滤")
    style: Optional[str] = Field(default=None, description="风格过滤")
    quality: str = Field(default="standard", description="质量要求")
    project_theme: str = Field(default="", description="项目主题")
    target_audience: str = Field(default="", description="目标受众")
    max_results: int = Field(default=10, description="最大结果数")


class MaterialResponse(BaseModel):
    """素材响应"""
    material_id: str
    filename: str
    media_type: str
    file_size: int
    description: str
    category: str
    tags: List[str]
    style: Optional[str]
    quality: str
    usage_rights: str
    duration: Optional[float]
    dimensions: Optional[List[int]]
    url: str
    thumbnail_url: Optional[str]
    created_at: str
    updated_at: str
    view_count: int
    rating: float


class MatchResponse(BaseModel):
    """匹配响应"""
    material_id: str
    local_path: str
    match_score: float
    confidence: float
    match_reasons: List[str]
    material_info: MaterialResponse


class MaterialAPI:
    """素材管理API类"""

    def __init__(self, storage_path: str = "/tmp/aura_render_outputs/materials"):
        self.storage = MaterialStorage(storage_path)
        self.download_manager = MaterialDownloadManager(self.storage)
        self.video_matcher = EnhancedVideoMatcher(self.storage)
        self.tag_manager = MaterialTagManager()

        # 创建FastAPI应用
        self.app = FastAPI(
            title="Material Management API",
            description="Aura Render 素材管理接口",
            version="1.0.0"
        )

        # 注册路由
        self._register_routes()

    def _register_routes(self):
        """注册API路由"""

        @self.app.post("/materials/upload", response_model=Dict[str, Any])
        async def upload_material(
            file: UploadFile = File(...),
            request: str = Form(..., description="JSON格式的MaterialUploadRequest")
        ):
            """上传素材文件"""
            try:
                # 解析请求数据
                upload_req = MaterialUploadRequest.parse_raw(request)
                return await self._handle_upload(file, upload_req)

            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Upload failed: {str(e)}")

        @self.app.get("/materials/{material_id}", response_model=MaterialResponse)
        async def get_material(material_id: str):
            """获取素材信息"""
            try:
                return await self._handle_get_material(material_id)
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Material not found: {str(e)}")

        @self.app.put("/materials/{material_id}", response_model=MaterialResponse)
        async def update_material(material_id: str, request: MaterialUpdateRequest):
            """更新素材信息"""
            try:
                return await self._handle_update_material(material_id, request)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Update failed: {str(e)}")

        @self.app.delete("/materials/{material_id}")
        async def delete_material(material_id: str):
            """删除素材"""
            try:
                return await self._handle_delete_material(material_id)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Delete failed: {str(e)}")

        @self.app.get("/materials", response_model=List[MaterialResponse])
        async def list_materials(
            media_type: Optional[str] = Query(default=None, description="媒体类型"),
            category: Optional[str] = Query(default=None, description="分类过滤"),
            limit: int = Query(default=50, description="结果数量限制"),
            offset: int = Query(default=0, description="结果偏移量")
        ):
            """列出素材"""
            try:
                return await self._handle_list_materials(media_type, category, limit, offset)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"List failed: {str(e)}")

        @self.app.post("/materials/search", response_model=List[MatchResponse])
        async def search_materials(request: MaterialSearchRequest):
            """智能搜索素材"""
            try:
                return await self._handle_search_materials(request)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Search failed: {str(e)}")

        @self.app.get("/materials/{material_id}/download")
        async def download_material(material_id: str):
            """下载素材文件"""
            try:
                return await self._handle_download_material(material_id)
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Download failed: {str(e)}")

        @self.app.get("/statistics")
        async def get_statistics():
            """获取系统统计"""
            try:
                return await self._handle_get_statistics()
            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Statistics failed: {str(e)}")

        @self.app.post("/materials/batch-delete")
        async def batch_delete_materials(material_ids: List[str]):
            """批量删除素材"""
            try:
                return await self._handle_batch_delete(material_ids)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Batch delete failed: {str(e)}")

        @self.app.get("/health")
        async def health_check():
            """健康检查"""
            return {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "storage_path": str(self.storage.base_path),
                "version": "1.0.0"
            }

    async def _handle_upload(self, file: UploadFile, request: MaterialUploadRequest) -> Dict[str, Any]:
        """处理文件上传"""
        # 生成材料ID
        material_id = str(uuid.uuid4())

        # 验证文件类型
        content_type = file.content_type or mimetypes.guess_type(file.filename or "")[0]
        media_type = self._detect_media_type(file.filename, content_type)

        # 读取文件内容
        content = await file.read()
        file_size = len(content)

        # 创建临时文件
        temp_file = tempfile.NamedTemporaryFile(delete=False)
        temp_file.write(content)
        temp_file.close()

        try:
            # 创建元数据
            metadata = MaterialMetadata(
                material_id=material_id,
                filename=file.filename or f"{material_id}.bin",
                media_type=media_type,
                file_size=file_size,
                primary_category=ContentCategory(request.category),
                quality_level=QualityLevel(request.quality),
                usage_rights=UsageRights(request.usage_rights),
                keywords=request.tags,
                source="user_upload",
                provider="api"
            )

            # 添加风格标签
            if request.style:
                try:
                    metadata.style_tags = [StyleTag(request.style)]
                except ValueError:
                    pass  # 忽略无效风格标签

            # 使用标签管理器增强元数据
            metadata = self.tag_manager.add_material_tags(
                material_id, metadata, request.description
            )

            # 保存到存储系统
            checksum = hashlib.md5(content).hexdigest()
            success = self.storage.save_material(
                material_id, temp_file.name, metadata,
                original_url="user_upload", checksum=checksum
            )

            if not success:
                raise RuntimeError("Failed to save material")

            # 返回成功响应
            return {
                "success": True,
                "material_id": material_id,
                "filename": metadata.filename,
                "file_size": file_size,
                "media_type": media_type.value,
                "message": "Material uploaded successfully",
                "url": f"/materials/{material_id}/download"
            }

        finally:
            # 清理临时文件
            if os.path.exists(temp_file.name):
                os.unlink(temp_file.name)

    async def _handle_get_material(self, material_id: str) -> MaterialResponse:
        """处理获取素材"""
        material_data = self.storage.get_material_metadata(material_id)

        if not material_data:
            raise ValueError("Material not found")

        # 构造响应
        parsed_metadata = json.loads(material_data.get('metadata', '{}'))

        return MaterialResponse(
            material_id=material_id,
            filename=material_data['filename'],
            media_type=material_data['media_type'],
            file_size=material_data['file_size'],
            description=parsed_metadata.get('description', ''),
            category=parsed_metadata.get('primary_category', 'lifestyle'),
            tags=parsed_metadata.get('keywords', []),
            style=parsed_metadata.get('style_tags', [None])[0] if parsed_metadata.get('style_tags') else None,
            quality=parsed_metadata.get('quality_level', 'standard'),
            usage_rights=parsed_metadata.get('usage_rights', 'free'),
            duration=parsed_metadata.get('duration'),
            dimensions=parsed_metadata.get('dimensions'),
            url=f"/materials/{material_id}/download",
            thumbnail_url=None,  # TODO: 实现缩略图生成
            created_at=material_data['created_at'],
            updated_at=parsed_metadata.get('updated_at', material_data['created_at']),
            view_count=material_data.get('download_count', 0),
            rating=parsed_metadata.get('rating', 0.0)
        )

    async def _handle_update_material(self, material_id: str,
                                    request: MaterialUpdateRequest) -> MaterialResponse:
        """处理更新素材"""
        # 获取现有素材
        material_data = self.storage.get_material_metadata(material_id)

        if not material_data:
            raise ValueError("Material not found")

        # 解析现有元数据
        parsed_metadata = json.loads(material_data.get('metadata', '{}'))

        # 更新字段
        updates = {}
        if request.description is not None:
            updates['description'] = request.description
        if request.category is not None:
            updates['primary_category'] = request.category
        if request.tags is not None:
            updates['keywords'] = request.tags
        if request.style is not None:
            updates['style_tags'] = [request.style]
        if request.quality is not None:
            updates['quality_level'] = request.quality
        if request.usage_rights is not None:
            updates['usage_rights'] = request.usage_rights

        # 合并更新
        parsed_metadata.update(updates)
        parsed_metadata['updated_at'] = datetime.now().isoformat()

        # 保存更新 - 这里需要实现存储系统的更新方法
        # 暂时通过重新保存实现
        updated_metadata_json = json.dumps(parsed_metadata)

        # 直接更新数据库 (简化实现)
        import sqlite3
        with sqlite3.connect(self.storage.db_path) as conn:
            conn.execute(
                "UPDATE materials SET metadata = ? WHERE material_id = ?",
                (updated_metadata_json, material_id)
            )

        # 返回更新后的素材信息
        return await self._handle_get_material(material_id)

    async def _handle_delete_material(self, material_id: str) -> Dict[str, Any]:
        """处理删除素材"""
        success = self.storage.delete_material(material_id)

        if not success:
            raise ValueError("Failed to delete material")

        return {
            "success": True,
            "material_id": material_id,
            "message": "Material deleted successfully"
        }

    async def _handle_list_materials(self, media_type: Optional[str],
                                   category: Optional[str],
                                   limit: int, offset: int) -> List[MaterialResponse]:
        """处理列出素材"""
        # 转换媒体类型
        media_type_enum = None
        if media_type:
            try:
                media_type_enum = MediaType(media_type)
            except ValueError:
                raise ValueError(f"Invalid media type: {media_type}")

        # 获取素材列表
        materials = self.storage.list_materials(
            media_type=media_type_enum,
            limit=limit + offset  # 简化实现，实际应该在SQL层面处理偏移
        )

        # 过滤和转换结果
        results = []
        count = 0
        for material_data in materials:
            # 应用偏移
            if count < offset:
                count += 1
                continue

            # 应用分类过滤
            if category:
                parsed_metadata = material_data.get('parsed_metadata', {})
                if parsed_metadata.get('primary_category') != category:
                    continue

            try:
                response = MaterialResponse(
                    material_id=material_data['material_id'],
                    filename=material_data['filename'],
                    media_type=material_data['media_type'],
                    file_size=material_data['file_size'],
                    description=material_data.get('parsed_metadata', {}).get('description', ''),
                    category=material_data.get('parsed_metadata', {}).get('primary_category', 'lifestyle'),
                    tags=material_data.get('parsed_metadata', {}).get('keywords', []),
                    style=None,  # 简化处理
                    quality=material_data.get('parsed_metadata', {}).get('quality_level', 'standard'),
                    usage_rights=material_data.get('parsed_metadata', {}).get('usage_rights', 'free'),
                    duration=material_data.get('parsed_metadata', {}).get('duration'),
                    dimensions=material_data.get('parsed_metadata', {}).get('dimensions'),
                    url=f"/materials/{material_data['material_id']}/download",
                    thumbnail_url=None,
                    created_at=material_data['created_at'],
                    updated_at=material_data.get('parsed_metadata', {}).get('updated_at', material_data['created_at']),
                    view_count=material_data.get('download_count', 0),
                    rating=material_data.get('parsed_metadata', {}).get('rating', 0.0)
                )
                results.append(response)

                # 限制结果数量
                if len(results) >= limit:
                    break

            except Exception as e:
                print(f"Error processing material {material_data.get('material_id', 'unknown')}: {e}")
                continue

        return results

    async def _handle_search_materials(self, request: MaterialSearchRequest) -> List[MatchResponse]:
        """处理智能搜索"""
        # 构造匹配上下文
        context = MatchingContext(
            shot_description=request.description,
            shot_duration=request.duration or 5.0,
            content_category=ContentCategory(request.category) if request.category else None,
            style_preferences=[StyleTag(request.style)] if request.style else [],
            quality_requirement=request.quality,
            project_theme=request.project_theme,
            target_audience=request.target_audience
        )

        # 执行智能匹配
        match_results = await self.video_matcher.find_best_matches(
            context, max_results=request.max_results
        )

        # 转换为响应格式
        responses = []
        for match_result in match_results:
            try:
                # 获取素材详细信息
                material_response = await self._handle_get_material(match_result.material_id)

                response = MatchResponse(
                    material_id=match_result.material_id,
                    local_path=match_result.local_path,
                    match_score=match_result.match_score,
                    confidence=match_result.confidence,
                    match_reasons=match_result.match_reasons,
                    material_info=material_response
                )
                responses.append(response)

            except Exception as e:
                print(f"Error creating match response for {match_result.material_id}: {e}")
                continue

        return responses

    async def _handle_download_material(self, material_id: str) -> FileResponse:
        """处理下载素材"""
        local_path = self.storage.get_material_path(material_id)

        if not local_path or not os.path.exists(local_path):
            raise ValueError("Material file not found")

        # 获取文件信息
        material_data = self.storage.get_material_metadata(material_id)
        filename = material_data['filename'] if material_data else f"{material_id}.bin"

        return FileResponse(
            path=local_path,
            filename=filename,
            media_type=mimetypes.guess_type(local_path)[0] or 'application/octet-stream'
        )

    async def _handle_get_statistics(self) -> Dict[str, Any]:
        """处理获取统计"""
        storage_stats = self.storage.get_storage_stats()
        download_stats = self.download_manager.get_download_stats()
        match_stats = self.video_matcher.get_match_statistics()

        return {
            "storage": storage_stats,
            "downloads": download_stats,
            "matching": match_stats,
            "timestamp": datetime.now().isoformat()
        }

    async def _handle_batch_delete(self, material_ids: List[str]) -> Dict[str, Any]:
        """处理批量删除"""
        results = {
            "success": [],
            "failed": [],
            "total": len(material_ids)
        }

        for material_id in material_ids:
            try:
                success = self.storage.delete_material(material_id)
                if success:
                    results["success"].append(material_id)
                else:
                    results["failed"].append(material_id)
            except Exception as e:
                results["failed"].append(material_id)
                print(f"Failed to delete material {material_id}: {e}")

        return {
            "deleted_count": len(results["success"]),
            "failed_count": len(results["failed"]),
            "failed_materials": results["failed"],
            "message": f"Deleted {len(results['success'])} materials successfully"
        }

    def _detect_media_type(self, filename: Optional[str], content_type: Optional[str]) -> MediaType:
        """检测媒体类型"""
        if not filename and not content_type:
            return MediaType.IMAGE  # 默认类型

        # 从文件名检测
        if filename:
            ext = Path(filename).suffix.lower()
            if ext in ['.mp4', '.avi', '.mov', '.webm', '.mkv']:
                return MediaType.VIDEO
            elif ext in ['.mp3', '.wav', '.ogg', '.aac', '.m4a']:
                return MediaType.AUDIO
            elif ext in ['.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg']:
                return MediaType.IMAGE
            elif ext in ['.txt', '.md', '.json', '.xml']:
                return MediaType.TEXT
            elif ext in ['.ttf', '.otf', '.woff', '.woff2']:
                return MediaType.FONT

        # 从MIME类型检测
        if content_type:
            if content_type.startswith('video/'):
                return MediaType.VIDEO
            elif content_type.startswith('audio/'):
                return MediaType.AUDIO
            elif content_type.startswith('image/'):
                return MediaType.IMAGE
            elif content_type.startswith('text/'):
                return MediaType.TEXT

        return MediaType.IMAGE  # 默认类型

    def run(self, host: str = "0.0.0.0", port: int = 8000):
        """运行API服务器"""
        uvicorn.run(self.app, host=host, port=port)


# 使用示例
if __name__ == "__main__":
    print("🚀 启动 Aura Render 素材管理API服务")

    # 创建API实例
    api = MaterialAPI()

    print("📋 API端点:")
    print("  POST /materials/upload - 上传素材")
    print("  GET /materials/{id} - 获取素材信息")
    print("  PUT /materials/{id} - 更新素材")
    print("  DELETE /materials/{id} - 删除素材")
    print("  GET /materials - 列出素材")
    print("  POST /materials/search - 智能搜索")
    print("  GET /materials/{id}/download - 下载素材")
    print("  GET /statistics - 系统统计")
    print("  GET /health - 健康检查")
    print()
    print("🌐 服务地址: http://localhost:8000")
    print("📚 API文档: http://localhost:8000/docs")

    # 启动服务器
    api.run()