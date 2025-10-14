"""
Materials API Endpoints

FastAPI endpoints for material search and management.
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query
from pydantic import BaseModel, Field

from materials import MaterialManager, MaterialType, MaterialSearchQuery
from config import get_settings


# Pydantic models for API
class MaterialSearchRequest(BaseModel):
    """Material search request model"""
    keywords: List[str] = Field(..., description="搜索关键词", min_items=1, max_items=10)
    material_type: Optional[str] = Field(None, description="素材类型: video, audio, image")
    max_duration: Optional[float] = Field(None, description="最大时长(秒)", gt=0)
    min_duration: Optional[float] = Field(None, description="最小时长(秒)", gt=0)
    preferred_aspect_ratio: Optional[str] = Field(None, description="首选宽高比: 16:9, 9:16, 1:1")
    min_quality: float = Field(0.0, description="最小质量分数", ge=0.0, le=1.0)
    limit: int = Field(10, description="结果数量限制", ge=1, le=50)
    offset: int = Field(0, description="结果偏移", ge=0)
    providers: Optional[List[str]] = Field(None, description="指定搜索提供商")
    
    class Config:
        json_schema_extra = {
            "example": {
                "keywords": ["科技", "创新", "未来"],
                "material_type": "video",
                "max_duration": 60,
                "preferred_aspect_ratio": "16:9",
                "min_quality": 0.5,
                "limit": 10
            }
        }


class MaterialInfoResponse(BaseModel):
    """Material information response"""
    material_id: str
    material_type: str
    url: str
    thumbnail_url: Optional[str]
    title: str
    description: Optional[str]
    tags: List[str]
    duration: Optional[float]
    width: Optional[int]
    height: Optional[int]
    format: Optional[str]
    license: str
    source: str
    author: Optional[str]
    relevance_score: float
    quality_score: float
    popularity_score: float


class MaterialSearchResponse(BaseModel):
    """Material search response"""
    query: Dict[str, Any]
    results: List[MaterialInfoResponse]
    total_count: int
    search_time: float
    providers_used: List[str]
    timestamp: datetime


class SmartSearchRequest(BaseModel):
    """Smart search request with context"""
    keywords: List[str] = Field(..., description="搜索关键词")
    material_type: str = Field(..., description="素材类型: video, audio, image")
    context: Optional[Dict[str, Any]] = Field(None, description="搜索上下文")
    max_results: int = Field(10, description="最大结果数", ge=1, le=20)
    
    class Config:
        json_schema_extra = {
            "example": {
                "keywords": ["办公室", "商务", "专业"],
                "material_type": "video",
                "context": {
                    "emotion": "professional",
                    "max_duration": 30,
                    "aspect_ratio": "16:9"
                },
                "max_results": 5
            }
        }


# Create router
materials_router = APIRouter(prefix="/materials", tags=["Materials"])

# Global material manager instance
_material_manager = None


def get_material_manager() -> MaterialManager:
    """Get or create material manager instance"""
    global _material_manager
    if _material_manager is None:
        settings = get_settings()
        
        # Build provider configs
        provider_configs = {}
        
        # External provider config (prioritized)
        provider_configs["external"] = {
            "base_url": settings.materials.external_base_url,
            "api_key": settings.materials.external_api_key
        }
        
        # Pexels config
        if settings.materials.pexels_api_key:
            provider_configs["pexels"] = {
                "api_key": settings.materials.pexels_api_key
            }
        
        # Pixabay config
        if settings.materials.pixabay_api_key:
            provider_configs["pixabay"] = {
                "api_key": settings.materials.pixabay_api_key
            }
        
        # Unsplash config
        if settings.materials.unsplash_access_key:
            provider_configs["unsplash"] = {
                "access_key": settings.materials.unsplash_access_key
            }
        
        # Freesound config
        if settings.materials.freesound_api_key:
            provider_configs["freesound"] = {
                "api_key": settings.materials.freesound_api_key
            }
        
        config = {
            "providers": provider_configs,
            "max_concurrent_requests": settings.materials.max_concurrent_requests,
            "request_timeout": settings.materials.request_timeout
        }
        
        _material_manager = MaterialManager(config)
    
    return _material_manager


def _convert_search_result(result) -> MaterialInfoResponse:
    """Convert MaterialSearchResult to API response format"""
    return MaterialInfoResponse(
        material_id=result.material_id,
        material_type=result.material_type.value,
        url=result.url,
        thumbnail_url=result.thumbnail_url,
        title=result.metadata.title,
        description=result.metadata.description,
        tags=result.metadata.tags,
        duration=result.metadata.duration,
        width=result.metadata.width,
        height=result.metadata.height,
        format=result.metadata.format.value if result.metadata.format else None,
        license=result.metadata.license.value,
        source=result.metadata.source,
        author=result.metadata.author,
        relevance_score=result.relevance_score,
        quality_score=result.quality_score,
        popularity_score=result.popularity_score
    )


@materials_router.get("/providers")
async def list_providers():
    """List available material providers"""
    manager = get_material_manager()
    
    return {
        "providers": manager.get_provider_info(),
        "available_by_type": {
            "video": manager.get_available_providers(MaterialType.VIDEO),
            "audio": manager.get_available_providers(MaterialType.AUDIO),
            "image": manager.get_available_providers(MaterialType.IMAGE)
        },
        "timestamp": datetime.now()
    }


@materials_router.post("/search", response_model=MaterialSearchResponse)
async def search_materials(request: MaterialSearchRequest):
    """Search for materials across multiple providers"""
    try:
        manager = get_material_manager()
        
        # Convert request to search query
        material_type = None
        if request.material_type:
            material_type = MaterialType(request.material_type.lower())
        
        query = MaterialSearchQuery(
            keywords=request.keywords,
            material_type=material_type,
            max_duration=request.max_duration,
            min_duration=request.min_duration,
            preferred_aspect_ratio=request.preferred_aspect_ratio,
            min_quality=request.min_quality,
            limit=request.limit,
            offset=request.offset
        )
        
        # Perform search
        response = await manager.search_and_aggregate(
            query=query,
            providers=request.providers,
            max_concurrent=3,
            sort_by="combined"
        )
        
        # Convert results
        api_results = [_convert_search_result(result) for result in response.results]
        
        # Extract providers used
        providers_used = []
        if "providers" in response.provider_info:
            providers_used = [p["name"] for p in response.provider_info["providers"]]
        
        return MaterialSearchResponse(
            query=request.model_dump(),
            results=api_results,
            total_count=response.total_count,
            search_time=response.search_time,
            providers_used=providers_used,
            timestamp=datetime.now()
        )
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search failed: {e}")


@materials_router.post("/smart-search", response_model=List[MaterialInfoResponse])
async def smart_search_materials(request: SmartSearchRequest):
    """Smart material search with context awareness"""
    try:
        manager = get_material_manager()
        
        # Convert material type
        material_type = MaterialType(request.material_type.lower())
        
        # Perform smart search
        results = await manager.smart_search(
            keywords=request.keywords,
            material_type=material_type,
            context=request.context,
            max_results=request.max_results
        )
        
        # Convert results
        return [_convert_search_result(result) for result in results]
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid request: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Smart search failed: {e}")


@materials_router.get("/material/{material_id}", response_model=MaterialInfoResponse)
async def get_material_info(material_id: str):
    """Get detailed information about a specific material"""
    try:
        manager = get_material_manager()
        
        result = await manager.get_material_info(material_id)
        
        if not result:
            raise HTTPException(status_code=404, detail="Material not found")
        
        return _convert_search_result(result)
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get material info: {e}")


@materials_router.get("/search/suggestions")
async def get_search_suggestions(
    query: str = Query(..., description="搜索查询"),
    material_type: Optional[str] = Query(None, description="素材类型"),
    limit: int = Query(10, description="建议数量", ge=1, le=20)
):
    """Get search suggestions based on query"""
    # This is a simplified implementation - in production you might use
    # a dedicated search suggestion service or analyze popular searches
    
    base_suggestions = {
        "video": [
            "科技创新", "商务办公", "自然风景", "城市生活", "人物肖像",
            "工业制造", "医疗健康", "教育培训", "金融理财", "餐饮美食"
        ],
        "audio": [
            "背景音乐", "音效", "环境音", "乐器演奏", "人声解说",
            "自然声音", "城市噪音", "电子音乐", "古典音乐", "流行音乐"
        ],
        "image": [
            "商务图片", "科技图像", "自然摄影", "人物写真", "产品展示",
            "抽象背景", "纹理素材", "图标设计", "插画艺术", "建筑景观"
        ]
    }
    
    suggestions = []
    query_lower = query.lower()
    
    # Get suggestions for specific type or all types
    if material_type:
        type_suggestions = base_suggestions.get(material_type, [])
        suggestions.extend([s for s in type_suggestions if query_lower in s.lower()])
    else:
        for type_name, type_suggestions in base_suggestions.items():
            suggestions.extend([s for s in type_suggestions if query_lower in s.lower()])
    
    # If no matches, return popular suggestions for the type
    if not suggestions:
        if material_type and material_type in base_suggestions:
            suggestions = base_suggestions[material_type][:limit]
        else:
            suggestions = base_suggestions["video"][:limit]
    
    return {
        "query": query,
        "suggestions": suggestions[:limit],
        "timestamp": datetime.now()
    }


@materials_router.get("/by-tags")
async def get_materials_by_tags(
    tags: str = Query(..., description="标签列表，逗号分隔，如：科技,创新,商务"),
    material_type: Optional[str] = Query(None, description="素材类型: video, audio, image"),
    limit: int = Query(10, description="结果数量", ge=1, le=50)
):
    """根据标签直接查询素材 - 简化版接口"""
    try:
        # 解析标签
        tag_list = [tag.strip() for tag in tags.split(",") if tag.strip()]
        if not tag_list:
            raise HTTPException(status_code=400, detail="至少需要提供一个标签")
        
        manager = get_material_manager()
        
        # 转换材料类型
        mat_type = None
        if material_type:
            mat_type = MaterialType(material_type.lower())
        
        # 使用smart_search进行查询
        results = await manager.smart_search(
            keywords=tag_list,
            material_type=mat_type or MaterialType.VIDEO,  # 默认视频
            context={"tags_only": True},  # 标识这是纯标签搜索
            max_results=limit
        )
        
        # 简化返回格式
        simplified_results = []
        for result in results:
            simplified_results.append({
                "id": result.material_id,
                "type": result.material_type.value,
                "title": result.metadata.title,
                "url": result.url,
                "thumbnail": result.thumbnail_url,
                "tags": result.metadata.tags,
                "duration": result.metadata.duration,
                "size": f"{result.metadata.width}x{result.metadata.height}" if result.metadata.width else None,
                "source": result.metadata.source,
                "quality": round(result.quality_score, 2),
                "relevance": round(result.relevance_score, 2)
            })
        
        return {
            "query_tags": tag_list,
            "material_type": material_type or "video",
            "results": simplified_results,
            "count": len(simplified_results),
            "timestamp": datetime.now()
        }
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"参数错误: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"查询失败: {e}")


@materials_router.get("/stats")
async def get_material_stats():
    """Get material system statistics"""
    manager = get_material_manager()
    
    provider_info = manager.get_provider_info()
    
    # Count available providers by type
    stats = {
        "providers": {
            "total": len(provider_info),
            "available": len([p for p in provider_info.values() if p["is_available"]]),
            "by_type": {
                "video": len(manager.get_available_providers(MaterialType.VIDEO)),
                "audio": len(manager.get_available_providers(MaterialType.AUDIO)),
                "image": len(manager.get_available_providers(MaterialType.IMAGE))
            }
        },
        "provider_details": provider_info,
        "timestamp": datetime.now()
    }
    
    return stats