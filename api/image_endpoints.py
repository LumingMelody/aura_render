"""
Image Generation API Endpoints

FastAPI endpoints for AI image generation:
- Single image generation with various styles
- Batch scene generation for video storyboards
- Provider management and testing
- Image gallery and management
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, status, Depends, Query, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from content_generator.image_generation_service import (
    get_image_generation_service,
    ImageProvider,
    ImageStyle,
    ImageSize,
    ImageGenerationRequest,
    GeneratedImage
)
from config import Settings, get_settings
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/images", tags=["Image Generation"])

# =============================
# Pydantic Models
# =============================

class ImageGenerationRequestModel(BaseModel):
    """Image generation request model"""
    prompt: str = Field(..., description="图像生成提示词", min_length=1, max_length=1000)
    style: str = Field("photorealistic", description="图像风格")
    size: str = Field("1024x1024", description="图像尺寸")
    quality: str = Field("standard", description="图像质量: standard, hd")
    provider: Optional[str] = Field(None, description="指定提供商: openai_dalle, stability_ai")
    negative_prompt: Optional[str] = Field(None, description="负面提示词")
    seed: Optional[int] = Field(None, description="随机种子")
    cfg_scale: float = Field(7.0, description="CFG缩放因子", ge=1.0, le=20.0)
    steps: int = Field(20, description="生成步数", ge=10, le=100)
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "A futuristic cityscape at sunset, cyberpunk style with neon lights",
                "style": "cinematic",
                "size": "1792x1024",
                "quality": "hd",
                "provider": "openai_dalle",
                "negative_prompt": "blurry, low quality",
                "cfg_scale": 7.5,
                "steps": 25
            }
        }

class BatchImageGenerationRequest(BaseModel):
    """Batch image generation request"""
    prompts: List[str] = Field(..., description="批量提示词列表", min_items=1, max_items=10)
    style: str = Field("cinematic", description="统一图像风格")
    size: str = Field("1792x1024", description="统一图像尺寸")
    quality: str = Field("standard", description="图像质量")
    provider: Optional[str] = Field(None, description="指定提供商")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompts": [
                    "Opening scene: Modern office building exterior",
                    "Product showcase: Sleek smartphone on desk",
                    "Team collaboration: People working together",
                    "Technology concept: Abstract digital network"
                ],
                "style": "corporate",
                "size": "1920x1080",
                "quality": "hd"
            }
        }

class GeneratedImageResponse(BaseModel):
    """Generated image response model"""
    image_id: str
    image_path: str
    thumbnail_path: Optional[str]
    prompt: str
    provider: str
    style: str
    size: str
    generation_time: float
    cost: float
    timestamp: datetime
    metadata: Dict[str, Any]

class BatchGenerationResponse(BaseModel):
    """Batch generation response model"""
    batch_id: str
    total_images: int
    successful_generations: int
    failed_generations: int
    total_cost: float
    total_time: float
    images: List[Optional[GeneratedImageResponse]]
    timestamp: datetime

class ImageProvidersResponse(BaseModel):
    """Available providers response"""
    providers: List[Dict[str, Any]]
    default_provider: Optional[str]
    timestamp: datetime

# =============================
# Utility Functions
# =============================

def _convert_style(style_str: str) -> ImageStyle:
    """Convert string to ImageStyle enum"""
    style_mapping = {
        "photorealistic": ImageStyle.PHOTOREALISTIC,
        "artistic": ImageStyle.ARTISTIC,
        "cartoon": ImageStyle.CARTOON,
        "cinematic": ImageStyle.CINEMATIC,
        "minimalist": ImageStyle.MINIMALIST,
        "vintage": ImageStyle.VINTAGE,
        "futuristic": ImageStyle.FUTURISTIC,
        "corporate": ImageStyle.CORPORATE
    }
    return style_mapping.get(style_str.lower(), ImageStyle.PHOTOREALISTIC)

def _convert_size(size_str: str) -> ImageSize:
    """Convert string to ImageSize enum"""
    size_mapping = {
        "1024x1024": ImageSize.SQUARE_1024,
        "1792x1024": ImageSize.LANDSCAPE_1792_1024,
        "1024x1792": ImageSize.PORTRAIT_1024_1792,
        "1920x1080": ImageSize.HD_1920_1080,
        "1080x1920": ImageSize.VERTICAL_1080_1920
    }
    return size_mapping.get(size_str, ImageSize.SQUARE_1024)

def _convert_provider(provider_str: Optional[str]) -> Optional[ImageProvider]:
    """Convert string to ImageProvider enum"""
    if not provider_str:
        return None
    
    provider_mapping = {
        "openai_dalle": ImageProvider.OPENAI_DALLE,
        "stability_ai": ImageProvider.STABILITY_AI,
        "midjourney": ImageProvider.MIDJOURNEY,
        "local_sd": ImageProvider.LOCAL_SD
    }
    return provider_mapping.get(provider_str.lower())

def _generated_image_to_response(result: GeneratedImage, image_id: str) -> GeneratedImageResponse:
    """Convert GeneratedImage to response model"""
    return GeneratedImageResponse(
        image_id=image_id,
        image_path=result.image_path,
        thumbnail_path=result.thumbnail_path,
        prompt=result.prompt,
        provider=result.provider.value,
        style=result.style.value,
        size=result.size,
        generation_time=result.generation_time,
        cost=result.cost,
        timestamp=datetime.utcnow(),
        metadata=result.metadata
    )

# =============================
# Image Generation Endpoints
# =============================

@router.post("/generate", response_model=GeneratedImageResponse)
async def generate_single_image(
    request: ImageGenerationRequestModel,
    settings: Settings = Depends(get_settings)
):
    """生成单张图像"""
    
    try:
        # Convert request parameters
        style = _convert_style(request.style)
        size = _convert_size(request.size)
        provider = _convert_provider(request.provider)
        
        # Get image generation service
        image_service = get_image_generation_service(settings)
        
        # Generate image
        result = await image_service.generate_image(
            prompt=request.prompt,
            style=style,
            size=size,
            provider=provider,
            quality=request.quality,
            negative_prompt=request.negative_prompt,
            seed=request.seed,
            cfg_scale=request.cfg_scale,
            steps=request.steps
        )
        
        if not result:
            raise HTTPException(
                status_code=500,
                detail="图像生成失败，未返回结果"
            )
        
        # Generate image ID
        image_id = f"img_{int(datetime.utcnow().timestamp())}_{hash(request.prompt) % 10000}"
        
        return _generated_image_to_response(result, image_id)
        
    except Exception as e:
        logger.error(f"Failed to generate image: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"图像生成失败: {str(e)}"
        )

@router.post("/generate/batch", response_model=BatchGenerationResponse)
async def generate_batch_images(
    request: BatchImageGenerationRequest,
    settings: Settings = Depends(get_settings)
):
    """批量生成图像（适用于视频分镜）"""
    
    try:
        # Convert parameters
        style = _convert_style(request.style)
        size = _convert_size(request.size)
        provider = _convert_provider(request.provider)
        
        # Get image generation service
        image_service = get_image_generation_service(settings)
        
        # Generate batch ID
        batch_id = f"batch_{int(datetime.utcnow().timestamp())}"
        
        logger.info(f"Starting batch generation {batch_id} with {len(request.prompts)} images")
        
        # Generate scene images
        results = await image_service.generate_scene_images(
            scene_descriptions=request.prompts,
            style=style,
            size=size
        )
        
        # Process results
        images = []
        total_cost = 0.0
        total_time = 0.0
        successful_count = 0
        
        for i, result in enumerate(results):
            if result:
                image_id = f"{batch_id}_img_{i+1:03d}"
                response_image = _generated_image_to_response(result, image_id)
                images.append(response_image)
                total_cost += result.cost
                total_time += result.generation_time
                successful_count += 1
            else:
                images.append(None)
        
        return BatchGenerationResponse(
            batch_id=batch_id,
            total_images=len(request.prompts),
            successful_generations=successful_count,
            failed_generations=len(request.prompts) - successful_count,
            total_cost=total_cost,
            total_time=total_time,
            images=images,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to generate batch images: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"批量图像生成失败: {str(e)}"
        )

@router.get("/scene-generation/{video_theme}")
async def generate_scene_images_for_theme(
    video_theme: str,
    num_scenes: int = Query(4, description="场景数量", ge=1, le=10),
    style: str = Query("cinematic", description="图像风格"),
    size: str = Query("1792x1024", description="图像尺寸"),
    settings: Settings = Depends(get_settings)
):
    """为特定视频主题生成场景图像"""
    
    try:
        # Create scene descriptions based on theme
        scene_templates = {
            "产品宣传": [
                f"{video_theme} product showcase in modern setting",
                f"Close-up detail shot of {video_theme} key features",
                f"People using {video_theme} in real environment",
                f"Brand logo and {video_theme} final presentation"
            ],
            "科技介绍": [
                f"Futuristic technology concept for {video_theme}",
                f"Digital interface and {video_theme} interaction",
                f"Innovation lab with {video_theme} development",
                f"Global network connecting {video_theme} users"
            ],
            "企业展示": [
                f"Modern office building representing {video_theme}",
                f"Professional team working on {video_theme}",
                f"Conference room presentation about {video_theme}",
                f"Success celebration for {video_theme} achievement"
            ]
        }
        
        # Get scene descriptions
        base_scenes = scene_templates.get(video_theme, [
            f"Opening scene showcasing {video_theme}",
            f"Main content demonstration of {video_theme}",
            f"Supporting details about {video_theme}",
            f"Conclusion and call-to-action for {video_theme}"
        ])
        
        # Adjust number of scenes
        if num_scenes <= len(base_scenes):
            scenes = base_scenes[:num_scenes]
        else:
            scenes = base_scenes + [f"Additional scene {i} for {video_theme}" 
                                   for i in range(len(base_scenes) + 1, num_scenes + 1)]
        
        # Generate batch request
        batch_request = BatchImageGenerationRequest(
            prompts=scenes,
            style=style,
            size=size,
            quality="standard"
        )
        
        return await generate_batch_images(batch_request, settings)
        
    except Exception as e:
        logger.error(f"Failed to generate theme scenes: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"主题场景生成失败: {str(e)}"
        )

# =============================
# Provider Management Endpoints
# =============================

@router.get("/providers", response_model=ImageProvidersResponse)
async def get_available_providers(
    settings: Settings = Depends(get_settings)
):
    """获取可用的图像生成提供商"""
    
    try:
        image_service = get_image_generation_service(settings)
        available_providers = image_service.get_available_providers()
        
        providers = []
        for provider in available_providers:
            provider_info = image_service.get_provider_info(provider)
            providers.append({
                "id": provider.value,
                "name": provider.value.replace('_', ' ').title(),
                "available": provider_info.get("available", False),
                "supported_sizes": provider_info.get("supported_sizes", [])
            })
        
        default_provider = available_providers[0].value if available_providers else None
        
        return ImageProvidersResponse(
            providers=providers,
            default_provider=default_provider,
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to get providers: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取提供商信息失败: {str(e)}"
        )

@router.post("/providers/{provider_name}/test")
async def test_provider(
    provider_name: str,
    settings: Settings = Depends(get_settings)
):
    """测试指定的图像生成提供商"""
    
    try:
        provider = _convert_provider(provider_name)
        if not provider:
            raise HTTPException(
                status_code=400,
                detail=f"未知的提供商: {provider_name}"
            )
        
        image_service = get_image_generation_service(settings)
        result = await image_service.test_generation(provider)
        
        return {
            "provider": provider_name,
            "test_result": result,
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Provider test failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"提供商测试失败: {str(e)}"
        )

# =============================
# Image File Endpoints
# =============================

@router.get("/file/{image_path:path}")
async def get_image_file(image_path: str):
    """获取生成的图像文件"""
    
    try:
        # Resolve full path (security: only allow files in temp directory)
        settings = Settings()
        base_dir = settings.temp_dir / "generated_images"
        
        full_path = Path(base_dir) / image_path
        
        # Security check: ensure path is within allowed directory
        if not str(full_path.resolve()).startswith(str(base_dir.resolve())):
            raise HTTPException(
                status_code=403,
                detail="访问被拒绝"
            )
        
        if not full_path.exists():
            raise HTTPException(
                status_code=404,
                detail="图像文件未找到"
            )
        
        return FileResponse(
            path=full_path,
            media_type="image/png",
            filename=full_path.name
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to serve image file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取图像文件失败: {str(e)}"
        )

@router.get("/gallery")
async def get_image_gallery(
    limit: int = Query(20, description="返回图像数量", ge=1, le=100),
    style_filter: Optional[str] = Query(None, description="按风格过滤"),
    settings: Settings = Depends(get_settings)
):
    """获取图像画廊列表"""
    
    try:
        base_dir = settings.temp_dir / "generated_images"
        
        if not base_dir.exists():
            return {
                "images": [],
                "total": 0,
                "timestamp": datetime.utcnow()
            }
        
        # Get all image files
        image_files = []
        for ext in ['*.png', '*.jpg', '*.jpeg']:
            image_files.extend(base_dir.glob(ext))
        
        # Filter by style if provided
        if style_filter:
            image_files = [f for f in image_files if style_filter.lower() in f.name.lower()]
        
        # Sort by modification time (newest first)
        image_files.sort(key=lambda f: f.stat().st_mtime, reverse=True)
        
        # Limit results
        image_files = image_files[:limit]
        
        # Build response
        images = []
        for img_file in image_files:
            # Look for thumbnail
            thumb_path = img_file.with_name(img_file.stem + '_thumb.jpg')
            
            images.append({
                "filename": img_file.name,
                "path": str(img_file.relative_to(base_dir)),
                "thumbnail_path": str(thumb_path.relative_to(base_dir)) if thumb_path.exists() else None,
                "size_bytes": img_file.stat().st_size,
                "created_at": datetime.fromtimestamp(img_file.stat().st_mtime).isoformat()
            })
        
        return {
            "images": images,
            "total": len(images),
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get image gallery: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取图像画廊失败: {str(e)}"
        )

# =============================
# Style and Configuration Endpoints
# =============================

@router.get("/styles")
async def get_available_styles():
    """获取可用的图像风格"""
    
    styles = []
    for style in ImageStyle:
        styles.append({
            "id": style.value,
            "name": style.value.replace('_', ' ').title(),
            "description": _get_style_description(style)
        })
    
    return {
        "styles": styles,
        "timestamp": datetime.utcnow()
    }

@router.get("/sizes")
async def get_available_sizes():
    """获取可用的图像尺寸"""
    
    sizes = []
    for size in ImageSize:
        sizes.append({
            "id": size.value,
            "dimensions": size.value,
            "aspect_ratio": _get_aspect_ratio(size),
            "recommended_for": _get_size_recommendation(size)
        })
    
    return {
        "sizes": sizes,
        "timestamp": datetime.utcnow()
    }

def _get_style_description(style: ImageStyle) -> str:
    """Get description for image style"""
    descriptions = {
        ImageStyle.PHOTOREALISTIC: "照片般真实的高质量图像",
        ImageStyle.ARTISTIC: "艺术性绘画风格，富有创意",
        ImageStyle.CARTOON: "卡通插画风格，色彩鲜艳",
        ImageStyle.CINEMATIC: "电影感镜头，戏剧性构图",
        ImageStyle.MINIMALIST: "简约设计，干净现代",
        ImageStyle.VINTAGE: "复古怀旧风格",
        ImageStyle.FUTURISTIC: "未来科技感设计",
        ImageStyle.CORPORATE: "商务专业风格"
    }
    return descriptions.get(style, "")

def _get_aspect_ratio(size: ImageSize) -> str:
    """Get aspect ratio description"""
    ratios = {
        ImageSize.SQUARE_1024: "1:1 (正方形)",
        ImageSize.LANDSCAPE_1792_1024: "16:9 (横屏)",
        ImageSize.PORTRAIT_1024_1792: "9:16 (竖屏)",
        ImageSize.HD_1920_1080: "16:9 (高清横屏)",
        ImageSize.VERTICAL_1080_1920: "9:16 (高清竖屏)"
    }
    return ratios.get(size, "")

def _get_size_recommendation(size: ImageSize) -> str:
    """Get recommendation for when to use this size"""
    recommendations = {
        ImageSize.SQUARE_1024: "社交媒体头像，产品展示",
        ImageSize.LANDSCAPE_1792_1024: "视频缩略图，横屏展示",
        ImageSize.PORTRAIT_1024_1792: "移动端展示，竖屏视频",
        ImageSize.HD_1920_1080: "高清视频背景，桌面壁纸",
        ImageSize.VERTICAL_1080_1920: "手机竖屏视频，故事展示"
    }
    return recommendations.get(size, "")