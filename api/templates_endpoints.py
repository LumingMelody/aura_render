"""
Templates API Endpoints

REST API endpoints for video templates and presets management.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from content_generator.templates_system import (
    get_templates_system,
    TemplateCategory,
    TemplateStyle,
    VideoTemplate,
    TemplatePreset
)

router = APIRouter(prefix="/api/templates", tags=["templates"])


class TemplateRequest(BaseModel):
    """Request model for creating custom templates"""
    name: str = Field(..., min_length=1, max_length=100)
    description: str = Field("", max_length=500)
    category: TemplateCategory
    style: TemplateStyle
    scenes: List[Dict[str, Any]]
    duration: float = Field(default=30.0, ge=1.0, le=3600.0)
    aspect_ratio: str = Field(default="16:9")
    resolution: str = Field(default="1920x1080")
    fps: int = Field(default=30, ge=1, le=60)
    customizable_elements: List[str] = Field(default_factory=list)
    color_scheme: Dict[str, str] = Field(default_factory=dict)
    tags: List[str] = Field(default_factory=list)


class CustomizationRequest(BaseModel):
    """Request model for applying template customizations"""
    template_id: str
    customizations: Dict[str, Any]


class PresetRequest(BaseModel):
    """Request model for saving template presets"""
    name: str = Field(..., min_length=1, max_length=100)
    template_id: str
    settings: Dict[str, Any]
    description: str = Field("", max_length=500)


class TemplateResponse(BaseModel):
    """Response model for template operations"""
    id: str
    name: str
    description: str
    category: str
    style: str
    duration: float
    aspect_ratio: str
    resolution: str
    fps: int
    customizable_elements: List[str]
    tags: List[str]
    is_premium: bool
    usage_count: int
    created_at: datetime


class PresetResponse(BaseModel):
    """Response model for preset operations"""
    id: str
    name: str
    template_id: str
    description: str
    thumbnail_url: Optional[str]


@router.get("/", response_model=List[TemplateResponse])
async def get_templates(
    category: Optional[TemplateCategory] = None,
    style: Optional[TemplateStyle] = None,
    is_premium: Optional[bool] = None
):
    """Get all templates with optional filtering"""
    try:
        templates_system = get_templates_system()
        templates = await templates_system.get_templates(category, style, is_premium)
        
        return [
            TemplateResponse(
                id=template.id,
                name=template.name,
                description=template.description,
                category=template.category.value,
                style=template.style.value,
                duration=template.duration,
                aspect_ratio=template.aspect_ratio,
                resolution=template.resolution,
                fps=template.fps,
                customizable_elements=template.customizable_elements,
                tags=template.tags,
                is_premium=template.is_premium,
                usage_count=template.usage_count,
                created_at=template.created_at
            )
            for template in templates
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get templates: {str(e)}"
        )


@router.get("/{template_id}", response_model=TemplateResponse)
async def get_template(template_id: str):
    """Get a specific template by ID"""
    try:
        templates_system = get_templates_system()
        template = await templates_system.get_template(template_id)
        
        if not template:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Template not found"
            )
        
        return TemplateResponse(
            id=template.id,
            name=template.name,
            description=template.description,
            category=template.category.value,
            style=template.style.value,
            duration=template.duration,
            aspect_ratio=template.aspect_ratio,
            resolution=template.resolution,
            fps=template.fps,
            customizable_elements=template.customizable_elements,
            tags=template.tags,
            is_premium=template.is_premium,
            usage_count=template.usage_count,
            created_at=template.created_at
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get template: {str(e)}"
        )


@router.post("/", response_model=TemplateResponse)
async def create_template(request: TemplateRequest):
    """Create a new custom template"""
    try:
        templates_system = get_templates_system()
        
        template = await templates_system.create_custom_template(
            name=request.name,
            description=request.description,
            category=request.category,
            style=request.style,
            scenes=request.scenes,
            duration=request.duration,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            fps=request.fps,
            customizable_elements=request.customizable_elements,
            color_scheme=request.color_scheme,
            tags=request.tags
        )
        
        return TemplateResponse(
            id=template.id,
            name=template.name,
            description=template.description,
            category=template.category.value,
            style=template.style.value,
            duration=template.duration,
            aspect_ratio=template.aspect_ratio,
            resolution=template.resolution,
            fps=template.fps,
            customizable_elements=template.customizable_elements,
            tags=template.tags,
            is_premium=template.is_premium,
            usage_count=template.usage_count,
            created_at=template.created_at
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create template: {str(e)}"
        )


@router.post("/customize")
async def customize_template(request: CustomizationRequest):
    """Apply customizations to a template"""
    try:
        templates_system = get_templates_system()
        
        customized_config = await templates_system.customize_template(
            request.template_id,
            request.customizations
        )
        
        return {
            "success": True,
            "config": customized_config,
            "message": "Template customized successfully"
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to customize template: {str(e)}"
        )


@router.get("/categories/list")
async def get_template_categories():
    """Get all available template categories"""
    return {
        "categories": [
            {"value": category.value, "label": category.value.replace("_", " ").title()}
            for category in TemplateCategory
        ]
    }


@router.get("/styles/list")
async def get_template_styles():
    """Get all available template styles"""
    return {
        "styles": [
            {"value": style.value, "label": style.value.replace("_", " ").title()}
            for style in TemplateStyle
        ]
    }


# Preset endpoints

@router.post("/presets", response_model=PresetResponse)
async def save_preset(request: PresetRequest):
    """Save a customized template as a preset"""
    try:
        templates_system = get_templates_system()
        
        preset = await templates_system.save_preset(
            name=request.name,
            template_id=request.template_id,
            settings=request.settings,
            description=request.description
        )
        
        return PresetResponse(
            id=preset.id,
            name=preset.name,
            template_id=preset.template_id,
            description=preset.description,
            thumbnail_url=preset.thumbnail_url
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save preset: {str(e)}"
        )


@router.get("/presets", response_model=List[PresetResponse])
async def get_presets(template_id: Optional[str] = None):
    """Get all presets, optionally filtered by template"""
    try:
        templates_system = get_templates_system()
        presets = await templates_system.get_presets(template_id)
        
        return [
            PresetResponse(
                id=preset.id,
                name=preset.name,
                template_id=preset.template_id,
                description=preset.description,
                thumbnail_url=preset.thumbnail_url
            )
            for preset in presets
        ]
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get presets: {str(e)}"
        )


@router.post("/presets/{preset_id}/apply")
async def apply_preset(preset_id: str):
    """Apply a saved preset"""
    try:
        templates_system = get_templates_system()
        
        config = await templates_system.apply_preset(preset_id)
        
        return {
            "success": True,
            "config": config,
            "message": "Preset applied successfully"
        }
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to apply preset: {str(e)}"
        )