"""
Unified Video Processing API

Comprehensive API endpoints for AI-powered video generation,
composition, effects processing, and rendering management.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks, status, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import tempfile
import uuid
import logging
from pathlib import Path

from video_processing import (
    get_composition_engine, get_effects_processor, 
    VideoComposition, CompositionLayer, VideoEffect,
    EffectCategory, EffectComplexity
)
from render_engine import get_render_manager, RenderConfig
from config import get_settings

# Configure router
video_router = APIRouter(prefix="/video", tags=["video"])
logger = logging.getLogger(__name__)

# Pydantic Models

class AIVideoGenerationRequest(BaseModel):
    """Request for AI-powered video generation"""
    prompt: str = Field(..., description="Natural language description of desired video", min_length=10)
    style_preferences: Optional[Dict[str, Any]] = Field(None, description="Visual and audio style preferences")
    content_requirements: Optional[Dict[str, Any]] = Field(None, description="Technical requirements")
    template_name: Optional[str] = Field(None, description="Composition template to use")
    
    class Config:
        schema_extra = {
            "example": {
                "prompt": "Create a cinematic video showing a beautiful sunset over mountains with peaceful ambient music",
                "style_preferences": {
                    "mood": "peaceful",
                    "color_scheme": "warm",
                    "pace": "slow"
                },
                "content_requirements": {
                    "duration": 15.0,
                    "resolution": [1920, 1080],
                    "framerate": 30.0
                },
                "template_name": "storytelling"
            }
        }

class VideoCompositionRequest(BaseModel):
    """Request for manual video composition"""
    composition_id: Optional[str] = None
    title: str = Field(..., description="Composition title")
    description: Optional[str] = None
    duration: float = Field(10.0, description="Video duration in seconds", gt=0, le=300)
    resolution: List[int] = Field([1920, 1080], description="Video resolution [width, height]")
    framerate: float = Field(30.0, description="Video framerate", gt=0, le=120)
    
    class Config:
        schema_extra = {
            "example": {
                "title": "My Custom Video",
                "description": "A manually composed video",
                "duration": 20.0,
                "resolution": [1920, 1080],
                "framerate": 30.0
            }
        }

class LayerRequest(BaseModel):
    """Request to add layer to composition"""
    layer_type: str = Field(..., description="Layer type: video, audio, image, text")
    source_path: Optional[str] = Field(None, description="Source file path or URL")
    content: Optional[str] = Field(None, description="Text content for text layers")
    start_time: float = Field(0.0, description="Layer start time in seconds", ge=0)
    duration: Optional[float] = Field(None, description="Layer duration in seconds")
    position: List[int] = Field([0, 0], description="Layer position [x, y]")
    size: Optional[List[int]] = Field(None, description="Layer size [width, height]")
    opacity: float = Field(1.0, description="Layer opacity (0-1)", ge=0, le=1)
    volume: float = Field(1.0, description="Audio volume (0-2)", ge=0, le=2)

class EffectRequest(BaseModel):
    """Request to add effect to composition"""
    effect_type: str = Field(..., description="Type of effect")
    parameters: Dict[str, Any] = Field(default_factory=dict, description="Effect parameters")
    start_time: float = Field(0.0, description="Effect start time", ge=0)
    duration: Optional[float] = Field(None, description="Effect duration")
    target_layers: List[str] = Field(default_factory=list, description="Target layer IDs")

class CinematicGradeRequest(BaseModel):
    """Request for cinematic color grading"""
    style_name: str = Field("blockbuster", description="Cinematic style preset")
    custom_parameters: Optional[Dict[str, Any]] = Field(None, description="Custom grading parameters")

class RenderRequest(BaseModel):
    """Request to render composition"""
    composition_id: str = Field(..., description="Composition ID to render")
    output_filename: str = Field(..., description="Output filename")
    render_config: Optional[Dict[str, Any]] = Field(None, description="Render configuration")
    priority: int = Field(0, description="Render priority", ge=0, le=10)

# Response Models

class CompositionResponse(BaseModel):
    """Response with composition data"""
    composition_id: str
    title: str
    description: Optional[str]
    duration: float
    resolution: List[int]
    framerate: float
    layer_count: int
    effects_count: int
    created_at: datetime
    updated_at: datetime
    status: str = "ready"

class LayerResponse(BaseModel):
    """Response with layer data"""
    layer_id: str
    layer_type: str
    start_time: float
    duration: Optional[float]
    position: List[int]
    created_at: datetime

class EffectResponse(BaseModel):
    """Response with effect data"""
    effect_id: str
    effect_name: str
    category: str
    complexity: str
    start_time: float
    duration: Optional[float]
    target_layers: List[str]
    created_at: datetime

class RenderResponse(BaseModel):
    """Response with render status"""
    task_id: str
    composition_id: str
    status: str
    output_path: Optional[str]
    progress: float
    estimated_completion: Optional[datetime]
    created_at: datetime

# Global services
composition_engine = None
effects_processor = None
render_manager = None

def get_services():
    """Initialize services lazily"""
    global composition_engine, effects_processor, render_manager
    if composition_engine is None:
        composition_engine = get_composition_engine()
        effects_processor = get_effects_processor()
        render_manager = get_render_manager()
    return composition_engine, effects_processor, render_manager

# API Endpoints

@video_router.post("/generate", response_model=CompositionResponse)
async def generate_ai_video(request: AIVideoGenerationRequest):
    """Generate video composition using AI"""
    try:
        comp_engine, _, _ = get_services()
        
        # Generate AI composition
        composition = await comp_engine.create_ai_composition(
            generation_prompt=request.prompt,
            style_preferences=request.style_preferences,
            content_requirements=request.content_requirements,
            template_name=request.template_name
        )
        
        return CompositionResponse(
            composition_id=composition.composition_id,
            title=composition.title,
            description=composition.description,
            duration=composition.duration,
            resolution=list(composition.resolution),
            framerate=composition.framerate,
            layer_count=len(composition.layers),
            effects_count=len(composition.effects_timeline),
            created_at=composition.created_at,
            updated_at=composition.updated_at
        )
        
    except Exception as e:
        logger.error(f"AI video generation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Video generation failed: {str(e)}"
        )

@video_router.post("/compositions", response_model=CompositionResponse)
async def create_composition(request: VideoCompositionRequest):
    """Create new video composition"""
    try:
        composition_id = request.composition_id or str(uuid.uuid4())
        
        composition = VideoComposition(
            composition_id=composition_id,
            title=request.title,
            description=request.description,
            duration=request.duration,
            resolution=tuple(request.resolution),
            framerate=request.framerate
        )
        
        # Store composition (in production, would use database)
        # For now, we'll just return the response
        
        return CompositionResponse(
            composition_id=composition.composition_id,
            title=composition.title,
            description=composition.description,
            duration=composition.duration,
            resolution=list(composition.resolution),
            framerate=composition.framerate,
            layer_count=0,
            effects_count=0,
            created_at=composition.created_at,
            updated_at=composition.updated_at
        )
        
    except Exception as e:
        logger.error(f"Composition creation failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Composition creation failed: {str(e)}"
        )

@video_router.post("/compositions/{composition_id}/layers", response_model=LayerResponse)
async def add_layer(composition_id: str, request: LayerRequest):
    """Add layer to composition"""
    try:
        layer_id = f"{request.layer_type}_{uuid.uuid4().hex[:8]}"
        
        layer = CompositionLayer(
            layer_id=layer_id,
            layer_type=request.layer_type,
            source_path=request.source_path,
            content=request.content,
            start_time=request.start_time,
            duration=request.duration,
            position=tuple(request.position),
            size=tuple(request.size) if request.size else None,
            opacity=request.opacity,
            volume=request.volume
        )
        
        # In production, would store in database
        
        return LayerResponse(
            layer_id=layer.layer_id,
            layer_type=layer.layer_type,
            start_time=layer.start_time,
            duration=layer.duration,
            position=list(layer.position),
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Layer addition failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Layer addition failed: {str(e)}"
        )

@video_router.post("/compositions/{composition_id}/effects", response_model=EffectResponse)
async def add_effect(composition_id: str, request: EffectRequest):
    """Add effect to composition"""
    try:
        _, effects_proc, _ = get_services()
        
        effect_id = f"effect_{uuid.uuid4().hex[:8]}"
        
        # Create effect based on type
        if request.effect_type == "cinematic_grade":
            style = request.parameters.get("style", "blockbuster")
            # Would create cinematic grade effect
            effect_name = f"Cinematic Grade ({style})"
            category = EffectCategory.CINEMATIC
            complexity = EffectComplexity.COMPLEX
            
        elif request.effect_type == "transition":
            transition_type = request.parameters.get("type", "fade")
            effect = effects_proc.create_transition_effect(
                transition_type=transition_type,
                duration=request.duration or 1.0,
                custom_parameters=request.parameters
            )
            effect_name = effect.effect_name
            category = effect.category
            complexity = effect.complexity
            
        elif request.effect_type == "motion":
            motion_type = request.parameters.get("type", "pan")
            intensity = request.parameters.get("intensity", 1.0)
            effect = effects_proc.create_motion_effect(
                motion_type=motion_type,
                intensity=intensity,
                duration=request.duration
            )
            effect_name = effect.effect_name
            category = effect.category
            complexity = effect.complexity
            
        else:
            # Generic effect
            effect_name = request.effect_type.replace("_", " ").title()
            category = EffectCategory.FILTERS
            complexity = EffectComplexity.SIMPLE
        
        return EffectResponse(
            effect_id=effect_id,
            effect_name=effect_name,
            category=category.value,
            complexity=complexity.value,
            start_time=request.start_time,
            duration=request.duration,
            target_layers=request.target_layers,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Effect addition failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Effect addition failed: {str(e)}"
        )

@video_router.post("/compositions/{composition_id}/cinematic-grade")
async def apply_cinematic_grade(composition_id: str, request: CinematicGradeRequest):
    """Apply cinematic color grading to composition"""
    try:
        _, effects_proc, _ = get_services()
        
        # Mock composition data for this example
        composition_data = {
            "duration": 30.0,
            "resolution": [1920, 1080]
        }
        
        effects = await effects_proc.apply_cinematic_grade(
            composition_data=composition_data,
            style_name=request.style_name
        )
        
        effect_responses = []
        for effect in effects:
            effect_responses.append(EffectResponse(
                effect_id=effect.effect_id,
                effect_name=effect.effect_name,
                category=effect.category.value,
                complexity=effect.complexity.value,
                start_time=effect.start_time,
                duration=effect.duration,
                target_layers=effect.target_layers,
                created_at=datetime.now()
            ))
        
        return {
            "composition_id": composition_id,
            "style_applied": request.style_name,
            "effects_added": len(effects),
            "effects": effect_responses,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Cinematic grading failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Cinematic grading failed: {str(e)}"
        )

@video_router.post("/render", response_model=RenderResponse)
async def render_composition(request: RenderRequest):
    """Submit composition for rendering"""
    try:
        comp_engine, _, render_mgr = get_services()
        
        # In production, would fetch composition from database
        # For now, create mock composition
        mock_composition = VideoComposition(
            composition_id=request.composition_id,
            title="Mock Composition",
            duration=10.0
        )
        
        # Create render config
        render_config = RenderConfig()
        if request.render_config:
            for key, value in request.render_config.items():
                if hasattr(render_config, key):
                    setattr(render_config, key, value)
        
        # Submit render task
        output_path = f"/tmp/aura_render_outputs/{request.output_filename}"
        
        composition_data = {
            "composition_id": request.composition_id,
            "layers": [],
            "effects_timeline": [],
            "duration": 10.0,
            "resolution": [1920, 1080],
            "framerate": 30.0
        }
        
        task_id = await render_mgr.submit_render_task(
            composition_data=composition_data,
            output_path=output_path,
            render_config=render_config,
            priority=request.priority
        )
        
        return RenderResponse(
            task_id=task_id,
            composition_id=request.composition_id,
            status="queued",
            output_path=output_path,
            progress=0.0,
            created_at=datetime.now()
        )
        
    except Exception as e:
        logger.error(f"Render submission failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Render submission failed: {str(e)}"
        )

@video_router.get("/render/{task_id}")
async def get_render_status(task_id: str):
    """Get render task status"""
    try:
        _, _, render_mgr = get_services()
        
        task_status = await render_mgr.get_task_status(task_id)
        
        if not task_status:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Render task not found"
            )
        
        return task_status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Render status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Render status check failed: {str(e)}"
        )

@video_router.get("/render")
async def get_render_queue():
    """Get render queue status"""
    try:
        _, _, render_mgr = get_services()
        return await render_mgr.get_queue_status()
        
    except Exception as e:
        logger.error(f"Render queue status failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Render queue status failed: {str(e)}"
        )

@video_router.get("/effects/library")
async def get_effects_library():
    """Get available effects library"""
    try:
        _, effects_proc, _ = get_services()
        
        # Get effect categories and types
        categories = [category.value for category in EffectCategory]
        complexities = [complexity.value for complexity in EffectComplexity]
        
        # Get processing statistics
        stats = effects_proc.get_processing_statistics()
        
        return {
            "categories": categories,
            "complexity_levels": complexities,
            "processing_statistics": stats,
            "available_presets": {
                "cinematic_styles": ["blockbuster", "indie_film", "sci_fi"],
                "transitions": ["fade", "dissolve", "wipe", "zoom", "slide"],
                "motion_effects": ["pan", "zoom", "rotate", "shake", "parallax"]
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Effects library query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Effects library query failed: {str(e)}"
        )

@video_router.get("/templates")
async def get_composition_templates():
    """Get available composition templates"""
    try:
        comp_engine, _, _ = get_services()
        
        templates = comp_engine.templates
        style_presets = comp_engine.style_presets
        
        return {
            "composition_templates": templates,
            "style_presets": style_presets,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"Templates query failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Templates query failed: {str(e)}"
        )

@video_router.post("/upload")
async def upload_media_file(file: UploadFile = File(...)):
    """Upload media file for use in compositions"""
    try:
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No filename provided"
            )
        
        # Validate file type
        allowed_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.mp3', '.wav', '.jpg', '.png'}
        file_extension = Path(file.filename).suffix.lower()
        
        if file_extension not in allowed_extensions:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File type {file_extension} not allowed"
            )
        
        # Create upload directory
        upload_dir = Path("/tmp/aura_uploads")
        upload_dir.mkdir(exist_ok=True)
        
        # Save file
        file_id = uuid.uuid4().hex
        file_path = upload_dir / f"{file_id}_{file.filename}"
        
        with file_path.open("wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "file_id": file_id,
            "filename": file.filename,
            "file_path": str(file_path),
            "file_size": len(content),
            "file_type": file_extension,
            "upload_time": datetime.now(),
            "status": "uploaded"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"File upload failed: {str(e)}"
        )

@video_router.get("/system/status")
async def get_system_status():
    """Get video processing system status"""
    try:
        comp_engine, effects_proc, render_mgr = get_services()
        
        # Get system statistics
        effects_stats = effects_proc.get_processing_statistics()
        queue_status = await render_mgr.get_queue_status()
        
        return {
            "system_status": "online",
            "components": {
                "composition_engine": "ready",
                "effects_processor": "ready",
                "render_manager": "ready"
            },
            "performance": {
                "effects_processing": effects_stats,
                "render_queue": queue_status["statistics"]
            },
            "capabilities": {
                "ai_generation": True,
                "real_time_effects": True,
                "batch_rendering": True,
                "cinematic_grading": True
            },
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        logger.error(f"System status check failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"System status check failed: {str(e)}"
        )