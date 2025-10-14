"""
AI Optimization API Endpoints

REST API endpoints for AI-powered video optimization features including:
- Video quality enhancement
- Content analysis and optimization
- Scene detection and analysis
- Performance optimization recommendations
"""

from fastapi import APIRouter, HTTPException, Depends, status, UploadFile, File, Form
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from pathlib import Path
import tempfile
import os

from ai_optimization import (
    get_video_optimizer,
    get_quality_enhancer,
    get_scene_analyzer,
    get_content_optimizer,
    OptimizationConfig,
    OptimizationLevel,
    OptimizationType,
    EnhancementConfig,
    QualityLevel,
    EnhancementType,
    TargetPlatform,
    OptimizationStrategy,
    ContentCategory,
    AudienceSegment
)

router = APIRouter(prefix="/api/ai-optimization", tags=["ai-optimization"])


# Request/Response Models

class OptimizationRequest(BaseModel):
    """Request for video optimization"""
    input_path: str
    output_path: str
    level: OptimizationLevel = OptimizationLevel.STANDARD
    target_quality: float = Field(0.85, ge=0.0, le=1.0)
    target_bitrate_mbps: Optional[float] = Field(None, ge=0.1, le=100.0)
    max_file_size_mb: Optional[float] = Field(None, ge=1.0)
    content_type: str = "general"
    target_platform: str = "web"
    enable_ai_enhancement: bool = True
    enable_scene_detection: bool = True
    enable_audio_optimization: bool = True
    enable_color_correction: bool = True
    use_neural_upscaling: bool = False
    use_adaptive_bitrate: bool = True


class QualityEnhancementRequest(BaseModel):
    """Request for quality enhancement"""
    input_path: str
    output_path: str
    enhancement_level: QualityLevel = QualityLevel.BALANCED
    enabled_enhancements: List[EnhancementType] = []
    upscale_factor: float = Field(1.0, ge=1.0, le=4.0)
    noise_reduction_strength: float = Field(0.5, ge=0.0, le=1.0)
    sharpening_strength: float = Field(0.3, ge=0.0, le=1.0)
    use_gpu: bool = True


class ContentAnalysisRequest(BaseModel):
    """Request for content analysis"""
    video_path: str
    target_platform: Optional[TargetPlatform] = None


class PlatformOptimizationRequest(BaseModel):
    """Request for platform-specific optimization"""
    video_path: str
    target_platform: TargetPlatform
    strategy: OptimizationStrategy = OptimizationStrategy.ENGAGEMENT


class BatchOptimizationRequest(BaseModel):
    """Request for batch optimization"""
    input_files: List[str] = Field(..., min_items=1, max_items=50)
    output_directory: str
    optimization_config: OptimizationRequest


# Response Models

class OptimizationResponse(BaseModel):
    """Response for optimization operations"""
    success: bool
    job_id: Optional[str] = None
    message: str
    processing_time: Optional[float] = None
    original_size_mb: Optional[float] = None
    optimized_size_mb: Optional[float] = None
    size_reduction_percent: Optional[float] = None
    quality_score: Optional[float] = None
    optimizations_applied: List[str] = []
    warnings: List[str] = []
    errors: List[str] = []


class ContentAnalysisResponse(BaseModel):
    """Response for content analysis"""
    category: str
    confidence: float
    characteristics: Dict[str, float]
    platform_scores: Dict[str, float]
    audience_scores: Dict[str, float]
    recommended_strategies: List[str]
    improvement_suggestions: List[str]


class SceneAnalysisResponse(BaseModel):
    """Response for scene analysis"""
    total_scenes: int
    total_duration: float
    average_scene_duration: float
    analysis_confidence: float
    processing_time: float
    scenes: List[Dict[str, Any]]
    transitions: List[Dict[str, Any]]
    scene_type_distribution: Dict[str, int]
    complexity_distribution: Dict[str, float]
    motion_distribution: Dict[str, float]


# Optimization Endpoints

@router.post("/optimize", response_model=OptimizationResponse)
async def optimize_video(request: OptimizationRequest):
    """Optimize video using AI-powered techniques"""
    try:
        optimizer = get_video_optimizer()
        
        # Convert request to optimization config
        config = OptimizationConfig(
            level=request.level,
            target_quality=request.target_quality,
            target_bitrate_mbps=request.target_bitrate_mbps,
            max_file_size_mb=request.max_file_size_mb,
            enable_ai_enhancement=request.enable_ai_enhancement,
            enable_scene_detection=request.enable_scene_detection,
            enable_audio_optimization=request.enable_audio_optimization,
            enable_color_correction=request.enable_color_correction,
            content_type=request.content_type,
            target_platform=request.target_platform,
            use_neural_upscaling=request.use_neural_upscaling,
            use_adaptive_bitrate=request.use_adaptive_bitrate
        )
        
        # Run optimization
        result = await optimizer.optimize_video(
            input_path=request.input_path,
            output_path=request.output_path,
            config=config
        )
        
        return OptimizationResponse(
            success=result.success,
            message="Video optimization completed successfully" if result.success else "Video optimization failed",
            processing_time=result.processing_time,
            original_size_mb=result.original_size_mb,
            optimized_size_mb=result.optimized_size_mb,
            size_reduction_percent=result.size_reduction_percent,
            quality_score=result.quality_score,
            optimizations_applied=result.optimizations_applied,
            warnings=result.warnings,
            errors=result.errors
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Optimization failed: {str(e)}"
        )


@router.post("/enhance-quality", response_model=OptimizationResponse)
async def enhance_video_quality(request: QualityEnhancementRequest):
    """Enhance video quality using AI techniques"""
    try:
        enhancer = get_quality_enhancer()
        
        # Convert request to enhancement config
        config = EnhancementConfig(
            enhancement_level=request.enhancement_level,
            enabled_enhancements=request.enabled_enhancements or None,
            upscale_factor=request.upscale_factor,
            noise_reduction_strength=request.noise_reduction_strength,
            sharpening_strength=request.sharpening_strength,
            use_gpu=request.use_gpu
        )
        
        # Run enhancement
        result = await enhancer.enhance_video(
            input_path=request.input_path,
            output_path=request.output_path,
            config=config
        )
        
        if result["success"]:
            improvements = result["improvements"]
            return OptimizationResponse(
                success=True,
                message="Video quality enhancement completed successfully",
                processing_time=result["processing_time"],
                quality_score=result["output_metrics"].overall_score,
                optimizations_applied=result["enhancements_applied"],
                warnings=[],
                errors=[]
            )
        else:
            return OptimizationResponse(
                success=False,
                message="Video quality enhancement failed",
                processing_time=result.get("processing_time", 0),
                errors=[result.get("error", "Unknown error")]
            )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Quality enhancement failed: {str(e)}"
        )


@router.post("/analyze-scenes", response_model=SceneAnalysisResponse)
async def analyze_video_scenes(video_path: str):
    """Analyze video scenes using AI"""
    try:
        analyzer = get_scene_analyzer()
        
        # Run scene analysis
        result = await analyzer.analyze_scenes(video_path)
        
        # Convert scenes to dict format
        scenes_data = []
        for scene in result.scenes:
            scenes_data.append({
                "scene_id": scene.scene_id,
                "start_time": scene.start_time,
                "end_time": scene.end_time,
                "duration": scene.duration,
                "scene_type": scene.scene_type.value,
                "confidence": scene.confidence,
                "motion_intensity": scene.motion_intensity,
                "complexity_score": scene.complexity_score,
                "brightness": scene.brightness,
                "contrast": scene.contrast,
                "color_diversity": scene.color_diversity,
                "keyframes": scene.keyframes,
                "dominant_colors": scene.dominant_colors
            })
        
        # Convert transitions to dict format
        transitions_data = []
        for transition in result.transitions:
            transitions_data.append({
                "timestamp": transition.timestamp,
                "transition_type": transition.transition_type.value,
                "confidence": transition.confidence,
                "duration": transition.duration,
                "from_scene": transition.from_scene,
                "to_scene": transition.to_scene
            })
        
        # Convert scene type distribution
        scene_type_dist = {scene_type.value: count for scene_type, count in result.scene_type_distribution.items()}
        
        return SceneAnalysisResponse(
            total_scenes=result.total_scenes,
            total_duration=result.total_duration,
            average_scene_duration=result.average_scene_duration,
            analysis_confidence=result.analysis_confidence,
            processing_time=result.processing_time,
            scenes=scenes_data,
            transitions=transitions_data,
            scene_type_distribution=scene_type_dist,
            complexity_distribution=result.complexity_distribution,
            motion_distribution=result.motion_distribution
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Scene analysis failed: {str(e)}"
        )


@router.post("/analyze-content", response_model=ContentAnalysisResponse)
async def analyze_video_content(request: ContentAnalysisRequest):
    """Analyze video content for optimization opportunities"""
    try:
        optimizer = get_content_optimizer()
        
        # Run content analysis
        result = await optimizer.analyze_content(
            video_path=request.video_path,
            target_platform=request.target_platform
        )
        
        # Convert platform scores to string keys
        platform_scores = {platform.value: score for platform, score in result.platform_scores.items()}
        audience_scores = {audience.value: score for audience, score in result.audience_scores.items()}
        
        return ContentAnalysisResponse(
            category=result.category.value,
            confidence=result.confidence,
            characteristics={
                "complexity_score": result.complexity_score,
                "entertainment_value": result.entertainment_value,
                "educational_value": result.educational_value,
                "emotional_impact": result.emotional_impact,
                "visual_appeal": result.visual_appeal,
                "pacing_score": result.pacing_score,
                "text_density": result.text_density,
                "audio_quality": result.audio_quality
            },
            platform_scores=platform_scores,
            audience_scores=audience_scores,
            recommended_strategies=[strategy.value for strategy in result.recommended_strategies],
            improvement_suggestions=result.improvement_suggestions
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Content analysis failed: {str(e)}"
        )


@router.post("/optimize-for-platform")
async def optimize_for_platform(request: PlatformOptimizationRequest):
    """Get platform-specific optimization recommendations"""
    try:
        optimizer = get_content_optimizer()
        
        # Get optimization recommendation
        recommendation = await optimizer.optimize_for_platform(
            video_path=request.video_path,
            target_platform=request.target_platform,
            strategy=request.strategy
        )
        
        return {
            "success": True,
            "platform": request.target_platform.value,
            "strategy": recommendation.strategy.value,
            "priority": recommendation.priority,
            "expected_improvement": recommendation.expected_improvement,
            "implementation_effort": recommendation.implementation_effort,
            "recommendations": {
                "bitrate_adjustment": recommendation.bitrate_adjustment,
                "resolution_recommendation": recommendation.resolution_recommendation,
                "duration_recommendation": recommendation.duration_recommendation,
                "thumbnail_suggestions": recommendation.thumbnail_suggestions,
                "title_suggestions": recommendation.title_suggestions
            }
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Platform optimization failed: {str(e)}"
        )


@router.post("/predict-performance")
async def predict_content_performance(video_path: str, target_platform: TargetPlatform, optimization_applied: bool = False):
    """Predict content performance on target platform"""
    try:
        content_optimizer = get_content_optimizer()
        
        # Analyze content first
        analysis = await content_optimizer.analyze_content(video_path, target_platform)
        
        # Predict performance
        prediction = await content_optimizer.predict_performance(
            analysis=analysis,
            target_platform=target_platform,
            optimization_applied=optimization_applied
        )
        
        return {
            "success": True,
            "prediction": prediction
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Performance prediction failed: {str(e)}"
        )


# Batch Operations

@router.post("/batch-optimize")
async def batch_optimize_videos(request: BatchOptimizationRequest):
    """Optimize multiple videos in batch"""
    try:
        optimizer = get_video_optimizer()
        
        # Convert request to optimization config
        config = OptimizationConfig(
            level=request.optimization_config.level,
            target_quality=request.optimization_config.target_quality,
            target_bitrate_mbps=request.optimization_config.target_bitrate_mbps,
            max_file_size_mb=request.optimization_config.max_file_size_mb,
            enable_ai_enhancement=request.optimization_config.enable_ai_enhancement,
            enable_scene_detection=request.optimization_config.enable_scene_detection,
            enable_audio_optimization=request.optimization_config.enable_audio_optimization,
            enable_color_correction=request.optimization_config.enable_color_correction,
            content_type=request.optimization_config.content_type,
            target_platform=request.optimization_config.target_platform,
            use_neural_upscaling=request.optimization_config.use_neural_upscaling,
            use_adaptive_bitrate=request.optimization_config.use_adaptive_bitrate
        )
        
        # Run batch optimization
        results = await optimizer.batch_optimize(
            input_files=request.input_files,
            output_dir=request.output_directory,
            config=config
        )
        
        # Process results
        successful_count = sum(1 for r in results if r.success)
        total_processing_time = sum(r.processing_time for r in results)
        total_size_reduction = sum(r.original_size_mb - r.optimized_size_mb for r in results if r.success)
        
        return {
            "success": True,
            "message": f"Batch optimization completed: {successful_count}/{len(results)} files processed successfully",
            "summary": {
                "total_files": len(request.input_files),
                "successful_optimizations": successful_count,
                "failed_optimizations": len(results) - successful_count,
                "total_processing_time": total_processing_time,
                "total_size_reduction_mb": total_size_reduction
            },
            "results": [
                {
                    "input_file": r.original_file_path,
                    "output_file": r.optimized_file_path,
                    "success": r.success,
                    "size_reduction_percent": r.size_reduction_percent,
                    "quality_score": r.quality_score,
                    "processing_time": r.processing_time,
                    "errors": r.errors
                }
                for r in results
            ]
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch optimization failed: {str(e)}"
        )


# Statistics and Information Endpoints

@router.get("/optimization-stats")
async def get_optimization_stats():
    """Get optimization performance statistics"""
    try:
        optimizer = get_video_optimizer()
        enhancer = get_quality_enhancer()
        analyzer = get_scene_analyzer()
        content_optimizer = get_content_optimizer()
        
        return {
            "video_optimization": optimizer.get_stats(),
            "quality_enhancement": enhancer.get_performance_stats(),
            "scene_analysis": analyzer.get_analysis_stats(),
            "content_optimization": content_optimizer.get_optimization_stats()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get statistics: {str(e)}"
        )


@router.get("/supported-platforms")
async def get_supported_platforms():
    """Get list of supported target platforms"""
    return {
        "platforms": [
            {
                "value": platform.value,
                "label": platform.value.replace("_", " ").title()
            }
            for platform in TargetPlatform
        ]
    }


@router.get("/optimization-strategies")
async def get_optimization_strategies():
    """Get list of available optimization strategies"""
    return {
        "strategies": [
            {
                "value": strategy.value,
                "label": strategy.value.replace("_", " ").title(),
                "description": _get_strategy_description(strategy)
            }
            for strategy in OptimizationStrategy
        ]
    }


@router.get("/enhancement-types")
async def get_enhancement_types():
    """Get list of available enhancement types"""
    return {
        "enhancement_types": [
            {
                "value": enhancement.value,
                "label": enhancement.value.replace("_", " ").title(),
                "description": _get_enhancement_description(enhancement)
            }
            for enhancement in EnhancementType
        ]
    }


# Upload endpoint for direct file processing

@router.post("/upload-and-optimize")
async def upload_and_optimize(
    file: UploadFile = File(...),
    level: str = Form("standard"),
    target_quality: float = Form(0.85),
    enable_ai_enhancement: bool = Form(True)
):
    """Upload video file and optimize it"""
    try:
        # Validate file type
        if not file.filename.lower().endswith(('.mp4', '.mov', '.avi', '.mkv')):
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported file format. Please upload MP4, MOV, AVI, or MKV files."
            )
        
        # Save uploaded file to temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as temp_input:
            content = await file.read()
            temp_input.write(content)
            temp_input_path = temp_input.name
        
        # Create temporary output file
        temp_output_path = tempfile.mktemp(suffix="_optimized" + Path(file.filename).suffix)
        
        try:
            # Create optimization config
            config = OptimizationConfig(
                level=OptimizationLevel(level),
                target_quality=target_quality,
                enable_ai_enhancement=enable_ai_enhancement
            )
            
            # Run optimization
            optimizer = get_video_optimizer()
            result = await optimizer.optimize_video(
                input_path=temp_input_path,
                output_path=temp_output_path,
                config=config
            )
            
            return {
                "success": result.success,
                "message": "Upload and optimization completed",
                "processing_time": result.processing_time,
                "size_reduction_percent": result.size_reduction_percent,
                "quality_score": result.quality_score,
                "optimizations_applied": result.optimizations_applied,
                "download_url": f"/download/{Path(temp_output_path).name}",  # Would need download endpoint
                "warnings": result.warnings,
                "errors": result.errors
            }
            
        finally:
            # Cleanup temporary files (in production, might want to keep output for download)
            if os.path.exists(temp_input_path):
                os.unlink(temp_input_path)
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Upload and optimization failed: {str(e)}"
        )


# Helper functions for descriptions

def _get_strategy_description(strategy: OptimizationStrategy) -> str:
    """Get description for optimization strategy"""
    descriptions = {
        OptimizationStrategy.ENGAGEMENT: "Maximize viewer engagement and interaction",
        OptimizationStrategy.RETENTION: "Maximize watch time and viewer retention",
        OptimizationStrategy.CONVERSION: "Maximize conversions and click-through rates",
        OptimizationStrategy.REACH: "Maximize reach and impressions",
        OptimizationStrategy.QUALITY: "Maximize perceived video quality",
        OptimizationStrategy.EFFICIENCY: "Minimize file size while maintaining quality",
        OptimizationStrategy.ACCESSIBILITY: "Optimize for accessibility and inclusive design"
    }
    return descriptions.get(strategy, "Optimization strategy")


def _get_enhancement_description(enhancement: EnhancementType) -> str:
    """Get description for enhancement type"""
    descriptions = {
        EnhancementType.UPSCALING: "Increase resolution using neural networks",
        EnhancementType.NOISE_REDUCTION: "Reduce video noise and artifacts",
        EnhancementType.SHARPENING: "Enhance image sharpness and detail",
        EnhancementType.COLOR_ENHANCEMENT: "Improve color accuracy and vibrancy",
        EnhancementType.ARTIFACT_REMOVAL: "Remove compression artifacts",
        EnhancementType.CONTRAST_ADJUSTMENT: "Optimize contrast and brightness",
        EnhancementType.SATURATION_BOOST: "Enhance color saturation",
        EnhancementType.EXPOSURE_CORRECTION: "Correct exposure and lighting issues"
    }
    return descriptions.get(enhancement, "Enhancement technique")