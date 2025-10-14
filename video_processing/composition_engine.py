"""
Unified Video Composition Engine

Advanced video composition system that integrates AI services, material management,
and rendering engines for intelligent video generation.
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

from ai_service import get_enhanced_qwen_service, get_prompt_manager
from materials import MaterialManager, MaterialType, MaterialSearchQuery
from render_engine import FFmpegRenderer, RenderConfig, get_render_manager
from video_processing import FFmpegEngine, get_ffmpeg_engine
from monitoring import get_error_handler, get_metrics_collector
from monitoring.error_handler import ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)

@dataclass
class CompositionLayer:
    """Single composition layer specification"""
    layer_id: str
    layer_type: str  # 'video', 'audio', 'image', 'text', 'effect'
    source_path: Optional[str] = None
    content: Optional[str] = None  # For text layers or generated content
    start_time: float = 0.0
    duration: Optional[float] = None
    position: Tuple[int, int] = (0, 0)
    size: Optional[Tuple[int, int]] = None
    opacity: float = 1.0
    volume: float = 1.0  # For audio layers
    
    # Effects and transformations
    effects: List[Dict[str, Any]] = field(default_factory=list)
    transforms: Dict[str, Any] = field(default_factory=dict)
    
    # AI-generated metadata
    ai_metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: float = 0.8

@dataclass
class VideoComposition:
    """Complete video composition specification"""
    composition_id: str
    title: str
    description: Optional[str] = None
    
    # Technical specifications
    duration: float = 10.0
    resolution: Tuple[int, int] = (1920, 1080)
    framerate: float = 30.0
    
    # Content layers
    layers: List[CompositionLayer] = field(default_factory=list)
    effects_timeline: List[Dict[str, Any]] = field(default_factory=list)
    
    # AI generation parameters
    generation_prompt: Optional[str] = None
    style_preferences: Dict[str, Any] = field(default_factory=dict)
    content_requirements: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)


class UnifiedCompositionEngine:
    """
    Unified video composition engine integrating AI services, 
    material management, and advanced rendering
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_handler = get_error_handler()
        self.metrics = get_metrics_collector()
        
        # Initialize services
        self.ai_service = get_enhanced_qwen_service()
        self.prompt_manager = get_prompt_manager()
        self.material_manager = MaterialManager()
        self.ffmpeg_engine = get_ffmpeg_engine()
        self.render_manager = get_render_manager()
        
        # Composition templates and presets
        self.templates = self._initialize_composition_templates()
        self.style_presets = self._initialize_style_presets()
        
        self.logger.info("Unified Composition Engine initialized")
    
    async def create_ai_composition(
        self,
        generation_prompt: str,
        style_preferences: Optional[Dict[str, Any]] = None,
        content_requirements: Optional[Dict[str, Any]] = None,
        template_name: Optional[str] = None
    ) -> VideoComposition:
        """
        Create a video composition using AI-driven content generation
        
        Args:
            generation_prompt: Natural language description of desired video
            style_preferences: Visual and audio style preferences
            content_requirements: Technical requirements (duration, resolution, etc.)
            template_name: Optional composition template to use
            
        Returns:
            Generated video composition
        """
        try:
            start_time = datetime.now()
            
            # Generate composition ID
            composition_id = f"comp_{int(start_time.timestamp())}"
            
            self.logger.info(f"Creating AI composition: {composition_id}")
            
            # Initialize composition with requirements
            composition = VideoComposition(
                composition_id=composition_id,
                title=f"AI Generated Video - {start_time.strftime('%Y%m%d_%H%M%S')}",
                generation_prompt=generation_prompt,
                style_preferences=style_preferences or {},
                content_requirements=content_requirements or {}
            )
            
            # Apply template if specified
            if template_name and template_name in self.templates:
                composition = self._apply_template(composition, template_name)
            
            # Apply content requirements
            if content_requirements:
                composition.duration = content_requirements.get('duration', 10.0)
                composition.resolution = tuple(content_requirements.get('resolution', [1920, 1080]))
                composition.framerate = content_requirements.get('framerate', 30.0)
            
            # AI-powered composition planning
            composition_plan = await self._generate_composition_plan(
                generation_prompt, 
                composition, 
                style_preferences
            )
            
            # Generate and collect materials
            materials_data = await self._collect_composition_materials(composition_plan)
            
            # Create composition layers
            layers = await self._create_composition_layers(
                composition_plan, 
                materials_data,
                composition.duration
            )
            composition.layers = layers
            
            # Generate effects timeline
            effects_timeline = await self._generate_effects_timeline(
                composition_plan, 
                layers, 
                composition.duration
            )
            composition.effects_timeline = effects_timeline
            
            # AI-powered composition optimization
            composition = await self._optimize_composition(composition)
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            self.metrics.record_composition_creation(
                composition_id=composition_id,
                processing_time=processing_time,
                layer_count=len(layers),
                effects_count=len(effects_timeline),
                success=True
            )
            
            self.logger.info(f"AI composition created successfully: {composition_id}")
            return composition
            
        except Exception as e:
            await self.error_handler.handle_error(
                exception=e,
                category=ErrorCategory.COMPOSITION,
                severity=ErrorSeverity.HIGH,
                context={
                    "generation_prompt": generation_prompt,
                    "template": template_name,
                    "operation": "ai_composition_creation"
                }
            )
            raise
    
    async def render_composition(
        self,
        composition: VideoComposition,
        output_path: str,
        render_config: Optional[RenderConfig] = None,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Render video composition to output file
        
        Args:
            composition: Video composition to render
            output_path: Output file path
            render_config: Optional render configuration
            progress_callback: Progress update callback
            
        Returns:
            Render result with metrics and metadata
        """
        try:
            self.logger.info(f"Starting composition render: {composition.composition_id}")
            
            # Convert composition to render-ready format
            composition_data = await self._prepare_composition_for_rendering(composition)
            
            # Submit render task
            task_id = await self.render_manager.submit_render_task(
                composition_data=composition_data,
                output_path=output_path,
                render_config=render_config or RenderConfig(),
                priority=5,
                progress_callback=progress_callback
            )
            
            self.logger.info(f"Render task submitted: {task_id}")
            
            # Wait for completion and return result
            return await self._monitor_render_task(task_id)
            
        except Exception as e:
            await self.error_handler.handle_error(
                exception=e,
                category=ErrorCategory.RENDERING,
                severity=ErrorSeverity.HIGH,
                context={
                    "composition_id": composition.composition_id,
                    "output_path": output_path,
                    "operation": "composition_rendering"
                }
            )
            raise
    
    async def _generate_composition_plan(
        self,
        generation_prompt: str,
        composition: VideoComposition,
        style_preferences: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate AI-powered composition plan"""
        
        # Prepare prompt for composition planning
        planning_prompt = await self.prompt_manager.render_prompt(
            "video_composition_planning",
            {
                "user_prompt": generation_prompt,
                "duration": composition.duration,
                "resolution": f"{composition.resolution[0]}x{composition.resolution[1]}",
                "style_preferences": json.dumps(style_preferences or {}),
                "available_materials": "video, audio, image, text"
            }
        )
        
        # Generate composition plan using AI
        response = await self.ai_service.generate_text(
            prompt=planning_prompt,
            max_tokens=2000,
            temperature=0.7
        )
        
        try:
            # Parse AI response into structured plan
            composition_plan = json.loads(response.content)
        except json.JSONDecodeError:
            # Fallback to basic plan if AI response is not valid JSON
            composition_plan = {
                "scenes": [
                    {
                        "scene_id": "scene_1",
                        "description": generation_prompt,
                        "start_time": 0.0,
                        "duration": composition.duration,
                        "materials_needed": [
                            {"type": "video", "keywords": ["background", "nature"]},
                            {"type": "audio", "keywords": ["ambient", "peaceful"]}
                        ]
                    }
                ],
                "overall_style": style_preferences or {"mood": "neutral", "pace": "medium"},
                "material_requirements": {
                    "video_count": 1,
                    "audio_count": 1,
                    "image_count": 0,
                    "text_count": 0
                }
            }
        
        return composition_plan
    
    async def _collect_composition_materials(self, composition_plan: Dict[str, Any]) -> Dict[str, Any]:
        """Collect materials based on composition plan"""
        
        materials_data = {
            "video_materials": [],
            "audio_materials": [],
            "image_materials": [],
            "text_content": []
        }
        
        # Extract material requirements from all scenes
        all_material_needs = []
        for scene in composition_plan.get("scenes", []):
            materials_needed = scene.get("materials_needed", [])
            all_material_needs.extend(materials_needed)
        
        # Search for materials by type
        for material_need in all_material_needs:
            material_type = material_need.get("type", "video")
            keywords = material_need.get("keywords", [])
            
            if not keywords:
                continue
            
            try:
                if material_type == "video":
                    results = await self.material_manager.smart_search(
                        keywords=keywords,
                        material_type=MaterialType.VIDEO,
                        max_results=3
                    )
                    materials_data["video_materials"].extend(results)
                
                elif material_type == "audio":
                    results = await self.material_manager.smart_search(
                        keywords=keywords,
                        material_type=MaterialType.AUDIO,
                        max_results=2
                    )
                    materials_data["audio_materials"].extend(results)
                
                elif material_type == "image":
                    results = await self.material_manager.smart_search(
                        keywords=keywords,
                        material_type=MaterialType.IMAGE,
                        max_results=5
                    )
                    materials_data["image_materials"].extend(results)
                
            except Exception as e:
                self.logger.warning(f"Failed to collect {material_type} materials: {e}")
                continue
        
        return materials_data
    
    async def _create_composition_layers(
        self,
        composition_plan: Dict[str, Any],
        materials_data: Dict[str, Any],
        total_duration: float
    ) -> List[CompositionLayer]:
        """Create composition layers from plan and materials"""
        
        layers = []
        layer_id_counter = 0
        
        # Create background video layer
        if materials_data.get("video_materials"):
            video_material = materials_data["video_materials"][0]
            
            video_layer = CompositionLayer(
                layer_id=f"video_{layer_id_counter}",
                layer_type="video",
                source_path=video_material.url,
                start_time=0.0,
                duration=total_duration,
                position=(0, 0),
                opacity=1.0,
                ai_metadata={
                    "material_id": video_material.material_id,
                    "title": video_material.metadata.title,
                    "relevance_score": video_material.relevance_score
                }
            )
            layers.append(video_layer)
            layer_id_counter += 1
        
        # Create audio layer
        if materials_data.get("audio_materials"):
            audio_material = materials_data["audio_materials"][0]
            
            audio_layer = CompositionLayer(
                layer_id=f"audio_{layer_id_counter}",
                layer_type="audio",
                source_path=audio_material.url,
                start_time=0.0,
                duration=total_duration,
                volume=0.7,
                ai_metadata={
                    "material_id": audio_material.material_id,
                    "title": audio_material.metadata.title,
                    "relevance_score": audio_material.relevance_score
                }
            )
            layers.append(audio_layer)
            layer_id_counter += 1
        
        # Add overlay images if available
        for i, image_material in enumerate(materials_data.get("image_materials", [])[:2]):
            start_time = (total_duration / 3) * i
            duration = min(total_duration / 3, 5.0)
            
            image_layer = CompositionLayer(
                layer_id=f"image_{layer_id_counter}",
                layer_type="image",
                source_path=image_material.url,
                start_time=start_time,
                duration=duration,
                position=(50 + i * 100, 50),
                size=(300, 200),
                opacity=0.8,
                ai_metadata={
                    "material_id": image_material.material_id,
                    "title": image_material.metadata.title,
                    "relevance_score": image_material.relevance_score
                }
            )
            layers.append(image_layer)
            layer_id_counter += 1
        
        return layers
    
    async def _generate_effects_timeline(
        self,
        composition_plan: Dict[str, Any],
        layers: List[CompositionLayer],
        total_duration: float
    ) -> List[Dict[str, Any]]:
        """Generate effects timeline for composition"""
        
        effects_timeline = []
        
        # Apply scene-based effects
        scenes = composition_plan.get("scenes", [])
        style = composition_plan.get("overall_style", {})
        
        for scene in scenes:
            scene_start = scene.get("start_time", 0.0)
            scene_duration = scene.get("duration", total_duration)
            
            # Add fade in at beginning
            if scene_start == 0.0:
                fade_in_effect = {
                    "effect_id": "fade_in_start",
                    "type": "fade_in",
                    "start_time": scene_start,
                    "duration": 1.0,
                    "parameters": {"strength": 1.0},
                    "target_layers": ["video_0"] if layers else [],
                    "enabled": True
                }
                effects_timeline.append(fade_in_effect)
            
            # Add fade out at end
            if scene_start + scene_duration >= total_duration - 1.0:
                fade_out_effect = {
                    "effect_id": "fade_out_end",
                    "type": "fade_out",
                    "start_time": total_duration - 1.0,
                    "duration": 1.0,
                    "parameters": {"strength": 1.0},
                    "target_layers": ["video_0"] if layers else [],
                    "enabled": True
                }
                effects_timeline.append(fade_out_effect)
            
            # Apply style-based effects
            mood = style.get("mood", "neutral")
            if mood == "dramatic":
                color_effect = {
                    "effect_id": f"color_grading_{scene['scene_id']}",
                    "type": "color_grading",
                    "start_time": scene_start,
                    "duration": scene_duration,
                    "parameters": {
                        "contrast": 1.2,
                        "saturation": 1.1,
                        "brightness": 0.05
                    },
                    "target_layers": [layer.layer_id for layer in layers if layer.layer_type == "video"],
                    "enabled": True
                }
                effects_timeline.append(color_effect)
        
        return effects_timeline
    
    async def _optimize_composition(self, composition: VideoComposition) -> VideoComposition:
        """AI-powered composition optimization"""
        
        try:
            # Analyze composition for optimization opportunities
            optimization_prompt = await self.prompt_manager.render_prompt(
                "composition_optimization",
                {
                    "layer_count": len(composition.layers),
                    "duration": composition.duration,
                    "effects_count": len(composition.effects_timeline),
                    "resolution": f"{composition.resolution[0]}x{composition.resolution[1]}"
                }
            )
            
            # Get AI suggestions for optimization
            response = await self.ai_service.generate_text(
                prompt=optimization_prompt,
                max_tokens=1000,
                temperature=0.3
            )
            
            # Apply optimization suggestions (simplified implementation)
            # In a full implementation, this would parse AI suggestions and apply them
            
            # Update composition metadata
            composition.updated_at = datetime.now()
            composition.tags.extend(["ai_optimized", "auto_generated"])
            
        except Exception as e:
            self.logger.warning(f"Composition optimization failed: {e}")
        
        return composition
    
    async def _prepare_composition_for_rendering(self, composition: VideoComposition) -> Dict[str, Any]:
        """Prepare composition data for rendering engines"""
        
        # Convert layers to render-compatible format
        render_layers = []
        for layer in composition.layers:
            render_layer = {
                "layer_id": layer.layer_id,
                "layer_type": layer.layer_type,
                "source_path": layer.source_path,
                "content": layer.content,
                "start_time": layer.start_time,
                "duration": layer.duration,
                "position": layer.position,
                "size": layer.size,
                "opacity": layer.opacity,
                "volume": layer.volume,
                "effects": layer.effects,
                "transforms": layer.transforms
            }
            render_layers.append(render_layer)
        
        return {
            "composition_id": composition.composition_id,
            "layers": render_layers,
            "effects_timeline": composition.effects_timeline,
            "duration": composition.duration,
            "resolution": composition.resolution,
            "framerate": composition.framerate,
            "metadata": {
                "title": composition.title,
                "description": composition.description,
                "created_at": composition.created_at.isoformat(),
                "generation_prompt": composition.generation_prompt,
                "style_preferences": composition.style_preferences
            }
        }
    
    async def _monitor_render_task(self, task_id: str) -> Dict[str, Any]:
        """Monitor render task completion"""
        
        max_wait_time = 300  # 5 minutes
        check_interval = 2   # 2 seconds
        elapsed_time = 0
        
        while elapsed_time < max_wait_time:
            task_status = await self.render_manager.get_task_status(task_id)
            
            if not task_status:
                raise RuntimeError(f"Render task not found: {task_id}")
            
            status = task_status.get("status")
            
            if status == "completed":
                return {
                    "success": True,
                    "task_id": task_id,
                    "result": task_status.get("result", {}),
                    "processing_time": task_status.get("processing_time"),
                    "output_path": task_status.get("output_path")
                }
            elif status == "failed":
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": task_status.get("error_message", "Unknown error"),
                    "processing_time": task_status.get("processing_time")
                }
            elif status == "cancelled":
                return {
                    "success": False,
                    "task_id": task_id,
                    "error": "Task was cancelled",
                    "processing_time": task_status.get("processing_time")
                }
            
            await asyncio.sleep(check_interval)
            elapsed_time += check_interval
        
        # Timeout
        raise RuntimeError(f"Render task timeout: {task_id}")
    
    def _initialize_composition_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize composition templates"""
        return {
            "storytelling": {
                "description": "Narrative-driven composition with multiple scenes",
                "default_duration": 30.0,
                "scene_structure": ["intro", "development", "climax", "conclusion"],
                "layer_priorities": ["video", "audio", "text", "effects"]
            },
            "product_showcase": {
                "description": "Product-focused composition with clean aesthetics",
                "default_duration": 15.0,
                "scene_structure": ["product_intro", "features", "call_to_action"],
                "layer_priorities": ["video", "image", "text", "audio"]
            },
            "social_media": {
                "description": "Short-form content optimized for social platforms",
                "default_duration": 10.0,
                "scene_structure": ["hook", "content", "engagement"],
                "layer_priorities": ["video", "text", "effects", "audio"]
            },
            "educational": {
                "description": "Learning-focused composition with clear information flow",
                "default_duration": 60.0,
                "scene_structure": ["introduction", "concepts", "examples", "summary"],
                "layer_priorities": ["video", "text", "image", "audio"]
            }
        }
    
    def _initialize_style_presets(self) -> Dict[str, Dict[str, Any]]:
        """Initialize style presets"""
        return {
            "corporate": {
                "color_scheme": ["#1E3A8A", "#F8FAFC", "#64748B"],
                "font_family": "Arial",
                "pace": "medium",
                "effects": ["fade", "slide"],
                "music_style": "corporate"
            },
            "creative": {
                "color_scheme": ["#F59E0B", "#EF4444", "#8B5CF6"],
                "font_family": "Helvetica",
                "pace": "fast",
                "effects": ["zoom", "rotate", "color_shift"],
                "music_style": "upbeat"
            },
            "minimal": {
                "color_scheme": ["#000000", "#FFFFFF", "#6B7280"],
                "font_family": "SF Pro",
                "pace": "slow",
                "effects": ["fade", "subtle_motion"],
                "music_style": "ambient"
            }
        }
    
    def _apply_template(self, composition: VideoComposition, template_name: str) -> VideoComposition:
        """Apply composition template"""
        template = self.templates.get(template_name)
        if not template:
            return composition
        
        # Apply template settings
        composition.duration = template.get("default_duration", composition.duration)
        composition.content_requirements.update({
            "template": template_name,
            "scene_structure": template.get("scene_structure", []),
            "layer_priorities": template.get("layer_priorities", [])
        })
        
        return composition


# Global instance
_composition_engine: Optional[UnifiedCompositionEngine] = None

def get_composition_engine() -> UnifiedCompositionEngine:
    """Get global composition engine instance"""
    global _composition_engine
    if _composition_engine is None:
        _composition_engine = UnifiedCompositionEngine()
    return _composition_engine