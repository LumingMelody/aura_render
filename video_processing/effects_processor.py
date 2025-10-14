"""
Advanced Effects Processor

Comprehensive video effects processing system with AI-powered effects,
real-time processing, and cinematic transformations.
"""

import asyncio
import logging
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import json
import math

from ai_service import get_enhanced_qwen_service
from monitoring import get_error_handler, get_metrics_collector
from monitoring.error_handler import ErrorCategory, ErrorSeverity

logger = logging.getLogger(__name__)

class EffectCategory(Enum):
    """Effect categories for organization"""
    COLOR_GRADING = "color_grading"
    TRANSITIONS = "transitions"
    MOTION = "motion"
    FILTERS = "filters"
    TEXT = "text"
    AUDIO = "audio"
    AI_ENHANCED = "ai_enhanced"
    CINEMATIC = "cinematic"

class EffectComplexity(Enum):
    """Effect processing complexity levels"""
    SIMPLE = "simple"      # Basic parameter adjustments
    MODERATE = "moderate"  # Multiple filter chains
    COMPLEX = "complex"    # Heavy processing/AI
    EXTREME = "extreme"    # Maximum quality/processing

@dataclass
class EffectParameter:
    """Single effect parameter specification"""
    name: str
    value: Any
    param_type: str  # 'float', 'int', 'string', 'bool', 'color', 'curve'
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    default_value: Any = None
    description: Optional[str] = None
    keyframes: Optional[List[Dict[str, Any]]] = None  # For animated parameters

@dataclass
class VideoEffect:
    """Complete video effect specification"""
    effect_id: str
    effect_name: str
    category: EffectCategory
    complexity: EffectComplexity
    
    # Processing parameters
    parameters: List[EffectParameter] = field(default_factory=list)
    enabled: bool = True
    
    # Timeline properties
    start_time: float = 0.0
    duration: Optional[float] = None
    
    # Target specification
    target_layers: List[str] = field(default_factory=list)
    blend_mode: str = "normal"
    opacity: float = 1.0
    
    # AI enhancement
    ai_enhanced: bool = False
    ai_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Performance metadata
    estimated_processing_time: Optional[float] = None
    memory_usage_mb: Optional[float] = None
    gpu_accelerated: bool = True

class AdvancedEffectsProcessor:
    """
    Advanced effects processing engine with AI enhancement,
    real-time capabilities, and cinematic-quality output
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.error_handler = get_error_handler()
        self.metrics = get_metrics_collector()
        self.ai_service = get_enhanced_qwen_service()
        
        # Effect library and templates
        self.effect_library = self._initialize_effect_library()
        self.cinematic_presets = self._initialize_cinematic_presets()
        
        # Performance optimization
        self.effect_cache = {}
        self.processing_stats = {
            "effects_processed": 0,
            "total_processing_time": 0.0,
            "cache_hits": 0,
            "ai_enhancements": 0
        }
        
        self.logger.info("Advanced Effects Processor initialized")
    
    async def process_effect_timeline(
        self,
        effects_timeline: List[VideoEffect],
        composition_data: Dict[str, Any],
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> List[Dict[str, Any]]:
        """
        Process complete effects timeline for video composition
        
        Args:
            effects_timeline: List of effects to apply
            composition_data: Video composition data
            progress_callback: Progress update callback
            
        Returns:
            Processed effects data ready for rendering
        """
        try:
            start_time = datetime.now()
            total_effects = len(effects_timeline)
            
            self.logger.info(f"Processing {total_effects} effects in timeline")
            
            # Optimize effects timeline
            optimized_timeline = await self._optimize_effects_timeline(effects_timeline)
            
            # Process effects in parallel where possible
            processed_effects = []
            
            for i, effect in enumerate(optimized_timeline):
                try:
                    # Process individual effect
                    processed_effect = await self._process_single_effect(
                        effect, composition_data
                    )
                    
                    if processed_effect:
                        processed_effects.append(processed_effect)
                    
                    # Update progress
                    if progress_callback:
                        progress = (i + 1) / total_effects * 100
                        progress_callback(progress)
                    
                except Exception as e:
                    self.logger.warning(f"Effect processing failed: {effect.effect_id} - {e}")
                    continue
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self.processing_stats["effects_processed"] += len(processed_effects)
            self.processing_stats["total_processing_time"] += processing_time
            
            self.metrics.record_effects_processing(
                effect_count=len(processed_effects),
                processing_time=processing_time,
                success=True
            )
            
            self.logger.info(f"Effects timeline processed successfully: {len(processed_effects)} effects in {processing_time:.2f}s")
            return processed_effects
            
        except Exception as e:
            await self.error_handler.handle_error(
                exception=e,
                category=ErrorCategory.EFFECTS_PROCESSING,
                severity=ErrorSeverity.HIGH,
                context={
                    "total_effects": len(effects_timeline),
                    "operation": "timeline_processing"
                }
            )
            raise
    
    async def create_ai_enhanced_effect(
        self,
        effect_description: str,
        target_mood: str,
        composition_context: Dict[str, Any]
    ) -> VideoEffect:
        """
        Create AI-enhanced effect based on natural language description
        
        Args:
            effect_description: Natural language effect description
            target_mood: Desired emotional impact
            composition_context: Context about the composition
            
        Returns:
            AI-generated video effect
        """
        try:
            self.logger.info(f"Creating AI-enhanced effect: {effect_description}")
            
            # Generate effect specification using AI
            effect_prompt = f"""Create a video effect specification for:
Description: {effect_description}
Target Mood: {target_mood}
Video Duration: {composition_context.get('duration', 10)} seconds
Resolution: {composition_context.get('resolution', [1920, 1080])}
Style: {composition_context.get('style', 'cinematic')}

Generate a JSON specification with effect type, parameters, and timing."""
            
            ai_response = await self.ai_service.generate_text(
                prompt=effect_prompt,
                max_tokens=1000,
                temperature=0.7
            )
            
            # Parse AI response
            try:
                effect_spec = json.loads(ai_response.content)
            except json.JSONDecodeError:
                # Fallback to predefined effect
                effect_spec = self._get_fallback_effect_spec(effect_description, target_mood)
            
            # Create VideoEffect from specification
            effect = self._create_effect_from_spec(effect_spec, ai_enhanced=True)
            
            self.processing_stats["ai_enhancements"] += 1
            
            return effect
            
        except Exception as e:
            self.logger.error(f"AI effect creation failed: {e}")
            # Return basic effect as fallback
            return self._create_basic_effect(effect_description)
    
    async def apply_cinematic_grade(
        self,
        composition_data: Dict[str, Any],
        style_name: str = "blockbuster"
    ) -> List[VideoEffect]:
        """
        Apply professional cinematic color grading
        
        Args:
            composition_data: Video composition data
            style_name: Cinematic style preset name
            
        Returns:
            List of color grading effects
        """
        try:
            if style_name not in self.cinematic_presets:
                style_name = "blockbuster"
            
            preset = self.cinematic_presets[style_name]
            duration = composition_data.get("duration", 10.0)
            
            effects = []
            
            # Primary color correction
            primary_grade = VideoEffect(
                effect_id=f"primary_grade_{style_name}",
                effect_name="Primary Color Correction",
                category=EffectCategory.COLOR_GRADING,
                complexity=EffectComplexity.MODERATE,
                duration=duration,
                parameters=[
                    EffectParameter("exposure", preset["exposure"], "float", -2.0, 2.0),
                    EffectParameter("contrast", preset["contrast"], "float", 0.0, 3.0),
                    EffectParameter("highlights", preset["highlights"], "float", -100, 100),
                    EffectParameter("shadows", preset["shadows"], "float", -100, 100),
                    EffectParameter("whites", preset["whites"], "float", -100, 100),
                    EffectParameter("blacks", preset["blacks"], "float", -100, 100)
                ]
            )
            effects.append(primary_grade)
            
            # Color wheels (3-way color correction)
            color_wheels = VideoEffect(
                effect_id=f"color_wheels_{style_name}",
                effect_name="3-Way Color Correction",
                category=EffectCategory.COLOR_GRADING,
                complexity=EffectComplexity.COMPLEX,
                duration=duration,
                parameters=[
                    EffectParameter("shadows_tint", preset["shadows_tint"], "color"),
                    EffectParameter("midtones_tint", preset["midtones_tint"], "color"),
                    EffectParameter("highlights_tint", preset["highlights_tint"], "color"),
                    EffectParameter("shadows_gain", preset["shadows_gain"], "float", 0.0, 2.0),
                    EffectParameter("midtones_gain", preset["midtones_gain"], "float", 0.0, 2.0),
                    EffectParameter("highlights_gain", preset["highlights_gain"], "float", 0.0, 2.0)
                ]
            )
            effects.append(color_wheels)
            
            # Film emulation
            film_look = VideoEffect(
                effect_id=f"film_emulation_{style_name}",
                effect_name="Film Look",
                category=EffectCategory.CINEMATIC,
                complexity=EffectComplexity.COMPLEX,
                duration=duration,
                parameters=[
                    EffectParameter("film_grain", preset["film_grain"], "float", 0.0, 1.0),
                    EffectParameter("vignette_strength", preset["vignette"], "float", 0.0, 1.0),
                    EffectParameter("saturation", preset["saturation"], "float", 0.0, 2.0),
                    EffectParameter("warmth", preset["warmth"], "float", -100, 100)
                ]
            )
            effects.append(film_look)
            
            return effects
            
        except Exception as e:
            self.logger.error(f"Cinematic grading failed: {e}")
            return []
    
    def create_transition_effect(
        self,
        transition_type: str,
        duration: float = 1.0,
        custom_parameters: Optional[Dict[str, Any]] = None
    ) -> VideoEffect:
        """
        Create professional transition effect
        
        Args:
            transition_type: Type of transition (fade, dissolve, wipe, etc.)
            duration: Transition duration in seconds
            custom_parameters: Custom transition parameters
            
        Returns:
            Configured transition effect
        """
        transition_configs = {
            "fade": {
                "complexity": EffectComplexity.SIMPLE,
                "parameters": [
                    EffectParameter("fade_type", "linear", "string"),
                    EffectParameter("fade_color", "#000000", "color")
                ]
            },
            "dissolve": {
                "complexity": EffectComplexity.SIMPLE,
                "parameters": [
                    EffectParameter("dissolve_type", "linear", "string")
                ]
            },
            "wipe": {
                "complexity": EffectComplexity.MODERATE,
                "parameters": [
                    EffectParameter("wipe_direction", "left_to_right", "string"),
                    EffectParameter("wipe_angle", 0.0, "float", 0.0, 360.0),
                    EffectParameter("feather", 10.0, "float", 0.0, 100.0)
                ]
            },
            "zoom": {
                "complexity": EffectComplexity.MODERATE,
                "parameters": [
                    EffectParameter("zoom_type", "in", "string"),
                    EffectParameter("zoom_center", [0.5, 0.5], "float"),
                    EffectParameter("zoom_amount", 2.0, "float", 1.0, 10.0)
                ]
            },
            "slide": {
                "complexity": EffectComplexity.MODERATE,
                "parameters": [
                    EffectParameter("slide_direction", "left", "string"),
                    EffectParameter("slide_distance", 1.0, "float", 0.0, 2.0)
                ]
            }
        }
        
        config = transition_configs.get(transition_type, transition_configs["fade"])
        
        # Apply custom parameters
        if custom_parameters:
            for param in config["parameters"]:
                if param.name in custom_parameters:
                    param.value = custom_parameters[param.name]
        
        return VideoEffect(
            effect_id=f"transition_{transition_type}_{int(datetime.now().timestamp())}",
            effect_name=f"{transition_type.title()} Transition",
            category=EffectCategory.TRANSITIONS,
            complexity=config["complexity"],
            duration=duration,
            parameters=config["parameters"]
        )
    
    def create_motion_effect(
        self,
        motion_type: str,
        intensity: float = 1.0,
        duration: Optional[float] = None
    ) -> VideoEffect:
        """
        Create dynamic motion effect
        
        Args:
            motion_type: Type of motion (pan, zoom, rotate, shake, etc.)
            intensity: Effect intensity (0.0-2.0)
            duration: Effect duration
            
        Returns:
            Configured motion effect
        """
        motion_configs = {
            "pan": {
                "parameters": [
                    EffectParameter("pan_direction", "horizontal", "string"),
                    EffectParameter("pan_speed", intensity, "float", 0.0, 5.0),
                    EffectParameter("pan_distance", intensity * 100, "float", 0.0, 500.0)
                ]
            },
            "zoom": {
                "parameters": [
                    EffectParameter("zoom_type", "in", "string"),
                    EffectParameter("zoom_speed", intensity, "float", 0.0, 3.0),
                    EffectParameter("zoom_center", [0.5, 0.5], "list")
                ]
            },
            "rotate": {
                "parameters": [
                    EffectParameter("rotation_speed", intensity * 45, "float", -180, 180),
                    EffectParameter("rotation_center", [0.5, 0.5], "list"),
                    EffectParameter("rotation_direction", "clockwise", "string")
                ]
            },
            "shake": {
                "parameters": [
                    EffectParameter("shake_intensity", intensity, "float", 0.0, 5.0),
                    EffectParameter("shake_frequency", intensity * 10, "float", 1.0, 50.0),
                    EffectParameter("shake_type", "random", "string")
                ]
            },
            "parallax": {
                "parameters": [
                    EffectParameter("parallax_strength", intensity, "float", 0.0, 2.0),
                    EffectParameter("parallax_direction", "horizontal", "string"),
                    EffectParameter("depth_layers", 3, "int", 2, 10)
                ]
            }
        }
        
        config = motion_configs.get(motion_type, motion_configs["pan"])
        
        return VideoEffect(
            effect_id=f"motion_{motion_type}_{int(datetime.now().timestamp())}",
            effect_name=f"{motion_type.title()} Motion",
            category=EffectCategory.MOTION,
            complexity=EffectComplexity.MODERATE,
            duration=duration,
            parameters=config["parameters"]
        )
    
    async def _optimize_effects_timeline(self, effects: List[VideoEffect]) -> List[VideoEffect]:
        """Optimize effects timeline for performance"""
        
        # Sort by start time and complexity
        optimized = sorted(effects, key=lambda e: (e.start_time, e.complexity.value))
        
        # Group compatible effects for batch processing
        batched_effects = []
        current_batch = []
        
        for effect in optimized:
            if (current_batch and 
                len(current_batch) < 3 and
                effect.category == current_batch[-1].category and
                abs(effect.start_time - current_batch[-1].start_time) < 0.1):
                current_batch.append(effect)
            else:
                if current_batch:
                    batched_effects.extend(current_batch)
                current_batch = [effect]
        
        if current_batch:
            batched_effects.extend(current_batch)
        
        return batched_effects
    
    async def _process_single_effect(
        self,
        effect: VideoEffect,
        composition_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Process a single effect"""
        
        try:
            # Check cache first
            cache_key = self._generate_effect_cache_key(effect)
            if cache_key in self.effect_cache:
                self.processing_stats["cache_hits"] += 1
                return self.effect_cache[cache_key]
            
            # Convert effect to render-compatible format
            processed_effect = {
                "effect_id": effect.effect_id,
                "type": effect.effect_name.lower().replace(" ", "_"),
                "category": effect.category.value,
                "start_time": effect.start_time,
                "duration": effect.duration,
                "enabled": effect.enabled,
                "target_layers": effect.target_layers,
                "blend_mode": effect.blend_mode,
                "opacity": effect.opacity,
                "parameters": {}
            }
            
            # Process parameters
            for param in effect.parameters:
                processed_effect["parameters"][param.name] = param.value
            
            # Add AI enhancement if enabled
            if effect.ai_enhanced:
                ai_enhancement = await self._apply_ai_enhancement(effect, composition_data)
                processed_effect["ai_enhancement"] = ai_enhancement
            
            # Cache the result
            self.effect_cache[cache_key] = processed_effect
            
            return processed_effect
            
        except Exception as e:
            self.logger.error(f"Effect processing failed: {effect.effect_id} - {e}")
            return None
    
    def _initialize_effect_library(self) -> Dict[str, Dict[str, Any]]:
        """Initialize built-in effect library"""
        return {
            "color_correction": {
                "category": EffectCategory.COLOR_GRADING,
                "complexity": EffectComplexity.MODERATE,
                "parameters": [
                    "exposure", "contrast", "brightness", "saturation",
                    "highlights", "shadows", "whites", "blacks"
                ]
            },
            "blur": {
                "category": EffectCategory.FILTERS,
                "complexity": EffectComplexity.SIMPLE,
                "parameters": ["radius", "type"]
            },
            "sharpen": {
                "category": EffectCategory.FILTERS,
                "complexity": EffectComplexity.SIMPLE,
                "parameters": ["amount", "radius", "threshold"]
            },
            "vignette": {
                "category": EffectCategory.CINEMATIC,
                "complexity": EffectComplexity.SIMPLE,
                "parameters": ["strength", "size", "feather", "roundness"]
            },
            "film_grain": {
                "category": EffectCategory.CINEMATIC,
                "complexity": EffectComplexity.MODERATE,
                "parameters": ["amount", "size", "intensity"]
            },
            "lens_flare": {
                "category": EffectCategory.CINEMATIC,
                "complexity": EffectComplexity.COMPLEX,
                "parameters": ["position", "intensity", "type", "rays"]
            }
        }
    
    def _initialize_cinematic_presets(self) -> Dict[str, Dict[str, Any]]:
        """Initialize cinematic color grading presets"""
        return {
            "blockbuster": {
                "exposure": 0.2,
                "contrast": 1.3,
                "highlights": -30,
                "shadows": 20,
                "whites": 10,
                "blacks": -15,
                "shadows_tint": "#1a2b3d",
                "midtones_tint": "#ffffff",
                "highlights_tint": "#ffa500",
                "shadows_gain": 0.9,
                "midtones_gain": 1.0,
                "highlights_gain": 1.1,
                "film_grain": 0.15,
                "vignette": 0.3,
                "saturation": 1.2,
                "warmth": 15
            },
            "indie_film": {
                "exposure": -0.1,
                "contrast": 1.1,
                "highlights": -20,
                "shadows": 30,
                "whites": -5,
                "blacks": -20,
                "shadows_tint": "#2d3a4a",
                "midtones_tint": "#f5f5dc",
                "highlights_tint": "#ffebcd",
                "shadows_gain": 0.8,
                "midtones_gain": 1.0,
                "highlights_gain": 1.05,
                "film_grain": 0.25,
                "vignette": 0.4,
                "saturation": 0.9,
                "warmth": -10
            },
            "sci_fi": {
                "exposure": 0.0,
                "contrast": 1.4,
                "highlights": -40,
                "shadows": 25,
                "whites": 20,
                "blacks": -25,
                "shadows_tint": "#001122",
                "midtones_tint": "#e6e6fa",
                "highlights_tint": "#00ffff",
                "shadows_gain": 0.7,
                "midtones_gain": 1.0,
                "highlights_gain": 1.2,
                "film_grain": 0.05,
                "vignette": 0.2,
                "saturation": 1.4,
                "warmth": -25
            }
        }
    
    def _get_fallback_effect_spec(self, description: str, mood: str) -> Dict[str, Any]:
        """Generate fallback effect specification"""
        return {
            "effect_type": "color_correction",
            "parameters": {
                "exposure": 0.1 if mood == "bright" else -0.1,
                "contrast": 1.2 if mood == "dramatic" else 1.0,
                "saturation": 1.3 if mood == "vibrant" else 1.0
            },
            "duration": 5.0,
            "complexity": "moderate"
        }
    
    def _create_effect_from_spec(self, spec: Dict[str, Any], ai_enhanced: bool = False) -> VideoEffect:
        """Create VideoEffect from specification"""
        
        effect_type = spec.get("effect_type", "color_correction")
        
        parameters = []
        for name, value in spec.get("parameters", {}).items():
            param = EffectParameter(
                name=name,
                value=value,
                param_type="float"  # Simplified for this implementation
            )
            parameters.append(param)
        
        return VideoEffect(
            effect_id=f"ai_{effect_type}_{int(datetime.now().timestamp())}",
            effect_name=f"AI {effect_type.replace('_', ' ').title()}",
            category=EffectCategory.AI_ENHANCED,
            complexity=EffectComplexity.COMPLEX,
            duration=spec.get("duration", 5.0),
            parameters=parameters,
            ai_enhanced=ai_enhanced
        )
    
    def _create_basic_effect(self, description: str) -> VideoEffect:
        """Create basic fallback effect"""
        return VideoEffect(
            effect_id=f"basic_{int(datetime.now().timestamp())}",
            effect_name="Basic Effect",
            category=EffectCategory.FILTERS,
            complexity=EffectComplexity.SIMPLE,
            parameters=[
                EffectParameter("intensity", 0.5, "float", 0.0, 1.0)
            ]
        )
    
    async def _apply_ai_enhancement(self, effect: VideoEffect, composition_data: Dict[str, Any]) -> Dict[str, Any]:
        """Apply AI enhancement to effect"""
        # Simplified AI enhancement - would be more sophisticated in production
        return {
            "enhanced": True,
            "confidence": 0.85,
            "suggestions": ["Increase intensity by 15%", "Add subtle animation"]
        }
    
    def _generate_effect_cache_key(self, effect: VideoEffect) -> str:
        """Generate cache key for effect"""
        param_string = "_".join([f"{p.name}={p.value}" for p in effect.parameters])
        return f"{effect.effect_name}_{param_string}_{effect.duration}"
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get processing statistics"""
        stats = self.processing_stats.copy()
        
        if stats["effects_processed"] > 0:
            stats["average_processing_time"] = (
                stats["total_processing_time"] / stats["effects_processed"]
            )
        else:
            stats["average_processing_time"] = 0.0
        
        stats["cache_hit_rate"] = (
            stats["cache_hits"] / max(stats["effects_processed"], 1) * 100
        )
        
        return stats


# Global instance
_effects_processor: Optional[AdvancedEffectsProcessor] = None

def get_effects_processor() -> AdvancedEffectsProcessor:
    """Get global effects processor instance"""
    global _effects_processor
    if _effects_processor is None:
        _effects_processor = AdvancedEffectsProcessor()
    return _effects_processor