"""
Effects and Transitions Processor Node

Handles visual effects, transitions, animations, and post-processing
for video content including color grading, motion graphics, and special effects.
"""

import asyncio
import json
import random
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, field
from enum import Enum

from .base_node import BaseNode, NodeConfig, NodeResult, ProcessingContext, NodeStatus
from ai_service import get_enhanced_qwen_service, get_prompt_manager


class EffectType(Enum):
    """Types of visual effects"""
    COLOR_GRADING = "color_grading"
    TRANSITION = "transition"
    ANIMATION = "animation"
    FILTER = "filter"
    OVERLAY = "overlay"
    PARTICLE = "particle"
    TEXT_ANIMATION = "text_animation"
    LOGO_ANIMATION = "logo_animation"


class TransitionType(Enum):
    """Types of transitions between scenes"""
    CUT = "cut"
    FADE = "fade"
    CROSS_FADE = "cross_fade"
    SLIDE = "slide"
    ZOOM = "zoom"
    ROTATE = "rotate"
    WIPE = "wipe"
    IRIS = "iris"


@dataclass
class EffectsConfig(NodeConfig):
    """Configuration for effects processing node"""
    enable_color_grading: bool = True
    enable_transitions: bool = True
    enable_animations: bool = True
    enable_filters: bool = True
    color_profile: str = "cinematic"  # Options: natural, cinematic, vibrant, vintage
    transition_duration: float = 0.5  # Default transition duration in seconds
    animation_style: str = "smooth"  # Options: smooth, snappy, organic
    quality_level: str = "high"  # Options: low, medium, high, ultra
    gpu_acceleration: bool = True
    particle_effects: bool = False
    motion_blur: bool = True
    bloom_effect: bool = False
    custom_effects: List[str] = field(default_factory=list)


@dataclass
class EffectDefinition:
    """Definition of a visual effect"""
    effect_id: str
    effect_type: EffectType
    name: str
    parameters: Dict[str, Any]
    duration: float
    start_time: float
    intensity: float = 1.0
    blend_mode: str = "normal"
    enabled: bool = True


class EffectsProcessorNode(BaseNode):
    """Effects and transitions processor node for video generation pipeline"""
    
    def __init__(self, config: EffectsConfig):
        super().__init__(config)
        self.config: EffectsConfig = config
        self.ai_service = get_enhanced_qwen_service()
        self.prompt_manager = get_prompt_manager()
        
    def get_required_inputs(self) -> List[str]:
        """Required inputs for effects processing"""
        return ['shot_blocks', 'emotion_analysis', 'video_duration']
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Output schema for effects processing"""
        return {
            'effects_timeline': 'list',
            'transitions': 'list',
            'color_grading_config': 'dict',
            'animation_sequences': 'list',
            'rendering_config': 'dict',
            'effects_metadata': 'dict'
        }
    
    def validate_input(self, context: ProcessingContext) -> bool:
        """Validate input for effects processing"""
        required_keys = ['shot_blocks', 'video_duration']
        
        for key in required_keys:
            if key not in context.intermediate_results:
                self.logger.error(f"Missing required input: {key}")
                return False
        
        # Validate shot blocks structure
        shot_blocks = context.intermediate_results.get('shot_blocks')
        if not isinstance(shot_blocks, list):
            self.logger.error("shot_blocks must be a list")
            return False
            
        return True
    
    async def process(self, context: ProcessingContext) -> NodeResult:
        """Process visual effects and transitions"""
        try:
            self.logger.info("Starting effects and transitions processing")
            
            # Extract input data
            shot_blocks = context.intermediate_results['shot_blocks']
            video_duration = context.intermediate_results['video_duration']
            emotion_analysis = context.intermediate_results.get('emotion_analysis', {})
            subtitle_data = context.intermediate_results.get('subtitle_tracks', [])
            
            # Step 1: Analyze scene requirements and generate effects plan
            effects_plan = await self._analyze_scene_requirements(
                shot_blocks, emotion_analysis, context
            )
            
            # Step 2: Generate color grading configuration
            color_grading = await self._generate_color_grading(
                effects_plan, emotion_analysis, context
            )
            
            # Step 3: Design transitions between scenes
            transitions = await self._design_scene_transitions(
                shot_blocks, effects_plan, context
            )
            
            # Step 4: Create animation sequences
            animations = await self._create_animation_sequences(
                shot_blocks, subtitle_data, effects_plan, context
            )
            
            # Step 5: Generate effects timeline
            effects_timeline = await self._generate_effects_timeline(
                shot_blocks, transitions, animations, color_grading, context
            )
            
            # Step 6: Create rendering configuration
            rendering_config = await self._create_rendering_config(
                effects_timeline, video_duration, context
            )
            
            # Prepare result
            effects_data = {
                'effects_timeline': effects_timeline,
                'transitions': transitions,
                'color_grading_config': color_grading,
                'animation_sequences': animations,
                'rendering_config': rendering_config,
                'effects_metadata': {
                    'total_effects': len(effects_timeline),
                    'transition_count': len(transitions),
                    'animation_count': len(animations),
                    'color_profile': self.config.color_profile,
                    'quality_level': self.config.quality_level,
                    'processing_settings': {
                        'gpu_acceleration': self.config.gpu_acceleration,
                        'motion_blur': self.config.motion_blur,
                        'bloom_effect': self.config.bloom_effect,
                        'particle_effects': self.config.particle_effects
                    }
                }
            }
            
            return NodeResult(
                status=NodeStatus.COMPLETED,
                data=effects_data,
                next_nodes=['render_compositor']
            )
            
        except Exception as e:
            self.logger.error(f"Effects processing failed: {e}")
            return NodeResult(
                status=NodeStatus.FAILED,
                error_message=str(e)
            )
    
    async def _analyze_scene_requirements(
        self,
        shot_blocks: List[Dict[str, Any]],
        emotion_analysis: Dict[str, Any],
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Analyze scene requirements and generate effects plan"""
        self.logger.info("Analyzing scene requirements")
        
        primary_emotion = emotion_analysis.get('primary_emotion', '中性')
        emotion_intensity = emotion_analysis.get('intensity', 0.5)
        
        # Use AI to analyze scene requirements
        scene_analysis_prompt = self.prompt_manager.render_prompt(
            'scene_effects_analysis',
            {
                'shot_blocks': shot_blocks,
                'primary_emotion': primary_emotion,
                'emotion_intensity': emotion_intensity,
                'color_profile': self.config.color_profile
            }
        )
        
        if scene_analysis_prompt:
            try:
                ai_response = await self.ai_service.generate_content(scene_analysis_prompt)
                effects_suggestions = self._parse_effects_suggestions(ai_response.content)
            except Exception as e:
                self.logger.warning(f"AI scene analysis failed: {e}")
                effects_suggestions = self._get_default_effects_plan(primary_emotion)
        else:
            effects_suggestions = self._get_default_effects_plan(primary_emotion)
        
        # Build comprehensive effects plan
        effects_plan = {
            'primary_emotion': primary_emotion,
            'intensity_level': emotion_intensity,
            'recommended_effects': effects_suggestions,
            'scene_analysis': self._analyze_individual_scenes(shot_blocks),
            'overall_style': self._determine_visual_style(primary_emotion),
            'pacing_analysis': self._analyze_pacing(shot_blocks)
        }
        
        return effects_plan
    
    async def _generate_color_grading(
        self,
        effects_plan: Dict[str, Any],
        emotion_analysis: Dict[str, Any],
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Generate color grading configuration"""
        self.logger.info("Generating color grading configuration")
        
        if not self.config.enable_color_grading:
            return {'enabled': False}
        
        primary_emotion = effects_plan.get('primary_emotion', '中性')
        intensity = effects_plan.get('intensity_level', 0.5)
        
        # Define color profiles for different emotions
        color_profiles = {
            '励志': {
                'temperature': 200,  # Warmer
                'tint': 50,
                'exposure': 0.2,
                'shadows': 0.1,
                'highlights': -0.1,
                'saturation': 1.15,
                'vibrance': 0.2,
                'curve_adjustments': 'slight_s_curve'
            },
            '专业': {
                'temperature': 0,  # Neutral
                'tint': 0,
                'exposure': 0.0,
                'shadows': 0.05,
                'highlights': -0.05,
                'saturation': 1.0,
                'vibrance': 0.1,
                'curve_adjustments': 'linear'
            },
            '创新': {
                'temperature': -100,  # Cooler
                'tint': -25,
                'exposure': 0.1,
                'shadows': 0.0,
                'highlights': -0.1,
                'saturation': 1.2,
                'vibrance': 0.3,
                'curve_adjustments': 'tech_curve'
            },
            '温馨': {
                'temperature': 300,  # Very warm
                'tint': 75,
                'exposure': 0.15,
                'shadows': 0.2,
                'highlights': -0.1,
                'saturation': 1.1,
                'vibrance': 0.15,
                'curve_adjustments': 'warm_curve'
            }
        }
        
        base_profile = color_profiles.get(primary_emotion, color_profiles['专业'])
        
        # Adjust intensity based on emotion strength
        adjusted_profile = {}
        for key, value in base_profile.items():
            if isinstance(value, (int, float)) and key != 'saturation':
                adjusted_profile[key] = value * intensity
            else:
                adjusted_profile[key] = value
        
        # Add profile-specific enhancements
        if self.config.color_profile == "cinematic":
            adjusted_profile.update({
                'film_grain': 0.1 * intensity,
                'vignette': 0.2,
                'contrast': 1.1
            })
        elif self.config.color_profile == "vibrant":
            adjusted_profile.update({
                'saturation': min(1.5, adjusted_profile.get('saturation', 1.0) * 1.2),
                'vibrance': adjusted_profile.get('vibrance', 0.0) + 0.2,
                'clarity': 0.15
            })
        elif self.config.color_profile == "vintage":
            adjusted_profile.update({
                'film_grain': 0.3,
                'vignette': 0.4,
                'fade_amount': 0.1,
                'split_toning_highlights': '#FFF8E1',
                'split_toning_shadows': '#3E2723'
            })
        
        return {
            'enabled': True,
            'profile_name': f"{primary_emotion}_{self.config.color_profile}",
            'adjustments': adjusted_profile,
            'lut_file': self._select_lut_file(primary_emotion, self.config.color_profile),
            'apply_to_all_scenes': True
        }
    
    async def _design_scene_transitions(
        self,
        shot_blocks: List[Dict[str, Any]],
        effects_plan: Dict[str, Any],
        context: ProcessingContext
    ) -> List[Dict[str, Any]]:
        """Design transitions between scenes"""
        self.logger.info("Designing scene transitions")
        
        if not self.config.enable_transitions or len(shot_blocks) <= 1:
            return []
        
        transitions = []
        pacing_analysis = effects_plan.get('pacing_analysis', {})
        overall_style = effects_plan.get('overall_style', 'standard')
        
        for i in range(len(shot_blocks) - 1):
            current_shot = shot_blocks[i]
            next_shot = shot_blocks[i + 1]
            
            # Calculate transition timing
            current_end = current_shot.get('start_time', 0) + current_shot.get('duration', 0)
            next_start = next_shot.get('start_time', current_end)
            
            # Determine transition type based on scene content and emotion
            transition_type = self._select_transition_type(
                current_shot, next_shot, overall_style, pacing_analysis
            )
            
            # Calculate appropriate duration
            transition_duration = self._calculate_transition_duration(
                transition_type, current_shot, next_shot
            )
            
            transition = {
                'transition_id': f'trans_{i+1:03d}',
                'type': transition_type.value,
                'start_time': current_end - (transition_duration / 2),
                'duration': transition_duration,
                'from_shot': current_shot.get('shot_id', f'shot_{i+1}'),
                'to_shot': next_shot.get('shot_id', f'shot_{i+2}'),
                'parameters': self._get_transition_parameters(
                    transition_type, current_shot, next_shot
                ),
                'easing': self._select_easing_function(transition_type, overall_style)
            }
            
            transitions.append(transition)
        
        return transitions
    
    async def _create_animation_sequences(
        self,
        shot_blocks: List[Dict[str, Any]],
        subtitle_data: List[Dict[str, Any]],
        effects_plan: Dict[str, Any],
        context: ProcessingContext
    ) -> List[Dict[str, Any]]:
        """Create animation sequences for text, logos, and elements"""
        self.logger.info("Creating animation sequences")
        
        if not self.config.enable_animations:
            return []
        
        animations = []
        animation_style = effects_plan.get('overall_style', 'standard')
        
        # Text animations for subtitles
        for subtitle in subtitle_data:
            if isinstance(subtitle, dict) and 'timing' in subtitle:
                text_animation = {
                    'animation_id': f'text_anim_{len(animations)+1:03d}',
                    'type': EffectType.TEXT_ANIMATION.value,
                    'target': 'subtitle',
                    'target_id': subtitle.get('subtitle_id'),
                    'start_time': subtitle['timing']['start'],
                    'duration': subtitle['timing']['duration'],
                    'animation_type': self._select_text_animation_type(animation_style),
                    'parameters': self._get_text_animation_parameters(subtitle, animation_style)
                }
                animations.append(text_animation)
        
        # Logo animations
        logo_moments = self._identify_logo_moments(shot_blocks)
        for moment in logo_moments:
            logo_animation = {
                'animation_id': f'logo_anim_{len(animations)+1:03d}',
                'type': EffectType.LOGO_ANIMATION.value,
                'target': 'logo',
                'start_time': moment['start_time'],
                'duration': moment['duration'],
                'animation_type': 'elegant_entrance',
                'parameters': {
                    'entrance_type': 'fade_scale',
                    'scale_from': 0.8,
                    'scale_to': 1.0,
                    'opacity_from': 0.0,
                    'opacity_to': 1.0,
                    'easing': 'ease_out_cubic'
                }
            }
            animations.append(logo_animation)
        
        # Element animations based on shot content
        for shot in shot_blocks:
            shot_animations = self._create_shot_animations(shot, animation_style)
            animations.extend(shot_animations)
        
        return animations
    
    async def _generate_effects_timeline(
        self,
        shot_blocks: List[Dict[str, Any]],
        transitions: List[Dict[str, Any]],
        animations: List[Dict[str, Any]],
        color_grading: Dict[str, Any],
        context: ProcessingContext
    ) -> List[Dict[str, Any]]:
        """Generate comprehensive effects timeline"""
        self.logger.info("Generating effects timeline")
        
        timeline = []
        
        # Add color grading as base effect
        if color_grading.get('enabled', False):
            timeline.append({
                'effect_id': 'color_grading_base',
                'type': EffectType.COLOR_GRADING.value,
                'start_time': 0.0,
                'duration': sum(shot.get('duration', 0) for shot in shot_blocks),
                'priority': 1,  # Base layer
                'parameters': color_grading['adjustments'],
                'enabled': True
            })
        
        # Add transitions
        for transition in transitions:
            timeline.append({
                'effect_id': transition['transition_id'],
                'type': EffectType.TRANSITION.value,
                'start_time': transition['start_time'],
                'duration': transition['duration'],
                'priority': 10,  # High priority
                'parameters': transition['parameters'],
                'enabled': True
            })
        
        # Add animations
        for animation in animations:
            timeline.append({
                'effect_id': animation['animation_id'],
                'type': animation['type'],
                'start_time': animation['start_time'],
                'duration': animation['duration'],
                'priority': 5,  # Medium priority
                'parameters': animation['parameters'],
                'enabled': True
            })
        
        # Add shot-specific effects
        for shot in shot_blocks:
            shot_effects = self._generate_shot_effects(shot)
            timeline.extend(shot_effects)
        
        # Sort timeline by start time and priority
        timeline.sort(key=lambda x: (x['start_time'], -x['priority']))
        
        return timeline
    
    async def _create_rendering_config(
        self,
        effects_timeline: List[Dict[str, Any]],
        video_duration: float,
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Create rendering configuration for effects processing"""
        self.logger.info("Creating rendering configuration")
        
        # Calculate render complexity
        complexity_score = len(effects_timeline) * 0.1
        has_transitions = any(e['type'] == EffectType.TRANSITION.value for e in effects_timeline)
        has_particles = any(e.get('parameters', {}).get('particle_effects', False) for e in effects_timeline)
        
        if has_transitions:
            complexity_score += 0.3
        if has_particles:
            complexity_score += 0.5
        if self.config.particle_effects:
            complexity_score += 0.2
        
        # Determine render settings based on quality and complexity
        quality_settings = {
            'low': {
                'resolution_scale': 0.5,
                'fps': 24,
                'bitrate': '2000k',
                'codec': 'h264',
                'preset': 'fast'
            },
            'medium': {
                'resolution_scale': 0.75,
                'fps': 30,
                'bitrate': '5000k',
                'codec': 'h264',
                'preset': 'medium'
            },
            'high': {
                'resolution_scale': 1.0,
                'fps': 30,
                'bitrate': '8000k',
                'codec': 'h264',
                'preset': 'slow'
            },
            'ultra': {
                'resolution_scale': 1.0,
                'fps': 60,
                'bitrate': '15000k',
                'codec': 'h265',
                'preset': 'veryslow'
            }
        }
        
        base_settings = quality_settings.get(self.config.quality_level, quality_settings['high'])
        
        # Adjust settings based on complexity
        if complexity_score > 1.0:
            base_settings['render_passes'] = 2
            base_settings['memory_limit'] = '4GB'
        else:
            base_settings['render_passes'] = 1
            base_settings['memory_limit'] = '2GB'
        
        rendering_config = {
            'quality_level': self.config.quality_level,
            'complexity_score': complexity_score,
            'settings': base_settings,
            'effects_processing': {
                'gpu_acceleration': self.config.gpu_acceleration,
                'motion_blur': self.config.motion_blur,
                'bloom_effect': self.config.bloom_effect,
                'particle_effects': self.config.particle_effects,
                'anti_aliasing': self.config.quality_level in ['high', 'ultra']
            },
            'optimization': {
                'enable_caching': True,
                'parallel_processing': self.config.gpu_acceleration,
                'memory_optimization': complexity_score > 0.8
            },
            'output_format': {
                'container': 'mp4',
                'video_codec': base_settings['codec'],
                'audio_codec': 'aac',
                'pixel_format': 'yuv420p'
            }
        }
        
        return rendering_config
    
    def _analyze_individual_scenes(self, shot_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze individual scenes for specific requirements"""
        scene_analyses = []
        
        for shot in shot_blocks:
            description = shot.get('description', '').lower()
            emotion_hint = shot.get('emotion_hint', '中性')
            
            analysis = {
                'shot_id': shot.get('shot_id'),
                'emotion': emotion_hint,
                'visual_style': self._infer_visual_style(description),
                'movement_type': self._infer_movement(description),
                'lighting_mood': self._infer_lighting_mood(emotion_hint),
                'suggested_effects': self._suggest_shot_effects(description, emotion_hint)
            }
            
            scene_analyses.append(analysis)
        
        return scene_analyses
    
    def _determine_visual_style(self, primary_emotion: str) -> str:
        """Determine overall visual style based on emotion"""
        style_mapping = {
            '励志': 'uplifting',
            '专业': 'clean',
            '创新': 'modern',
            '温馨': 'warm',
            '激动': 'dynamic',
            '中性': 'standard'
        }
        
        return style_mapping.get(primary_emotion, 'standard')
    
    def _analyze_pacing(self, shot_blocks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze video pacing for transition timing"""
        if not shot_blocks:
            return {'average_duration': 3.0, 'tempo': 'medium'}
        
        durations = [shot.get('duration', 3.0) for shot in shot_blocks]
        average_duration = sum(durations) / len(durations)
        
        if average_duration < 2.0:
            tempo = 'fast'
        elif average_duration > 4.0:
            tempo = 'slow'
        else:
            tempo = 'medium'
        
        return {
            'average_duration': average_duration,
            'tempo': tempo,
            'variation': max(durations) - min(durations),
            'durations': durations
        }
    
    def _select_transition_type(
        self,
        current_shot: Dict[str, Any],
        next_shot: Dict[str, Any],
        style: str,
        pacing: Dict[str, Any]
    ) -> TransitionType:
        """Select appropriate transition type"""
        tempo = pacing.get('tempo', 'medium')
        current_emotion = current_shot.get('emotion_hint', '中性')
        next_emotion = next_shot.get('emotion_hint', '中性')
        
        # Fast tempo prefers quick cuts or cross-fades
        if tempo == 'fast':
            return random.choice([TransitionType.CUT, TransitionType.CROSS_FADE])
        
        # Emotional changes might need softer transitions
        if current_emotion != next_emotion:
            return TransitionType.FADE
        
        # Style-based selection
        if style == 'modern':
            return random.choice([TransitionType.SLIDE, TransitionType.ZOOM])
        elif style == 'warm':
            return TransitionType.FADE
        elif style == 'dynamic':
            return random.choice([TransitionType.ROTATE, TransitionType.WIPE])
        else:
            return TransitionType.CROSS_FADE
    
    def _calculate_transition_duration(
        self,
        transition_type: TransitionType,
        current_shot: Dict[str, Any],
        next_shot: Dict[str, Any]
    ) -> float:
        """Calculate appropriate transition duration"""
        base_duration = self.config.transition_duration
        
        # Adjust based on transition type
        type_multipliers = {
            TransitionType.CUT: 0.0,
            TransitionType.FADE: 1.0,
            TransitionType.CROSS_FADE: 0.8,
            TransitionType.SLIDE: 1.2,
            TransitionType.ZOOM: 1.5,
            TransitionType.ROTATE: 2.0,
            TransitionType.WIPE: 1.8,
            TransitionType.IRIS: 2.2
        }
        
        return base_duration * type_multipliers.get(transition_type, 1.0)
    
    def _get_transition_parameters(
        self,
        transition_type: TransitionType,
        current_shot: Dict[str, Any],
        next_shot: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Get parameters for specific transition type"""
        base_params = {'blend_mode': 'normal', 'intensity': 1.0}
        
        if transition_type == TransitionType.SLIDE:
            directions = ['left', 'right', 'up', 'down']
            base_params.update({
                'direction': random.choice(directions),
                'easing': 'ease_in_out'
            })
        elif transition_type == TransitionType.ZOOM:
            base_params.update({
                'zoom_direction': random.choice(['in', 'out']),
                'center_point': [0.5, 0.5]  # Center of screen
            })
        elif transition_type == TransitionType.ROTATE:
            base_params.update({
                'rotation_angle': random.choice([90, 180, 270]),
                'rotation_center': [0.5, 0.5]
            })
        
        return base_params
    
    def _select_easing_function(self, transition_type: TransitionType, style: str) -> str:
        """Select appropriate easing function"""
        if style == 'dynamic':
            return 'ease_out_back'
        elif style == 'warm':
            return 'ease_in_out_sine'
        elif style == 'modern':
            return 'ease_out_cubic'
        else:
            return 'ease_in_out'
    
    def _parse_effects_suggestions(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response for effects suggestions"""
        try:
            # Try to extract JSON
            import re
            json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
        except:
            pass
        
        # Fallback parsing
        suggestions = []
        lines = ai_response.split('\n')
        for line in lines:
            if any(keyword in line.lower() for keyword in ['effect', 'filter', 'color', 'animation']):
                suggestions.append({
                    'name': line.strip(),
                    'type': 'general',
                    'intensity': 0.5
                })
        
        return suggestions if suggestions else self._get_default_effects_plan('中性')
    
    def _get_default_effects_plan(self, emotion: str) -> List[Dict[str, Any]]:
        """Get default effects plan for emotion"""
        default_plans = {
            '励志': [
                {'name': 'warm_color_grade', 'type': 'color_grading', 'intensity': 0.7},
                {'name': 'subtle_glow', 'type': 'filter', 'intensity': 0.3},
                {'name': 'smooth_transitions', 'type': 'transition', 'intensity': 0.8}
            ],
            '专业': [
                {'name': 'neutral_color_grade', 'type': 'color_grading', 'intensity': 0.5},
                {'name': 'clean_cuts', 'type': 'transition', 'intensity': 0.6}
            ],
            '创新': [
                {'name': 'cool_color_grade', 'type': 'color_grading', 'intensity': 0.8},
                {'name': 'tech_transitions', 'type': 'transition', 'intensity': 0.9},
                {'name': 'subtle_particle', 'type': 'particle', 'intensity': 0.4}
            ]
        }
        
        return default_plans.get(emotion, default_plans['专业'])
    
    def _select_lut_file(self, emotion: str, color_profile: str) -> str:
        """Select appropriate LUT file for color grading"""
        lut_mapping = {
            ('励志', 'cinematic'): 'warm_cinematic.cube',
            ('创新', 'cinematic'): 'cool_cinematic.cube',
            ('专业', 'natural'): 'neutral_natural.cube',
            ('温馨', 'vintage'): 'warm_vintage.cube'
        }
        
        return lut_mapping.get((emotion, color_profile), 'default_natural.cube')
    
    def _infer_visual_style(self, description: str) -> str:
        """Infer visual style from shot description"""
        if any(word in description for word in ['close', 'detail', 'focus']):
            return 'intimate'
        elif any(word in description for word in ['wide', 'landscape', 'overview']):
            return 'expansive'
        elif any(word in description for word in ['action', 'moving', 'dynamic']):
            return 'energetic'
        else:
            return 'standard'
    
    def _infer_movement(self, description: str) -> str:
        """Infer camera movement from description"""
        if any(word in description for word in ['pan', 'sweep']):
            return 'pan'
        elif any(word in description for word in ['zoom', 'push']):
            return 'zoom'
        elif any(word in description for word in ['track', 'follow']):
            return 'track'
        else:
            return 'static'
    
    def _infer_lighting_mood(self, emotion: str) -> str:
        """Infer lighting mood from emotion"""
        mood_mapping = {
            '励志': 'bright',
            '专业': 'neutral',
            '创新': 'cool',
            '温馨': 'warm',
            '激动': 'dramatic',
            '中性': 'balanced'
        }
        
        return mood_mapping.get(emotion, 'balanced')
    
    def _suggest_shot_effects(self, description: str, emotion: str) -> List[str]:
        """Suggest effects for specific shot"""
        effects = []
        
        # Based on description
        if 'logo' in description.lower():
            effects.extend(['logo_glow', 'elegant_entrance'])
        if any(word in description.lower() for word in ['product', 'feature']):
            effects.extend(['subtle_highlight', 'depth_of_field'])
        if 'text' in description.lower():
            effects.extend(['text_animation', 'reading_ease'])
        
        # Based on emotion
        if emotion == '励志':
            effects.extend(['warm_glow', 'subtle_rays'])
        elif emotion == '创新':
            effects.extend(['tech_grid', 'particle_subtle'])
        
        return effects
    
    def _select_text_animation_type(self, style: str) -> str:
        """Select text animation type based on style"""
        animations = {
            'uplifting': 'fade_up_scale',
            'clean': 'simple_fade',
            'modern': 'slide_in',
            'warm': 'gentle_fade',
            'dynamic': 'bounce_in',
            'standard': 'fade_in'
        }
        
        return animations.get(style, 'fade_in')
    
    def _get_text_animation_parameters(
        self, 
        subtitle: Dict[str, Any], 
        style: str
    ) -> Dict[str, Any]:
        """Get text animation parameters"""
        return {
            'entrance_duration': 0.3,
            'exit_duration': 0.2,
            'easing_in': 'ease_out',
            'easing_out': 'ease_in',
            'scale_from': 0.9,
            'scale_to': 1.0,
            'opacity_from': 0.0,
            'opacity_to': 1.0
        }
    
    def _identify_logo_moments(self, shot_blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify moments where logo should appear"""
        logo_moments = []
        
        for shot in shot_blocks:
            description = shot.get('description', '').lower()
            if 'logo' in description or shot.get('shot_id', '').startswith('shot_001'):
                logo_moments.append({
                    'start_time': shot.get('start_time', 0),
                    'duration': min(3.0, shot.get('duration', 3.0)),
                    'shot_id': shot.get('shot_id')
                })
        
        return logo_moments
    
    def _create_shot_animations(self, shot: Dict[str, Any], style: str) -> List[Dict[str, Any]]:
        """Create animations specific to a shot"""
        animations = []
        description = shot.get('description', '').lower()
        
        # Add subtle entrance animation for key elements
        if any(keyword in description for keyword in ['feature', 'product', 'highlight']):
            animations.append({
                'animation_id': f'element_anim_{shot.get("shot_id", "")}_001',
                'type': EffectType.ANIMATION.value,
                'target': 'main_element',
                'start_time': shot.get('start_time', 0) + 0.2,
                'duration': 0.8,
                'animation_type': 'subtle_entrance',
                'parameters': {
                    'scale_from': 0.95,
                    'scale_to': 1.0,
                    'opacity_from': 0.0,
                    'opacity_to': 1.0,
                    'easing': 'ease_out_cubic'
                }
            })
        
        return animations
    
    def _generate_shot_effects(self, shot: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate effects specific to a shot"""
        effects = []
        shot_id = shot.get('shot_id', '')
        start_time = shot.get('start_time', 0)
        duration = shot.get('duration', 3.0)
        
        # Add subtle vignette for focus
        if 'close' in shot.get('description', '').lower():
            effects.append({
                'effect_id': f'vignette_{shot_id}',
                'type': EffectType.FILTER.value,
                'start_time': start_time,
                'duration': duration,
                'priority': 3,
                'parameters': {
                    'vignette_amount': 0.2,
                    'vignette_roundness': 0.8,
                    'vignette_feather': 0.6
                },
                'enabled': True
            })
        
        return effects