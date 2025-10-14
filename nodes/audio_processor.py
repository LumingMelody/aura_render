"""
Audio Processing Node

Handles all audio-related processing for video generation including
background music selection, voice synthesis, audio effects, and mixing.
"""

import asyncio
import json
import random
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path

from .base_node import BaseNode, NodeConfig, NodeResult, ProcessingContext, NodeStatus
from ai_service import get_enhanced_qwen_service, get_prompt_manager


@dataclass
class AudioProcessingConfig(NodeConfig):
    """Configuration for audio processing node"""
    voice_engine: str = "edge_tts"  # Options: edge_tts, azure_tts, local_tts
    voice_name: str = "zh-CN-XiaoxiaoNeural"
    voice_speed: float = 1.0
    voice_pitch: float = 0.0
    music_library_path: Optional[str] = None
    audio_format: str = "wav"
    sample_rate: int = 44100
    channels: int = 2
    enable_audio_effects: bool = True
    normalize_audio: bool = True
    fade_in_duration: float = 0.5
    fade_out_duration: float = 1.0


class AudioProcessingNode(BaseNode):
    """Audio processing node for video generation pipeline"""
    
    def __init__(self, config: AudioProcessingConfig):
        super().__init__(config)
        self.config: AudioProcessingConfig = config
        self.ai_service = get_enhanced_qwen_service()
        self.prompt_manager = get_prompt_manager()
        
    def get_required_inputs(self) -> List[str]:
        """Required inputs for audio processing"""
        return ['script_content', 'video_duration', 'emotion_analysis']
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Output schema for audio processing"""
        return {
            'voice_audio_path': 'str',
            'background_music_path': 'str', 
            'mixed_audio_path': 'str',
            'audio_segments': 'list',
            'audio_metadata': 'dict',
            'audio_duration': 'float'
        }
    
    def validate_input(self, context: ProcessingContext) -> bool:
        """Validate input for audio processing"""
        required_keys = ['script_content', 'video_duration']
        
        for key in required_keys:
            if key not in context.intermediate_results:
                self.logger.error(f"Missing required input: {key}")
                return False
        
        # Validate script content structure
        script_content = context.intermediate_results.get('script_content')
        if not isinstance(script_content, (dict, list)):
            self.logger.error("script_content must be a dictionary or list")
            return False
            
        return True
    
    async def process(self, context: ProcessingContext) -> NodeResult:
        """Process audio generation and mixing"""
        try:
            self.logger.info("Starting audio processing")
            
            # Extract input data
            script_content = context.intermediate_results['script_content']
            video_duration = context.intermediate_results['video_duration']
            emotion_analysis = context.intermediate_results.get('emotion_analysis', {})
            
            # Step 1: Generate voice narration
            voice_result = await self._generate_voice_narration(
                script_content, emotion_analysis, context
            )
            
            # Step 2: Select and process background music
            music_result = await self._select_background_music(
                emotion_analysis, video_duration, context
            )
            
            # Step 3: Apply audio effects
            effects_result = await self._apply_audio_effects(
                voice_result, music_result, context
            )
            
            # Step 4: Mix audio tracks
            final_audio = await self._mix_audio_tracks(
                effects_result, video_duration, context
            )
            
            # Prepare result
            audio_data = {
                'voice_audio_path': voice_result.get('audio_path'),
                'background_music_path': music_result.get('audio_path'),
                'mixed_audio_path': final_audio.get('mixed_path'),
                'audio_segments': final_audio.get('segments', []),
                'audio_metadata': {
                    'voice_settings': {
                        'engine': self.config.voice_engine,
                        'voice_name': self.config.voice_name,
                        'speed': self.config.voice_speed,
                        'pitch': self.config.voice_pitch
                    },
                    'music_metadata': music_result.get('metadata', {}),
                    'effects_applied': effects_result.get('effects', []),
                    'total_duration': final_audio.get('duration', 0.0)
                },
                'audio_duration': final_audio.get('duration', 0.0)
            }
            
            return NodeResult(
                status=NodeStatus.COMPLETED,
                data=audio_data,
                next_nodes=['subtitle_generator', 'effects_processor']
            )
            
        except Exception as e:
            self.logger.error(f"Audio processing failed: {e}")
            return NodeResult(
                status=NodeStatus.FAILED,
                error_message=str(e)
            )
    
    async def _generate_voice_narration(
        self, 
        script_content: Union[Dict, List], 
        emotion_analysis: Dict[str, Any],
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Generate voice narration from script"""
        self.logger.info("Generating voice narration")
        
        # Extract text content from script
        if isinstance(script_content, list):
            text_content = " ".join([segment.get('text', '') for segment in script_content])
        elif isinstance(script_content, dict):
            text_content = script_content.get('full_script', '')
        else:
            text_content = str(script_content)
        
        # Analyze emotion for voice parameters
        primary_emotion = emotion_analysis.get('primary_emotion', '中性')
        voice_adjustments = self._get_voice_adjustments(primary_emotion)
        
        # Simulate voice generation (in real implementation, call TTS API)
        voice_segments = []
        current_time = 0.0
        
        # Split text into segments for better control
        sentences = self._split_text_into_sentences(text_content)
        
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                # Estimate duration (roughly 150 words per minute)
                word_count = len(sentence.split())
                estimated_duration = (word_count / 150) * 60
                
                segment = {
                    'segment_id': f'voice_{i+1:03d}',
                    'text': sentence.strip(),
                    'start_time': current_time,
                    'duration': estimated_duration,
                    'voice_settings': voice_adjustments,
                    'emotion_hint': primary_emotion
                }
                
                voice_segments.append(segment)
                current_time += estimated_duration
        
        # Simulate audio file path
        audio_path = f"/tmp/voice_narration_{context.task_id}.{self.config.audio_format}"
        
        return {
            'audio_path': audio_path,
            'segments': voice_segments,
            'total_duration': current_time,
            'voice_settings': voice_adjustments
        }
    
    async def _select_background_music(
        self,
        emotion_analysis: Dict[str, Any],
        video_duration: float,
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Select and process background music"""
        self.logger.info("Selecting background music")
        
        # Get emotion-based music selection
        primary_emotion = emotion_analysis.get('primary_emotion', '中性')
        music_category = self._map_emotion_to_music_category(primary_emotion)
        
        # Use AI to help select music
        music_prompt = self.prompt_manager.render_prompt(
            'music_selection',
            {
                'emotion': primary_emotion,
                'duration': video_duration,
                'category': music_category
            }
        )
        
        if music_prompt:
            try:
                ai_response = await self.ai_service.generate_content(music_prompt)
                music_suggestions = self._parse_music_suggestions(ai_response.content)
            except Exception as e:
                self.logger.warning(f"AI music selection failed: {e}")
                music_suggestions = self._get_default_music_suggestions(music_category)
        else:
            music_suggestions = self._get_default_music_suggestions(music_category)
        
        # Select the best music track
        selected_music = random.choice(music_suggestions) if music_suggestions else {
            'name': 'default_ambient.mp3',
            'category': 'ambient',
            'mood': '中性',
            'tempo': 'medium'
        }
        
        # Simulate music processing
        music_path = f"/tmp/background_music_{context.task_id}.{self.config.audio_format}"
        
        return {
            'audio_path': music_path,
            'selected_track': selected_music,
            'metadata': {
                'original_duration': video_duration + 2.0,  # Slightly longer for transitions
                'loop_points': self._calculate_loop_points(video_duration),
                'volume_adjustment': -20.0,  # Background level in dB
                'fade_in': self.config.fade_in_duration,
                'fade_out': self.config.fade_out_duration
            }
        }
    
    async def _apply_audio_effects(
        self,
        voice_result: Dict[str, Any],
        music_result: Dict[str, Any],
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Apply audio effects to voice and music"""
        self.logger.info("Applying audio effects")
        
        if not self.config.enable_audio_effects:
            return {'voice': voice_result, 'music': music_result, 'effects': []}
        
        applied_effects = []
        
        # Voice effects
        voice_effects = []
        if self.config.normalize_audio:
            voice_effects.append('normalize')
            applied_effects.append('voice_normalize')
        
        # Add subtle reverb for professional sound
        voice_effects.append('reverb_light')
        applied_effects.append('voice_reverb')
        
        # Music effects
        music_effects = []
        if self.config.normalize_audio:
            music_effects.append('normalize')
            applied_effects.append('music_normalize')
        
        # EQ for background music
        music_effects.append('eq_background')
        applied_effects.append('music_eq')
        
        # Simulate effects processing
        processed_voice_path = f"/tmp/voice_processed_{context.task_id}.{self.config.audio_format}"
        processed_music_path = f"/tmp/music_processed_{context.task_id}.{self.config.audio_format}"
        
        return {
            'voice': {
                **voice_result,
                'processed_path': processed_voice_path,
                'effects_applied': voice_effects
            },
            'music': {
                **music_result, 
                'processed_path': processed_music_path,
                'effects_applied': music_effects
            },
            'effects': applied_effects
        }
    
    async def _mix_audio_tracks(
        self,
        effects_result: Dict[str, Any],
        video_duration: float,
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Mix voice and music tracks together"""
        self.logger.info("Mixing audio tracks")
        
        voice_data = effects_result['voice']
        music_data = effects_result['music']
        
        # Calculate mixing parameters
        voice_volume = 0.8  # 80% volume for voice
        music_volume = 0.3  # 30% volume for background music
        
        # Create mixing timeline
        mix_segments = []
        
        # Add voice segments with background music
        for segment in voice_data.get('segments', []):
            mix_segments.append({
                'type': 'voice_with_music',
                'start_time': segment['start_time'],
                'duration': segment['duration'],
                'voice_volume': voice_volume,
                'music_volume': music_volume * 0.5,  # Lower music during speech
                'voice_segment': segment,
                'transition': 'cross_fade'
            })
        
        # Add music-only segments (intro, outro, between voice)
        voice_segments = voice_data.get('segments', [])
        if voice_segments:
            # Intro music
            first_voice_start = voice_segments[0]['start_time']
            if first_voice_start > 0:
                mix_segments.insert(0, {
                    'type': 'music_only',
                    'start_time': 0.0,
                    'duration': first_voice_start,
                    'music_volume': music_volume,
                    'transition': 'fade_in'
                })
            
            # Outro music
            last_segment = voice_segments[-1]
            voice_end_time = last_segment['start_time'] + last_segment['duration']
            if voice_end_time < video_duration:
                mix_segments.append({
                    'type': 'music_only',
                    'start_time': voice_end_time,
                    'duration': video_duration - voice_end_time,
                    'music_volume': music_volume,
                    'transition': 'fade_out'
                })
        
        # Simulate final mixing
        mixed_audio_path = f"/tmp/mixed_audio_{context.task_id}.{self.config.audio_format}"
        
        return {
            'mixed_path': mixed_audio_path,
            'segments': mix_segments,
            'duration': video_duration,
            'mixing_metadata': {
                'voice_volume': voice_volume,
                'music_volume': music_volume,
                'sample_rate': self.config.sample_rate,
                'channels': self.config.channels,
                'format': self.config.audio_format
            }
        }
    
    def _get_voice_adjustments(self, emotion: str) -> Dict[str, Any]:
        """Get voice parameter adjustments based on emotion"""
        emotion_mappings = {
            '励志': {'speed': 1.1, 'pitch': 0.1, 'energy': 'high'},
            '专业': {'speed': 1.0, 'pitch': 0.0, 'energy': 'medium'},
            '创新': {'speed': 1.05, 'pitch': 0.05, 'energy': 'medium-high'},
            '温馨': {'speed': 0.95, 'pitch': -0.05, 'energy': 'low'},
            '激动': {'speed': 1.15, 'pitch': 0.15, 'energy': 'very-high'},
            '中性': {'speed': 1.0, 'pitch': 0.0, 'energy': 'medium'}
        }
        
        return emotion_mappings.get(emotion, emotion_mappings['中性'])
    
    def _map_emotion_to_music_category(self, emotion: str) -> str:
        """Map emotion to music category"""
        emotion_to_music = {
            '励志': 'inspirational',
            '专业': 'corporate', 
            '创新': 'tech',
            '温馨': 'ambient',
            '激动': 'energetic',
            '中性': 'neutral'
        }
        
        return emotion_to_music.get(emotion, 'neutral')
    
    def _split_text_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for TTS processing"""
        import re
        
        # Simple sentence splitting (can be enhanced)
        sentences = re.split(r'[。！？.!?]', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _parse_music_suggestions(self, ai_response: str) -> List[Dict[str, Any]]:
        """Parse AI response for music suggestions"""
        try:
            # Try to extract JSON from AI response
            import re
            json_match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            if json_match:
                suggestions_data = json.loads(json_match.group())
                return suggestions_data.get('suggestions', [])
        except:
            pass
        
        # Fallback: extract text-based suggestions
        suggestions = []
        lines = ai_response.split('\n')
        for line in lines:
            if 'track' in line.lower() or 'music' in line.lower():
                suggestions.append({
                    'name': line.strip(),
                    'category': 'auto_selected',
                    'confidence': 0.5
                })
        
        return suggestions
    
    def _get_default_music_suggestions(self, category: str) -> List[Dict[str, Any]]:
        """Get default music suggestions for category"""
        default_tracks = {
            'inspirational': [
                {'name': 'uplifting_piano.mp3', 'mood': '励志', 'tempo': 'medium'},
                {'name': 'motivational_strings.mp3', 'mood': '励志', 'tempo': 'fast'},
            ],
            'corporate': [
                {'name': 'professional_ambient.mp3', 'mood': '专业', 'tempo': 'slow'},
                {'name': 'business_light.mp3', 'mood': '专业', 'tempo': 'medium'},
            ],
            'tech': [
                {'name': 'modern_electronic.mp3', 'mood': '创新', 'tempo': 'medium'},
                {'name': 'futuristic_ambient.mp3', 'mood': '创新', 'tempo': 'medium'},
            ],
            'ambient': [
                {'name': 'soft_piano.mp3', 'mood': '温馨', 'tempo': 'slow'},
                {'name': 'gentle_strings.mp3', 'mood': '温馨', 'tempo': 'slow'},
            ],
            'energetic': [
                {'name': 'upbeat_electronic.mp3', 'mood': '激动', 'tempo': 'fast'},
                {'name': 'dynamic_beat.mp3', 'mood': '激动', 'tempo': 'very-fast'},
            ]
        }
        
        return default_tracks.get(category, [
            {'name': 'neutral_ambient.mp3', 'mood': '中性', 'tempo': 'medium'}
        ])
    
    def _calculate_loop_points(self, duration: float) -> List[Dict[str, float]]:
        """Calculate loop points for background music"""
        # Simple loop point calculation
        base_loop_duration = 30.0  # 30 second loops
        loops_needed = int(duration / base_loop_duration) + 1
        
        loop_points = []
        for i in range(loops_needed):
            loop_points.append({
                'start': i * base_loop_duration,
                'end': min((i + 1) * base_loop_duration, duration)
            })
        
        return loop_points