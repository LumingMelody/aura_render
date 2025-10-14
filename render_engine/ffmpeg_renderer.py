"""
FFmpeg-based Video Renderer

Production-ready video rendering using FFmpeg with comprehensive
error handling, progress tracking, and quality control.
"""

import os
import re
import json
import asyncio
import subprocess
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging

@dataclass
class RenderConfig:
    """FFmpeg rendering configuration"""
    # Video settings
    video_codec: str = "libx264"
    audio_codec: str = "aac"
    pixel_format: str = "yuv420p"
    frame_rate: int = 30
    
    # Quality settings
    crf: int = 23  # 0-51, lower = better quality
    preset: str = "medium"  # ultrafast, superfast, veryfast, faster, fast, medium, slow, slower, veryslow
    tune: Optional[str] = None  # film, animation, grain, stillimage, fastdecode, zerolatency
    
    # Resolution and bitrate
    resolution: Optional[tuple] = None  # (width, height) or None for auto
    video_bitrate: Optional[str] = None  # e.g., "2M", "5000k"
    audio_bitrate: str = "128k"
    
    # Hardware acceleration
    hardware_accel: Optional[str] = None  # cuda, videotoolbox, qsv, vaapi
    
    # Advanced options
    two_pass: bool = False
    custom_filters: List[str] = field(default_factory=list)
    metadata: Dict[str, str] = field(default_factory=dict)
    
    # Output settings
    format: str = "mp4"
    movflags: str = "faststart"  # For web optimization


class FFmpegRenderer:
    """FFmpeg-based video renderer with production capabilities"""
    
    def __init__(self, config: RenderConfig = None):
        self.config = config or RenderConfig()
        self.logger = logging.getLogger(__name__)
        self._validate_ffmpeg()
        
    def _validate_ffmpeg(self):
        """Validate FFmpeg installation"""
        try:
            result = subprocess.run(['ffmpeg', '-version'], 
                                  capture_output=True, text=True, timeout=10)
            if result.returncode != 0:
                raise RuntimeError("FFmpeg not properly installed")
            
            # Extract version info
            version_line = result.stdout.split('\n')[0]
            self.logger.info(f"FFmpeg available: {version_line}")
            
        except (subprocess.TimeoutExpired, FileNotFoundError) as e:
            raise RuntimeError(f"FFmpeg not available: {e}")
    
    async def render_video(
        self,
        composition_data: Dict[str, Any],
        output_path: str,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """
        Render video from composition data
        
        Args:
            composition_data: Video composition specification
            output_path: Output file path
            progress_callback: Optional progress update callback
            
        Returns:
            Render result with statistics and metadata
        """
        try:
            self.logger.info(f"Starting render: {output_path}")
            
            # Prepare render context
            render_context = await self._prepare_render_context(composition_data, output_path)
            
            # Generate FFmpeg commands
            commands = await self._generate_ffmpeg_commands(render_context)
            
            # Execute rendering
            result = await self._execute_render(commands, progress_callback)
            
            # Validate output
            validation_result = await self._validate_output(output_path)
            
            # Compile final result
            return {
                'success': True,
                'output_path': output_path,
                'render_time': result['render_time'],
                'file_size': result['file_size'],
                'validation': validation_result,
                'metadata': await self._extract_media_info(output_path),
                'statistics': result['statistics']
            }
            
        except Exception as e:
            self.logger.error(f"Render failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'output_path': output_path
            }
    
    async def _prepare_render_context(self, composition_data: Dict[str, Any], output_path: str) -> Dict[str, Any]:
        """Prepare rendering context from composition data"""
        
        # Ensure output directory exists
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Extract layers and timeline
        layers = composition_data.get('layers', [])
        effects_timeline = composition_data.get('effects_timeline', [])
        audio_path = composition_data.get('mixed_audio_path')
        subtitle_files = composition_data.get('subtitle_files', {})
        
        # Analyze composition requirements
        total_duration = max([
            layer.get('duration', 0) + layer.get('start_time', 0) 
            for layer in layers
        ] + [0])
        
        has_video_layers = any(layer.get('layer_type') == 'video' for layer in layers)
        has_audio = audio_path is not None
        has_subtitles = len(subtitle_files) > 0
        has_effects = len(effects_timeline) > 0
        
        return {
            'output_path': output_path,
            'layers': layers,
            'effects_timeline': effects_timeline,
            'audio_path': audio_path,
            'subtitle_files': subtitle_files,
            'total_duration': total_duration,
            'has_video_layers': has_video_layers,
            'has_audio': has_audio,
            'has_subtitles': has_subtitles,
            'has_effects': has_effects,
            'temp_dir': Path(output_path).parent / 'temp'
        }
    
    async def _generate_ffmpeg_commands(self, context: Dict[str, Any]) -> List[str]:
        """Generate FFmpeg commands for rendering"""
        commands = []
        
        # Create temporary directory
        temp_dir = context['temp_dir']
        temp_dir.mkdir(exist_ok=True)
        
        if context['has_video_layers']:
            # Generate video with effects
            video_cmd = await self._build_video_command(context)
            commands.append(video_cmd)
        
        if context['has_audio'] and context['has_video_layers']:
            # Combine video and audio
            final_cmd = await self._build_final_composition_command(context)
            commands.append(final_cmd)
        elif context['has_audio'] and not context['has_video_layers']:
            # Audio-only output
            audio_cmd = await self._build_audio_only_command(context)
            commands.append(audio_cmd)
        
        return commands
    
    async def _build_video_command(self, context: Dict[str, Any]) -> str:
        """Build FFmpeg command for video processing"""
        cmd_parts = ['ffmpeg', '-y']  # -y to overwrite output files
        
        # Hardware acceleration
        if self.config.hardware_accel:
            cmd_parts.extend(['-hwaccel', self.config.hardware_accel])
        
        # Input sources
        input_mapping = {}
        input_index = 0
        
        # Add video layer inputs
        for layer in context['layers']:
            if layer.get('layer_type') == 'video' and layer.get('source_path'):
                if os.path.exists(layer['source_path']):
                    cmd_parts.extend(['-i', layer['source_path']])
                    input_mapping[layer.get('layer_id', f'layer_{input_index}')] = input_index
                    input_index += 1
                else:
                    # Generate solid color or pattern for missing video
                    color = layer.get('color', '#000000')
                    duration = layer.get('duration', 5.0)
                    resolution = self.config.resolution or (1920, 1080)
                    
                    cmd_parts.extend([
                        '-f', 'lavfi',
                        '-i', f'color=color={color}:size={resolution[0]}x{resolution[1]}:duration={duration}:rate={self.config.frame_rate}'
                    ])
                    input_mapping[layer.get('layer_id', f'layer_{input_index}')] = input_index
                    input_index += 1
        
        # Build filter graph for video processing
        filter_complex = []
        video_layers = [l for l in context['layers'] if l.get('layer_type') == 'video']
        
        # Check if we have any valid inputs
        valid_inputs = len([idx for idx in input_mapping.values() if idx is not None])
        
        if valid_inputs == 0:
            # No valid inputs, create a simple color source
            self.logger.warning("No valid inputs found, creating default color source")
            cmd_parts.extend([
                '-f', 'lavfi',
                '-i', f'color=color=#000000:size=1920x1080:duration={context.get("total_duration", 5.0)}:rate={self.config.frame_rate}'
            ])
            # Update input mapping for the color source we just added
            input_mapping['default_color'] = input_index
            input_index += 1
            valid_inputs = 1
        
        if len(video_layers) == 1 and not context['effects_timeline'] and not video_layers[0].get('source_path'):
            # Simple color source - no filtergraph needed, but ensure we have a valid input
            if valid_inputs > 0:
                cmd_parts.extend(['-map', '0:v'])
            else:
                self.logger.error("No valid inputs for simple mapping")
                return None
        else:
            # Apply effects from timeline
            valid_layer_count = 0
            for i, layer in enumerate(video_layers):
                layer_id = layer.get('layer_id', f'layer_{i}')
                input_idx = input_mapping.get(layer_id)
                
                # Skip layers with invalid input mapping
                if input_idx is None:
                    self.logger.warning(f"Skipping layer {layer_id} - no valid input mapping")
                    continue
                
                # Apply layer-specific effects
                layer_filters = []
                
                # Apply effects from timeline that affect this layer
                for effect in context['effects_timeline']:
                    if effect.get('enabled', True):
                        effect_filter = self._build_effect_filter(effect, f'[{input_idx}:v]')
                        if effect_filter:
                            layer_filters.append(effect_filter)
                
                # Chain filters for this layer
                if layer_filters:
                    layer_chain = f'[{input_idx}:v]' + ''.join(layer_filters) + f'[v{valid_layer_count}]'
                    filter_complex.append(layer_chain)
                else:
                    # No effects, just copy
                    filter_complex.append(f'[{input_idx}:v]copy[v{valid_layer_count}]')
                
                valid_layer_count += 1
            
            # Combine all video layers
            if valid_layer_count > 1:
                # Multiple layers - need compositing
                overlay_chain = f'[v0]'
                for i in range(1, valid_layer_count):
                    overlay_chain += f'[v{i}]overlay'
                    if i < valid_layer_count - 1:
                        overlay_chain += f'[tmp{i}];[tmp{i}]'
                overlay_chain += '[vout]'
                filter_complex.append(overlay_chain)
            elif valid_layer_count == 1:
                # Single layer
                filter_complex.append('[v0]copy[vout]')
            else:
                # No valid layers, fallback to first available input
                self.logger.warning("No valid layers processed, falling back to first available input")
                if valid_inputs > 0:
                    # Find first valid input index
                    first_input_idx = None
                    for layer_id, idx in input_mapping.items():
                        if idx is not None:
                            first_input_idx = idx
                            break
                    if first_input_idx is not None:
                        filter_complex = [f'[{first_input_idx}:v]copy[vout]']
                    else:
                        self.logger.error("No valid inputs found in mapping")
                        return None
                else:
                    self.logger.error("Emergency fallback: no valid inputs available")
                    return None
            
            # Add filter complex if we have filters
            if filter_complex:
                cmd_parts.extend(['-filter_complex', ';'.join(filter_complex)])
                cmd_parts.extend(['-map', '[vout]'])
            else:
                # Fallback to simple mapping
                if valid_inputs > 0:
                    # Find first valid input index
                    first_input_idx = None
                    for layer_id, idx in input_mapping.items():
                        if idx is not None:
                            first_input_idx = idx
                            break
                    if first_input_idx is not None:
                        cmd_parts.extend(['-map', f'{first_input_idx}:v'])
                    else:
                        self.logger.error("No valid inputs found in mapping")
                        return None
                else:
                    self.logger.error("Emergency fallback: no inputs available")
                    return None
        
        # Video encoding settings
        cmd_parts.extend(['-c:v', self.config.video_codec])
        
        if self.config.crf is not None:
            cmd_parts.extend(['-crf', str(self.config.crf)])
        
        if self.config.preset:
            cmd_parts.extend(['-preset', self.config.preset])
        
        if self.config.tune:
            cmd_parts.extend(['-tune', self.config.tune])
        
        if self.config.pixel_format:
            cmd_parts.extend(['-pix_fmt', self.config.pixel_format])
        
        if self.config.frame_rate:
            cmd_parts.extend(['-r', str(self.config.frame_rate)])
        
        if self.config.resolution:
            cmd_parts.extend(['-s', f'{self.config.resolution[0]}x{self.config.resolution[1]}'])
        
        if self.config.video_bitrate:
            cmd_parts.extend(['-b:v', self.config.video_bitrate])
        
        # Duration
        if context['total_duration'] > 0:
            cmd_parts.extend(['-t', str(context['total_duration'])])
        
        # Output (temporary video file if we need to add audio later)
        if context['has_audio']:
            temp_video = context['temp_dir'] / 'video_temp.mp4'
            cmd_parts.append(str(temp_video))
        else:
            cmd_parts.append(context['output_path'])
        
        return ' '.join(f'"{part}"' if ' ' in str(part) else str(part) for part in cmd_parts)
    
    async def _build_final_composition_command(self, context: Dict[str, Any]) -> str:
        """Build final composition command combining video and audio"""
        cmd_parts = ['ffmpeg', '-y']
        
        # Input video (from previous step)
        temp_video = context['temp_dir'] / 'video_temp.mp4'
        cmd_parts.extend(['-i', str(temp_video)])
        
        # Input audio
        if context['audio_path'] and os.path.exists(context['audio_path']):
            cmd_parts.extend(['-i', context['audio_path']])
        
        # Map video and audio
        cmd_parts.extend(['-map', '0:v', '-map', '1:a'])
        
        # Video settings (copy since already processed)
        cmd_parts.extend(['-c:v', 'copy'])
        
        # Audio settings
        cmd_parts.extend(['-c:a', self.config.audio_codec])
        cmd_parts.extend(['-b:a', self.config.audio_bitrate])
        
        # Add subtitles if available
        subtitle_filters = []
        for subtitle_info in context['subtitle_files'].values():
            if subtitle_info.get('format') == 'srt':
                # Create subtitle file
                subtitle_path = context['temp_dir'] / subtitle_info['filename']
                subtitle_path.write_text(subtitle_info['content'], encoding='utf-8')
                subtitle_filters.append(f"subtitles='{subtitle_path}'")
        
        if subtitle_filters:
            cmd_parts.extend(['-vf', ','.join(subtitle_filters)])
        
        # Metadata
        for key, value in self.config.metadata.items():
            cmd_parts.extend(['-metadata', f'{key}={value}'])
        
        # Output format settings
        if self.config.movflags:
            cmd_parts.extend(['-movflags', self.config.movflags])
        
        # Final output
        cmd_parts.append(context['output_path'])
        
        return ' '.join(f'"{part}"' if ' ' in str(part) else str(part) for part in cmd_parts)
    
    async def _build_audio_only_command(self, context: Dict[str, Any]) -> str:
        """Build command for audio-only output"""
        cmd_parts = ['ffmpeg', '-y']
        
        cmd_parts.extend(['-i', context['audio_path']])
        cmd_parts.extend(['-c:a', self.config.audio_codec])
        cmd_parts.extend(['-b:a', self.config.audio_bitrate])
        
        # Duration
        if context['total_duration'] > 0:
            cmd_parts.extend(['-t', str(context['total_duration'])])
        
        cmd_parts.append(context['output_path'])
        
        return ' '.join(f'"{part}"' if ' ' in str(part) else str(part) for part in cmd_parts)
    
    def _build_effect_filter(self, effect: Dict[str, Any], input_label: str) -> Optional[str]:
        """Build FFmpeg filter for a specific effect"""
        effect_type = effect.get('type', '')
        parameters = effect.get('parameters', {})
        
        if effect_type == 'color_grading':
            # Color adjustment filter
            brightness = parameters.get('exposure', 0) * 0.1  # Convert to brightness
            contrast = parameters.get('contrast', 1.0)
            saturation = parameters.get('saturation', 1.0)
            
            return f"eq=brightness={brightness}:contrast={contrast}:saturation={saturation}"
        
        elif effect_type == 'blur':
            radius = parameters.get('radius', 5)
            return f"boxblur={radius}"
        
        elif effect_type == 'vignette':
            angle = parameters.get('angle', 'PI/5')
            x0 = parameters.get('x0', 'w/2')
            y0 = parameters.get('y0', 'h/2')
            return f"vignette=angle={angle}:x0={x0}:y0={y0}"
        
        elif effect_type == 'scale':
            width = parameters.get('width', -1)
            height = parameters.get('height', -1)
            return f"scale={width}:{height}"
        
        # Add more effect types as needed
        
        return None
    
    async def _execute_render(
        self, 
        commands: List[str], 
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> Dict[str, Any]:
        """Execute FFmpeg rendering commands"""
        import time
        
        start_time = time.time()
        total_commands = len(commands)
        
        statistics = {
            'commands_executed': 0,
            'total_commands': total_commands,
            'errors': []
        }
        
        for i, command in enumerate(commands):
            try:
                self.logger.info(f"Executing command {i+1}/{total_commands}")
                self.logger.debug(f"Command: {command}")
                
                # Execute command with progress tracking
                process = await asyncio.create_subprocess_shell(
                    command,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode != 0:
                    error_msg = stderr.decode('utf-8') if stderr else 'Unknown error'
                    statistics['errors'].append(error_msg)
                    raise RuntimeError(f"FFmpeg command failed: {error_msg}")
                
                statistics['commands_executed'] += 1
                
                # Update progress
                if progress_callback:
                    progress = (i + 1) / total_commands * 100
                    progress_callback(progress)
                
            except Exception as e:
                statistics['errors'].append(str(e))
                raise
        
        end_time = time.time()
        render_time = end_time - start_time
        
        return {
            'render_time': render_time,
            'file_size': 0,  # Will be updated after validation
            'statistics': statistics
        }
    
    async def _validate_output(self, output_path: str) -> Dict[str, Any]:
        """Validate rendered output file"""
        if not os.path.exists(output_path):
            return {'valid': False, 'error': 'Output file not created'}
        
        # Check file size
        file_size = os.path.getsize(output_path)
        if file_size == 0:
            return {'valid': False, 'error': 'Output file is empty'}
        
        # Basic media validation using ffprobe
        try:
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', output_path
            ]
            
            result = await asyncio.create_subprocess_exec(
                *probe_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode != 0:
                return {'valid': False, 'error': 'ffprobe validation failed'}
            
            probe_data = json.loads(stdout.decode('utf-8'))
            
            return {
                'valid': True,
                'file_size': file_size,
                'format': probe_data.get('format', {}),
                'streams': probe_data.get('streams', [])
            }
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {e}'}
    
    async def _extract_media_info(self, file_path: str) -> Dict[str, Any]:
        """Extract comprehensive media information"""
        try:
            probe_cmd = [
                'ffprobe', '-v', 'quiet', '-print_format', 'json',
                '-show_format', '-show_streams', file_path
            ]
            
            result = await asyncio.create_subprocess_exec(
                *probe_cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await result.communicate()
            
            if result.returncode == 0:
                return json.loads(stdout.decode('utf-8'))
            else:
                return {'error': 'Failed to extract media info'}
                
        except Exception as e:
            return {'error': str(e)}