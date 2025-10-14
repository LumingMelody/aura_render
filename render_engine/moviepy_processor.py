"""
MoviePy Processor

High-level video processing using MoviePy for complex compositions
and Python-friendly video editing operations.
"""

import os
import tempfile
from typing import Dict, Any, List, Optional, Tuple, Union
from pathlib import Path
import logging

try:
    from moviepy import (
        VideoFileClip, AudioFileClip, ImageClip, TextClip, ColorClip,
        CompositeVideoClip, CompositeAudioClip, concatenate_videoclips,
        concatenate_audioclips
    )
    from moviepy.video.fx import resize, fadeout, fadein
    from moviepy.audio.fx import volumex
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False


class MoviePyProcessor:
    """MoviePy-based video processor for high-level operations"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        if not MOVIEPY_AVAILABLE:
            self.logger.warning("MoviePy not available - install with: pip install moviepy")
    
    def is_available(self) -> bool:
        """Check if MoviePy is available"""
        return MOVIEPY_AVAILABLE
    
    async def create_composition_video(
        self,
        composition_data: Dict[str, Any],
        output_path: str,
        progress_callback: Optional[callable] = None
    ) -> Dict[str, Any]:
        """
        Create video composition using MoviePy
        
        Args:
            composition_data: Composition specification
            output_path: Output video path
            progress_callback: Progress update callback
            
        Returns:
            Processing result
        """
        if not MOVIEPY_AVAILABLE:
            raise RuntimeError("MoviePy not available")
        
        try:
            self.logger.info("Starting MoviePy composition")
            
            # Extract composition elements
            layers = composition_data.get('layers', [])
            total_duration = composition_data.get('duration', 10.0)
            resolution = composition_data.get('resolution', (1920, 1080))
            fps = composition_data.get('fps', 30)
            
            # Process video layers
            video_clips = await self._create_video_clips(layers, total_duration, resolution)
            
            # Process audio layers
            audio_clips = await self._create_audio_clips(layers, total_duration)
            
            # Create composite video
            if video_clips:
                final_video = CompositeVideoClip(video_clips, size=resolution)
                final_video = final_video.set_duration(total_duration)
                final_video = final_video.set_fps(fps)
            else:
                # Create blank video if no video clips
                final_video = ColorClip(size=resolution, color=(0, 0, 0), duration=total_duration)
                final_video = final_video.set_fps(fps)
            
            # Add audio if available
            if audio_clips:
                final_audio = CompositeAudioClip(audio_clips)
                final_video = final_video.with_audio(final_audio)
            
            # Write video file
            self.logger.info(f"Writing video to: {output_path}")
            
            # Ensure output directory exists
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Write with progress callback
            def progress_wrapper(t):
                if progress_callback:
                    progress = (t / total_duration) * 100
                    progress_callback(progress)
            
            final_video.write_videofile(
                output_path,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=f"{output_path}.temp-audio.m4a",
                remove_temp=True,
                progress_bar=False,  # We have our own progress callback
                
                logger=None
            )
            
            # Cleanup
            final_video.close()
            
            # Get output file info
            file_size = os.path.getsize(output_path)
            
            return {
                'success': True,
                'output_path': output_path,
                'file_size': file_size,
                'duration': total_duration,
                'resolution': resolution,
                'fps': fps
            }
            
        except Exception as e:
            self.logger.error(f"MoviePy composition failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _create_video_clips(
        self,
        layers: List[Dict[str, Any]],
        total_duration: float,
        resolution: Tuple[int, int]
    ) -> List:
        """Create video clips from layer specifications"""
        video_clips = []
        
        for layer in layers:
            if layer.get('layer_type') != 'video':
                continue
            
            try:
                clip = await self._create_single_video_clip(layer, resolution)
                if clip:
                    video_clips.append(clip)
            except Exception as e:
                self.logger.warning(f"Failed to create video clip: {e}")
                continue
        
        # If no video clips, create a default background
        if not video_clips:
            bg_clip = ColorClip(
                size=resolution,
                color=(20, 20, 20),  # Dark gray background
                duration=total_duration
            )
            video_clips.append(bg_clip)
        
        return video_clips
    
    async def _create_single_video_clip(
        self,
        layer: Dict[str, Any],
        resolution: Tuple[int, int]
    ):
        """Create a single video clip from layer spec"""
        source_path = layer.get('source_path')
        start_time = layer.get('start_time', 0)
        duration = layer.get('duration')
        position = layer.get('position', 'center')
        opacity = layer.get('opacity', 1.0)
        
        clip = None
        
        if source_path and os.path.exists(source_path):
            # Load video file
            clip = VideoFileClip(source_path)
        else:
            # Create solid color or pattern
            color = layer.get('color', '#000000')
            color_rgb = self._hex_to_rgb(color)
            clip = ColorClip(size=resolution, color=color_rgb, duration=duration or 5.0)
        
        if clip:
            # Apply timing
            if duration:
                clip = clip.set_duration(duration)
            
            if start_time > 0:
                clip = clip.set_start(start_time)
            
            # Apply position
            if position != 'center':
                clip = clip.set_position(position)
            
            # Apply opacity
            if opacity != 1.0:
                clip = clip.set_opacity(opacity)
            
            # Apply resize if needed
            if hasattr(clip, 'size') and clip.size != resolution:
                clip = clip.resize(resolution)
            
            # Apply effects
            effects = layer.get('effects', [])
            for effect in effects:
                clip = self._apply_video_effect(clip, effect)
        
        return clip
    
    async def _create_audio_clips(
        self,
        layers: List[Dict[str, Any]],
        total_duration: float
    ) -> List:
        """Create audio clips from layer specifications"""
        audio_clips = []
        
        for layer in layers:
            if layer.get('layer_type') != 'audio':
                continue
            
            try:
                clip = await self._create_single_audio_clip(layer)
                if clip:
                    audio_clips.append(clip)
            except Exception as e:
                self.logger.warning(f"Failed to create audio clip: {e}")
                continue
        
        return audio_clips
    
    async def _create_single_audio_clip(self, layer: Dict[str, Any]):
        """Create a single audio clip from layer spec"""
        source_path = layer.get('source_path')
        start_time = layer.get('start_time', 0)
        duration = layer.get('duration')
        volume = layer.get('volume', 1.0)
        
        if not source_path or not os.path.exists(source_path):
            return None
        
        # Load audio file
        clip = AudioFileClip(source_path)
        
        # Apply timing
        if duration:
            clip = clip.set_duration(duration)
        
        if start_time > 0:
            clip = clip.set_start(start_time)
        
        # Apply volume
        if volume != 1.0:
            clip = clip.fx(volumex, volume)
        
        return clip
    
    def _apply_video_effect(self, clip, effect: Dict[str, Any]):
        """Apply video effect to clip"""
        effect_type = effect.get('type', '')
        params = effect.get('parameters', {})
        
        try:
            if effect_type == 'fade_in':
                duration = params.get('duration', 1.0)
                return clip.fx(fadein, duration)
            
            elif effect_type == 'fade_out':
                duration = params.get('duration', 1.0)
                return clip.fx(fadeout, duration)
            
            elif effect_type == 'resize':
                size = params.get('size', (1920, 1080))
                return clip.fx(resize, newsize=size)
            
            # Add more effects as needed
            
        except Exception as e:
            self.logger.warning(f"Failed to apply effect {effect_type}: {e}")
        
        return clip
    
    def create_text_clip(
        self,
        text: str,
        duration: float,
        position: str = 'bottom',
        font_size: int = 50,
        color: str = 'white',
        font: str = 'Arial'
    ):
        """Create a text clip for subtitles or titles"""
        if not MOVIEPY_AVAILABLE:
            return None
        
        try:
            clip = TextClip(
                text,
                font_size=font_size,
                color=color,
                font=font
            ).set_duration(duration).set_position(position)
            
            return clip
            
        except Exception as e:
            self.logger.error(f"Failed to create text clip: {e}")
            return None
    
    def create_image_clip(
        self,
        image_path: str,
        duration: float,
        position: str = 'center',
        resize: Optional[Tuple[int, int]] = None
    ):
        """Create an image clip"""
        if not MOVIEPY_AVAILABLE:
            return None
        
        try:
            clip = ImageClip(image_path).set_duration(duration).set_position(position)
            
            if resize:
                clip = clip.resize(resize)
            
            return clip
            
        except Exception as e:
            self.logger.error(f"Failed to create image clip: {e}")
            return None
    
    def _hex_to_rgb(self, hex_color: str) -> Tuple[int, int, int]:
        """Convert hex color to RGB tuple"""
        hex_color = hex_color.lstrip('#')
        
        if len(hex_color) != 6:
            return (0, 0, 0)  # Default to black
        
        try:
            return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
        except ValueError:
            return (0, 0, 0)
    
    def get_video_info(self, video_path: str) -> Dict[str, Any]:
        """Get video file information using MoviePy"""
        if not MOVIEPY_AVAILABLE or not os.path.exists(video_path):
            return {}
        
        try:
            clip = VideoFileClip(video_path)
            
            info = {
                'duration': clip.duration,
                'size': clip.size,
                'fps': clip.fps,
                'has_audio': clip.audio is not None
            }
            
            clip.close()
            return info
            
        except Exception as e:
            self.logger.error(f"Failed to get video info: {e}")
            return {}
    
    @staticmethod
    def concatenate_videos(video_paths: List[str], output_path: str) -> bool:
        """Concatenate multiple video files"""
        if not MOVIEPY_AVAILABLE:
            return False
        
        try:
            clips = []
            for path in video_paths:
                if os.path.exists(path):
                    clip = VideoFileClip(path)
                    clips.append(clip)
            
            if not clips:
                return False
            
            final_clip = concatenate_videoclips(clips)
            final_clip.write_videofile(output_path)
            
            # Cleanup
            for clip in clips:
                clip.close()
            final_clip.close()
            
            return True
            
        except Exception as e:
            logging.getLogger(__name__).error(f"Video concatenation failed: {e}")
            return False