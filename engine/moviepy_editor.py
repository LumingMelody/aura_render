#!/usr/bin/env python3
"""
MoviePy Video Editor Implementation

A concrete implementation of AbstractVideoEditor using MoviePy library.
This provides basic video editing capabilities with Python-friendly API.
"""

import sys
from pathlib import Path
import logging
from typing import Dict, Any, List, Optional, Union
from datetime import datetime

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from moviepy import (
        VideoFileClip, AudioFileClip, ImageClip, TextClip, CompositeVideoClip,
        concatenate_videoclips, vfx, afx
    )
    MOVIEPY_AVAILABLE = True
except ImportError:
    MOVIEPY_AVAILABLE = False
    print("⚠️  MoviePy not available. Please install: pip install moviepy")

from engine.abstract_video_editor import AbstractVideoEditor
from config import settings

logger = logging.getLogger(__name__)


class Source:
    """Source media representation"""
    def __init__(self, path: str, media_type: str = "video"):
        self.path = path
        self.media_type = media_type  # "video", "audio", "image"


class Effect:
    """Effect representation"""
    def __init__(self, effect_type: str, **params):
        self.effect_type = effect_type
        self.params = params


class Transition:
    """Transition representation"""
    def __init__(self, transition_type: str, duration: float = 1.0):
        self.transition_type = transition_type
        self.duration = duration


class TrackType:
    """Track type enumeration"""
    VIDEO = "video"
    AUDIO = "audio" 
    TEXT = "text"


class MoviePyEditor(AbstractVideoEditor):
    """MoviePy implementation of video editor"""
    
    def __init__(self):
        if not MOVIEPY_AVAILABLE:
            raise ImportError("MoviePy is required for video editing")
        
        self.video_clips = []
        self.audio_clips = []
        self.text_clips = []
        
        self.tracks = {
            TrackType.VIDEO: [],
            TrackType.AUDIO: [],
            TrackType.TEXT: []
        }
        
        logger.info("🎬 MoviePy editor initialized")
    
    def load_source(self, source: Source) -> Any:
        """Load source media file"""
        try:
            if source.media_type == "video":
                clip = VideoFileClip(source.path)
                logger.info(f"📹 Loaded video: {source.path} ({clip.duration:.2f}s)")
                return clip
            elif source.media_type == "audio":
                clip = AudioFileClip(source.path)
                logger.info(f"🎵 Loaded audio: {source.path} ({clip.duration:.2f}s)")
                return clip
            elif source.media_type == "image":
                clip = ImageClip(source.path)
                logger.info(f"🖼️  Loaded image: {source.path}")
                return clip
            else:
                raise ValueError(f"Unsupported media type: {source.media_type}")
                
        except Exception as e:
            logger.error(f"❌ Failed to load {source.path}: {e}")
            raise
    
    def create_clip(self, media_ref: Any, in_point: float, out_point: float,
                   speed: float = 1.0, reverse: bool = False) -> Any:
        """Create a clip with specified parameters"""
        try:
            # Apply time constraints
            clip = media_ref.subclip(in_point, out_point)
            
            # Apply speed change
            if speed != 1.0:
                if hasattr(clip, 'fx'):
                    clip = clip.fx(vfx.speedx, speed)
                else:
                    # For audio clips
                    clip = clip.speedx(speed)
            
            # Apply reverse
            if reverse:
                if hasattr(clip, 'fx'):
                    clip = clip.fx(vfx.time_mirror)
                else:
                    # For audio clips
                    clip = clip.time_mirror()
            
            logger.info(f"✂️  Created clip: {in_point:.2f}s-{out_point:.2f}s, speed={speed}x")
            return clip
            
        except Exception as e:
            logger.error(f"❌ Failed to create clip: {e}")
            raise
    
    def apply_effect(self, clip_ref: Any, effect: Effect) -> Any:
        """Apply an effect to a clip"""
        try:
            if effect.effect_type == "fade_in":
                duration = effect.params.get("duration", 1.0)
                clip_ref = clip_ref.fadein(duration)
                
            elif effect.effect_type == "fade_out":
                duration = effect.params.get("duration", 1.0)
                clip_ref = clip_ref.fadeout(duration)
                
            elif effect.effect_type == "resize":
                width = effect.params.get("width")
                height = effect.params.get("height")
                if width and height:
                    clip_ref = clip_ref.resize((width, height))
                elif width or height:
                    clip_ref = clip_ref.resize(width=width, height=height)
                    
            elif effect.effect_type == "crop":
                x1 = effect.params.get("x1", 0)
                y1 = effect.params.get("y1", 0)
                x2 = effect.params.get("x2")
                y2 = effect.params.get("y2")
                if x2 and y2:
                    clip_ref = clip_ref.crop(x1=x1, y1=y1, x2=x2, y2=y2)
                    
            elif effect.effect_type == "brightness":
                factor = effect.params.get("factor", 1.0)
                clip_ref = clip_ref.fx(vfx.multiply_color, factor)
                
            elif effect.effect_type == "text_overlay":
                text = effect.params.get("text", "")
                font = effect.params.get("font", "Arial")
                font_size = effect.params.get("font_size", 50)
                color = effect.params.get("color", "white")
                position = effect.params.get("position", ("center", "center"))
                
                text_clip = TextClip(text, font=font, font_size=font_size, color=color)
                text_clip = text_clip.set_position(position).set_duration(clip_ref.duration)
                clip_ref = CompositeVideoClip([clip_ref, text_clip])
                
            else:
                logger.warning(f"⚠️  Unknown effect type: {effect.effect_type}")
                
            logger.info(f"🎨 Applied effect: {effect.effect_type}")
            return clip_ref
            
        except Exception as e:
            logger.error(f"❌ Failed to apply effect {effect.effect_type}: {e}")
            raise
    
    def add_transition(self, clip1_ref: Any, clip2_ref: Any, transition: Transition) -> Any:
        """Add transition between clips"""
        try:
            if transition.transition_type == "crossfade":
                # Create crossfade transition
                clip1_ref = clip1_ref.fadeout(transition.duration)
                clip2_ref = clip2_ref.fadein(transition.duration)
                
                # Overlap the clips
                clip2_ref = clip2_ref.set_start(clip1_ref.duration - transition.duration)
                result = CompositeVideoClip([clip1_ref, clip2_ref])
                
            elif transition.transition_type == "cut":
                # Simple concatenation
                result = concatenate_videoclips([clip1_ref, clip2_ref])
                
            else:
                logger.warning(f"⚠️  Unknown transition type: {transition.transition_type}")
                result = concatenate_videoclips([clip1_ref, clip2_ref])
            
            logger.info(f"🔄 Added transition: {transition.transition_type}")
            return result
            
        except Exception as e:
            logger.error(f"❌ Failed to add transition: {e}")
            raise
    
    def add_to_track(self, track_type: str, clip_ref: Any, timeline_start: float) -> None:
        """Add clip to specified track"""
        try:
            # Set the start time
            clip_ref = clip_ref.set_start(timeline_start)
            
            # Add to appropriate track
            if track_type == TrackType.VIDEO:
                self.tracks[TrackType.VIDEO].append(clip_ref)
            elif track_type == TrackType.AUDIO:
                self.tracks[TrackType.AUDIO].append(clip_ref)
            elif track_type == TrackType.TEXT:
                self.tracks[TrackType.TEXT].append(clip_ref)
            else:
                raise ValueError(f"Unknown track type: {track_type}")
                
            logger.info(f"➕ Added clip to {track_type} track at {timeline_start:.2f}s")
            
        except Exception as e:
            logger.error(f"❌ Failed to add clip to track: {e}")
            raise
    
    def set_track_volume(self, track_type: str, volume: float) -> None:
        """Set track volume"""
        try:
            if track_type == TrackType.AUDIO:
                for clip in self.tracks[TrackType.AUDIO]:
                    if hasattr(clip, 'volumex'):
                        clip.volumex(volume)
            elif track_type == TrackType.VIDEO:
                # Set opacity for video clips
                for clip in self.tracks[TrackType.VIDEO]:
                    if hasattr(clip, 'set_opacity'):
                        clip.set_opacity(volume)
                        
            logger.info(f"🔊 Set {track_type} volume to {volume}")
            
        except Exception as e:
            logger.error(f"❌ Failed to set track volume: {e}")
            raise
    
    def render(self, output_path: str, fps: float = 24, resolution: Dict[str, int] = None) -> None:
        """Render final video"""
        try:
            logger.info(f"🎬 Starting render to {output_path}")
            
            # Combine all video clips
            video_clips = self.tracks[TrackType.VIDEO] + self.tracks[TrackType.TEXT]
            
            if not video_clips:
                raise ValueError("No video clips to render")
            
            # Create composite video
            if len(video_clips) == 1:
                final_video = video_clips[0]
            else:
                final_video = CompositeVideoClip(video_clips)
            
            # Add audio if available
            audio_clips = self.tracks[TrackType.AUDIO]
            if audio_clips:
                if len(audio_clips) == 1:
                    final_audio = audio_clips[0]
                else:
                    # Mix audio clips
                    final_audio = CompositeVideoClip(audio_clips).audio
                final_video = final_video.with_audio(final_audio)
            
            # Apply resolution if specified
            if resolution:
                final_video = final_video.resize((resolution["width"], resolution["height"]))
            
            # Ensure output directory exists
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Render video
            final_video.write_videofile(
                str(output_path),
                fps=fps,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile='temp-audio.m4a',
                remove_temp=True,
                
                logger=None  # Suppress moviepy logs
            )
            
            logger.info(f"✅ Render completed: {output_path}")
            
        except Exception as e:
            logger.error(f"❌ Render failed: {e}")
            raise
        finally:
            # Clean up clips to free memory
            self._cleanup_clips()
    
    def _cleanup_clips(self):
        """Clean up clips to free memory"""
        try:
            for track in self.tracks.values():
                for clip in track:
                    if hasattr(clip, 'close'):
                        clip.close()
            logger.info("🧹 Cleaned up clips")
        except Exception as e:
            logger.warning(f"⚠️  Cleanup warning: {e}")


# Factory function for easy creation
def create_video_editor() -> MoviePyEditor:
    """Create a video editor instance"""
    return MoviePyEditor()


if __name__ == "__main__":
    # Simple test
    if MOVIEPY_AVAILABLE:
        print("🎬 MoviePy Editor Test")
        editor = create_video_editor()
        print("✅ Editor created successfully")
    else:
        print("❌ MoviePy not available")