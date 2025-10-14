"""
Quality Validator

Validates rendered video quality and provides quality metrics
and recommendations for improvement.
"""

import os
import json
import asyncio
import subprocess
from typing import Dict, Any, Optional, List, Tuple
from dataclasses import dataclass
from pathlib import Path
import logging


@dataclass
class QualityMetrics:
    """Video quality metrics"""
    # File metrics
    file_size_mb: float
    duration_seconds: float
    bitrate_kbps: float
    
    # Video metrics
    resolution: Tuple[int, int]
    frame_rate: float
    video_codec: str
    pixel_format: str
    
    # Audio metrics
    audio_codec: Optional[str] = None
    audio_bitrate_kbps: Optional[float] = None
    audio_sample_rate: Optional[int] = None
    audio_channels: Optional[int] = None
    
    # Quality scores (0-10)
    overall_quality: float = 0.0
    video_quality: float = 0.0
    audio_quality: float = 0.0
    
    # Analysis results
    has_issues: bool = False
    issues: List[str] = None
    recommendations: List[str] = None
    
    def __post_init__(self):
        if self.issues is None:
            self.issues = []
        if self.recommendations is None:
            self.recommendations = []


class QualityValidator:
    """Video quality validation and analysis"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Quality thresholds
        self.min_bitrate_kbps = 1000  # Minimum acceptable bitrate
        self.max_file_size_mb = 500   # Maximum reasonable file size
        self.min_duration_sec = 0.5   # Minimum duration
        self.max_duration_sec = 3600  # Maximum duration (1 hour)
        self.target_resolutions = [(1920, 1080), (1280, 720), (3840, 2160)]
        
    async def validate_video(self, file_path: str) -> QualityMetrics:
        """
        Comprehensive video quality validation
        
        Args:
            file_path: Path to video file
            
        Returns:
            Quality metrics with scores and recommendations
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Video file not found: {file_path}")
            
            # Extract technical metadata
            metadata = await self._extract_metadata(file_path)
            
            # Create quality metrics
            metrics = self._create_quality_metrics(file_path, metadata)
            
            # Analyze quality
            await self._analyze_video_quality(metrics, metadata)
            await self._analyze_audio_quality(metrics, metadata)
            
            # Calculate overall quality score
            self._calculate_overall_quality(metrics)
            
            # Generate recommendations
            self._generate_recommendations(metrics)
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Quality validation failed: {e}")
            
            # Return minimal metrics with error
            file_size = os.path.getsize(file_path) / (1024 * 1024) if os.path.exists(file_path) else 0
            return QualityMetrics(
                file_size_mb=file_size,
                duration_seconds=0,
                bitrate_kbps=0,
                resolution=(0, 0),
                frame_rate=0,
                video_codec="unknown",
                pixel_format="unknown",
                has_issues=True,
                issues=[f"Validation error: {str(e)}"]
            )
    
    async def _extract_metadata(self, file_path: str) -> Dict[str, Any]:
        """Extract video metadata using ffprobe"""
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                file_path
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode('utf-8') if stderr else 'Unknown error'
                raise RuntimeError(f"ffprobe failed: {error_msg}")
            
            return json.loads(stdout.decode('utf-8'))
            
        except Exception as e:
            self.logger.error(f"Metadata extraction failed: {e}")
            return {}
    
    def _create_quality_metrics(self, file_path: str, metadata: Dict[str, Any]) -> QualityMetrics:
        """Create quality metrics from metadata"""
        # File info
        file_size_bytes = os.path.getsize(file_path)
        file_size_mb = file_size_bytes / (1024 * 1024)
        
        # Format info
        format_info = metadata.get('format', {})
        duration_str = format_info.get('duration', '0')
        duration_seconds = float(duration_str)
        
        # Calculate overall bitrate
        bitrate_str = format_info.get('bit_rate', '0')
        bitrate_kbps = float(bitrate_str) / 1000 if bitrate_str != 'N/A' else 0
        
        # Find video stream
        video_stream = None
        audio_stream = None
        
        for stream in metadata.get('streams', []):
            if stream.get('codec_type') == 'video':
                video_stream = stream
            elif stream.get('codec_type') == 'audio':
                audio_stream = stream
        
        # Video info
        if video_stream:
            width = int(video_stream.get('width', 0))
            height = int(video_stream.get('height', 0))
            resolution = (width, height)
            
            # Frame rate
            r_frame_rate = video_stream.get('r_frame_rate', '0/1')
            if '/' in r_frame_rate:
                num, den = r_frame_rate.split('/')
                frame_rate = float(num) / float(den) if float(den) != 0 else 0
            else:
                frame_rate = float(r_frame_rate)
            
            video_codec = video_stream.get('codec_name', 'unknown')
            pixel_format = video_stream.get('pix_fmt', 'unknown')
        else:
            resolution = (0, 0)
            frame_rate = 0
            video_codec = 'none'
            pixel_format = 'none'
        
        # Audio info
        audio_codec = None
        audio_bitrate_kbps = None
        audio_sample_rate = None
        audio_channels = None
        
        if audio_stream:
            audio_codec = audio_stream.get('codec_name', 'unknown')
            audio_bitrate_str = audio_stream.get('bit_rate', '0')
            audio_bitrate_kbps = float(audio_bitrate_str) / 1000 if audio_bitrate_str != 'N/A' else 0
            audio_sample_rate = int(audio_stream.get('sample_rate', 0))
            audio_channels = int(audio_stream.get('channels', 0))
        
        return QualityMetrics(
            file_size_mb=file_size_mb,
            duration_seconds=duration_seconds,
            bitrate_kbps=bitrate_kbps,
            resolution=resolution,
            frame_rate=frame_rate,
            video_codec=video_codec,
            pixel_format=pixel_format,
            audio_codec=audio_codec,
            audio_bitrate_kbps=audio_bitrate_kbps,
            audio_sample_rate=audio_sample_rate,
            audio_channels=audio_channels
        )
    
    async def _analyze_video_quality(self, metrics: QualityMetrics, metadata: Dict[str, Any]):
        """Analyze video quality and assign score"""
        score = 10.0  # Start with perfect score
        
        # Check resolution
        width, height = metrics.resolution
        if width == 0 or height == 0:
            metrics.issues.append("Video has no video stream or invalid resolution")
            score = 0.0
            return
        
        # Resolution quality scoring
        total_pixels = width * height
        if total_pixels >= 8294400:  # 4K (3840x2160)
            resolution_score = 10.0
        elif total_pixels >= 2073600:  # 1080p (1920x1080)
            resolution_score = 9.0
        elif total_pixels >= 921600:   # 720p (1280x720)
            resolution_score = 7.5
        elif total_pixels >= 409920:   # 480p (854x480)
            resolution_score = 6.0
        else:
            resolution_score = 4.0
            metrics.issues.append("Video resolution is below 480p")
        
        # Frame rate scoring
        if metrics.frame_rate >= 60:
            framerate_score = 10.0
        elif metrics.frame_rate >= 30:
            framerate_score = 9.0
        elif metrics.frame_rate >= 24:
            framerate_score = 8.0
        else:
            framerate_score = 5.0
            metrics.issues.append(f"Low frame rate: {metrics.frame_rate} fps")
        
        # Bitrate scoring (relative to resolution)
        expected_bitrate = self._calculate_expected_bitrate(width, height, metrics.frame_rate)
        bitrate_ratio = metrics.bitrate_kbps / expected_bitrate if expected_bitrate > 0 else 0
        
        if bitrate_ratio >= 1.0:
            bitrate_score = 10.0
        elif bitrate_ratio >= 0.8:
            bitrate_score = 8.5
        elif bitrate_ratio >= 0.6:
            bitrate_score = 7.0
        elif bitrate_ratio >= 0.4:
            bitrate_score = 5.5
            metrics.issues.append("Video bitrate is lower than recommended")
        else:
            bitrate_score = 3.0
            metrics.issues.append("Video bitrate is significantly low")
        
        # Codec scoring
        codec_scores = {
            'h264': 8.5,
            'h265': 10.0,
            'hevc': 10.0,
            'vp9': 9.0,
            'av1': 10.0,
            'mpeg4': 6.0,
            'mpeg2': 5.0
        }
        codec_score = codec_scores.get(metrics.video_codec.lower(), 7.0)
        
        # Pixel format scoring
        if metrics.pixel_format in ['yuv420p', 'yuv420p10le']:
            pixel_format_score = 10.0
        elif 'yuv420' in metrics.pixel_format:
            pixel_format_score = 9.0
        else:
            pixel_format_score = 7.0
        
        # Calculate weighted video quality score
        metrics.video_quality = (
            resolution_score * 0.3 +
            framerate_score * 0.2 +
            bitrate_score * 0.3 +
            codec_score * 0.15 +
            pixel_format_score * 0.05
        )
    
    async def _analyze_audio_quality(self, metrics: QualityMetrics, metadata: Dict[str, Any]):
        """Analyze audio quality and assign score"""
        if not metrics.audio_codec:
            metrics.audio_quality = 0.0
            return
        
        # Codec scoring
        codec_scores = {
            'aac': 9.0,
            'mp3': 7.5,
            'flac': 10.0,
            'opus': 9.5,
            'vorbis': 8.0,
            'ac3': 7.0
        }
        codec_score = codec_scores.get(metrics.audio_codec.lower(), 6.0)
        
        # Bitrate scoring
        if metrics.audio_bitrate_kbps:
            if metrics.audio_bitrate_kbps >= 320:
                bitrate_score = 10.0
            elif metrics.audio_bitrate_kbps >= 256:
                bitrate_score = 9.0
            elif metrics.audio_bitrate_kbps >= 192:
                bitrate_score = 8.0
            elif metrics.audio_bitrate_kbps >= 128:
                bitrate_score = 7.0
            elif metrics.audio_bitrate_kbps >= 96:
                bitrate_score = 6.0
                metrics.issues.append("Audio bitrate is below recommended levels")
            else:
                bitrate_score = 4.0
                metrics.issues.append("Audio bitrate is very low")
        else:
            bitrate_score = 5.0
        
        # Sample rate scoring
        if metrics.audio_sample_rate:
            if metrics.audio_sample_rate >= 48000:
                sample_rate_score = 10.0
            elif metrics.audio_sample_rate >= 44100:
                sample_rate_score = 9.0
            elif metrics.audio_sample_rate >= 22050:
                sample_rate_score = 7.0
            else:
                sample_rate_score = 5.0
                metrics.issues.append("Audio sample rate is low")
        else:
            sample_rate_score = 5.0
        
        # Channels scoring
        if metrics.audio_channels:
            if metrics.audio_channels >= 2:
                channels_score = 10.0
            else:
                channels_score = 8.0
        else:
            channels_score = 5.0
        
        # Calculate weighted audio quality score
        metrics.audio_quality = (
            codec_score * 0.3 +
            bitrate_score * 0.4 +
            sample_rate_score * 0.2 +
            channels_score * 0.1
        )
    
    def _calculate_expected_bitrate(self, width: int, height: int, fps: float) -> float:
        """Calculate expected bitrate for given resolution and framerate"""
        # Base bitrate per pixel at 30fps (in kbps)
        base_bitrate_per_pixel = 0.1
        
        # Calculate total pixels
        total_pixels = width * height
        
        # Base bitrate for 30fps
        base_bitrate = total_pixels * base_bitrate_per_pixel
        
        # Adjust for frame rate
        fps_multiplier = fps / 30.0 if fps > 0 else 1.0
        
        return base_bitrate * fps_multiplier
    
    def _calculate_overall_quality(self, metrics: QualityMetrics):
        """Calculate overall quality score"""
        if metrics.video_quality == 0:
            metrics.overall_quality = 0.0
            return
        
        # Weight video more heavily than audio
        if metrics.audio_quality > 0:
            metrics.overall_quality = (metrics.video_quality * 0.7 + metrics.audio_quality * 0.3)
        else:
            # Video only
            metrics.overall_quality = metrics.video_quality * 0.9
        
        # Apply penalties for major issues
        if len(metrics.issues) > 0:
            penalty = min(len(metrics.issues) * 0.5, 3.0)
            metrics.overall_quality = max(0, metrics.overall_quality - penalty)
        
        # Check if there are significant issues
        metrics.has_issues = len(metrics.issues) > 0 or metrics.overall_quality < 7.0
    
    def _generate_recommendations(self, metrics: QualityMetrics):
        """Generate quality improvement recommendations"""
        recommendations = []
        
        # Video recommendations
        if metrics.video_quality < 8.0:
            width, height = metrics.resolution
            
            if width < 1280 or height < 720:
                recommendations.append("Consider increasing video resolution to at least 720p")
            
            if metrics.frame_rate < 30:
                recommendations.append("Consider increasing frame rate to 30fps for smoother playback")
            
            if metrics.bitrate_kbps < self.min_bitrate_kbps:
                expected = self._calculate_expected_bitrate(width, height, metrics.frame_rate)
                recommendations.append(f"Increase video bitrate to at least {expected:.0f} kbps for better quality")
            
            if metrics.video_codec not in ['h264', 'h265', 'hevc']:
                recommendations.append("Consider using H.264 or H.265 codec for better compression")
        
        # Audio recommendations
        if metrics.audio_quality > 0 and metrics.audio_quality < 8.0:
            if metrics.audio_bitrate_kbps and metrics.audio_bitrate_kbps < 128:
                recommendations.append("Increase audio bitrate to at least 128 kbps")
            
            if metrics.audio_sample_rate and metrics.audio_sample_rate < 44100:
                recommendations.append("Use audio sample rate of at least 44.1 kHz")
            
            if metrics.audio_codec not in ['aac', 'opus', 'flac']:
                recommendations.append("Consider using AAC or Opus codec for better audio quality")
        
        # File size recommendations
        if metrics.file_size_mb > self.max_file_size_mb:
            recommendations.append("File size is large - consider reducing bitrate or resolution")
        elif metrics.file_size_mb < 1.0 and metrics.duration_seconds > 10:
            recommendations.append("File size seems small - quality might be compromised")
        
        metrics.recommendations = recommendations
    
    def format_quality_report(self, metrics: QualityMetrics) -> str:
        """Format quality metrics into a human-readable report"""
        report = []
        report.append("=== VIDEO QUALITY REPORT ===")
        report.append("")
        
        # Overall score
        report.append(f"Overall Quality Score: {metrics.overall_quality:.1f}/10")
        report.append(f"Video Quality: {metrics.video_quality:.1f}/10")
        report.append(f"Audio Quality: {metrics.audio_quality:.1f}/10")
        report.append("")
        
        # Technical details
        report.append("Technical Details:")
        report.append(f"  File Size: {metrics.file_size_mb:.2f} MB")
        report.append(f"  Duration: {metrics.duration_seconds:.2f} seconds")
        report.append(f"  Overall Bitrate: {metrics.bitrate_kbps:.0f} kbps")
        report.append("")
        
        report.append("Video:")
        report.append(f"  Resolution: {metrics.resolution[0]}x{metrics.resolution[1]}")
        report.append(f"  Frame Rate: {metrics.frame_rate:.2f} fps")
        report.append(f"  Codec: {metrics.video_codec}")
        report.append(f"  Pixel Format: {metrics.pixel_format}")
        report.append("")
        
        if metrics.audio_codec:
            report.append("Audio:")
            report.append(f"  Codec: {metrics.audio_codec}")
            report.append(f"  Bitrate: {metrics.audio_bitrate_kbps:.0f} kbps")
            report.append(f"  Sample Rate: {metrics.audio_sample_rate} Hz")
            report.append(f"  Channels: {metrics.audio_channels}")
            report.append("")
        
        # Issues
        if metrics.issues:
            report.append("Issues Found:")
            for issue in metrics.issues:
                report.append(f"  • {issue}")
            report.append("")
        
        # Recommendations
        if metrics.recommendations:
            report.append("Recommendations:")
            for rec in metrics.recommendations:
                report.append(f"  • {rec}")
        
        return "\n".join(report)