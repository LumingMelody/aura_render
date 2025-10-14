"""
Advanced Export Options and Cloud Storage Integration

Provides comprehensive export capabilities with multiple format support,
cloud storage integration, and advanced delivery options.
"""

import asyncio
import hashlib
import mimetypes
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, BinaryIO
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
import json
import logging

from pydantic import BaseModel, Field
import aiofiles
import boto3
from botocore.exceptions import ClientError


class ExportFormat(str, Enum):
    """Available export formats"""
    MP4 = "mp4"
    MOV = "mov"
    AVI = "avi"
    WEBM = "webm"
    GIF = "gif"
    MP3 = "mp3"
    WAV = "wav"
    JSON = "json"  # For project data


class VideoQuality(str, Enum):
    """Video quality presets"""
    LOW = "low"         # 480p
    MEDIUM = "medium"   # 720p
    HIGH = "high"       # 1080p
    ULTRA = "ultra"     # 4K
    CUSTOM = "custom"


class CloudProvider(str, Enum):
    """Supported cloud storage providers"""
    AWS_S3 = "aws_s3"
    GOOGLE_CLOUD = "google_cloud"
    AZURE_BLOB = "azure_blob"
    DROPBOX = "dropbox"
    LOCAL = "local"


class ExportStatus(str, Enum):
    """Export job status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class ExportSettings:
    """Export configuration settings"""
    format: ExportFormat
    quality: VideoQuality
    resolution: Optional[str] = None  # e.g., "1920x1080"
    bitrate: Optional[int] = None
    fps: Optional[int] = None
    codec: Optional[str] = None
    audio_codec: Optional[str] = None
    container_options: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.container_options is None:
            self.container_options = {}


@dataclass
class CloudStorageConfig:
    """Cloud storage configuration"""
    provider: CloudProvider
    bucket_name: str
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    base_path: str = ""
    make_public: bool = False
    expiry_days: Optional[int] = None


class ExportJob(BaseModel):
    """Export job tracking"""
    id: str
    user_id: str
    project_id: Optional[str] = None
    video_id: Optional[str] = None
    
    # Export configuration
    settings: Dict[str, Any]
    cloud_config: Optional[Dict[str, Any]] = None
    
    # Status tracking
    status: ExportStatus = ExportStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # Progress
    progress_percentage: float = 0.0
    
    # Results
    local_path: Optional[str] = None
    cloud_url: Optional[str] = None
    download_url: Optional[str] = None
    file_size: Optional[int] = None
    
    # Error handling
    error_message: Optional[str] = None
    retry_count: int = 0
    
    # Metadata
    metadata: Dict[str, Any] = Field(default_factory=dict)


class CloudStorageManager:
    """Cloud storage operations manager"""
    
    def __init__(self):
        self.providers = {
            CloudProvider.AWS_S3: self._upload_to_s3,
            CloudProvider.LOCAL: self._upload_to_local,
            # Add other providers as needed
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def upload_file(
        self,
        file_path: Path,
        config: CloudStorageConfig,
        destination_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload file to cloud storage"""
        if config.provider not in self.providers:
            raise ValueError(f"Unsupported provider: {config.provider}")
        
        upload_func = self.providers[config.provider]
        return await upload_func(file_path, config, destination_key)
    
    async def _upload_to_s3(
        self,
        file_path: Path,
        config: CloudStorageConfig,
        destination_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload file to AWS S3"""
        try:
            # Create S3 client
            session = boto3.Session(
                aws_access_key_id=config.access_key,
                aws_secret_access_key=config.secret_key,
                region_name=config.region
            )
            
            s3_client = session.client(
                's3',
                endpoint_url=config.endpoint_url
            )
            
            # Generate destination key
            if not destination_key:
                destination_key = f"{config.base_path}/{file_path.name}"
            elif config.base_path:
                destination_key = f"{config.base_path}/{destination_key}"
            
            # Determine content type
            content_type, _ = mimetypes.guess_type(str(file_path))
            if not content_type:
                content_type = 'application/octet-stream'
            
            # Upload file
            extra_args = {'ContentType': content_type}
            
            if config.make_public:
                extra_args['ACL'] = 'public-read'
            
            # Set expiry if specified
            if config.expiry_days:
                expiry_date = datetime.now() + timedelta(days=config.expiry_days)
                extra_args['Expires'] = expiry_date
            
            await asyncio.to_thread(
                s3_client.upload_file,
                str(file_path),
                config.bucket_name,
                destination_key,
                ExtraArgs=extra_args
            )
            
            # Generate URLs
            cloud_url = f"s3://{config.bucket_name}/{destination_key}"
            
            if config.make_public:
                if config.endpoint_url:
                    download_url = f"{config.endpoint_url}/{config.bucket_name}/{destination_key}"
                else:
                    download_url = f"https://{config.bucket_name}.s3.{config.region}.amazonaws.com/{destination_key}"
            else:
                # Generate presigned URL for private files
                download_url = s3_client.generate_presigned_url(
                    'get_object',
                    Params={'Bucket': config.bucket_name, 'Key': destination_key},
                    ExpiresIn=config.expiry_days * 24 * 3600 if config.expiry_days else 3600
                )
            
            return {
                'cloud_url': cloud_url,
                'download_url': download_url,
                'key': destination_key,
                'bucket': config.bucket_name,
                'success': True
            }
            
        except ClientError as e:
            self.logger.error(f"S3 upload failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    async def _upload_to_local(
        self,
        file_path: Path,
        config: CloudStorageConfig,
        destination_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """Upload file to local storage (for testing/development)"""
        try:
            # Create local storage directory
            local_storage_dir = Path(config.bucket_name)
            local_storage_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate destination path
            if not destination_key:
                destination_key = file_path.name
            
            destination_path = local_storage_dir / config.base_path / destination_key
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Copy file
            async with aiofiles.open(file_path, 'rb') as src:
                async with aiofiles.open(destination_path, 'wb') as dst:
                    while chunk := await src.read(8192):
                        await dst.write(chunk)
            
            return {
                'cloud_url': f"local://{destination_path}",
                'download_url': f"file://{destination_path.absolute()}",
                'path': str(destination_path),
                'success': True
            }
            
        except Exception as e:
            self.logger.error(f"Local upload failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }


class ExportManager:
    """Advanced export and delivery manager"""
    
    def __init__(self, storage_dir: str = "export_jobs"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        self.temp_dir = self.storage_dir / "temp"
        self.temp_dir.mkdir(exist_ok=True)
        
        self.cloud_manager = CloudStorageManager()
        self.active_jobs: Dict[str, ExportJob] = {}
        
        # Export presets
        self.quality_presets = {
            VideoQuality.LOW: {
                'resolution': '854x480',
                'bitrate': 1000,
                'fps': 24
            },
            VideoQuality.MEDIUM: {
                'resolution': '1280x720', 
                'bitrate': 2500,
                'fps': 30
            },
            VideoQuality.HIGH: {
                'resolution': '1920x1080',
                'bitrate': 5000,
                'fps': 30
            },
            VideoQuality.ULTRA: {
                'resolution': '3840x2160',
                'bitrate': 15000,
                'fps': 30
            }
        }
        
        self.logger = logging.getLogger(__name__)
    
    async def create_export_job(
        self,
        user_id: str,
        settings: ExportSettings,
        source_path: str,
        cloud_config: Optional[CloudStorageConfig] = None,
        project_id: Optional[str] = None,
        video_id: Optional[str] = None
    ) -> ExportJob:
        """Create a new export job"""
        job_id = f"export_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{user_id[:8]}"
        
        job = ExportJob(
            id=job_id,
            user_id=user_id,
            project_id=project_id,
            video_id=video_id,
            settings=settings.__dict__,
            cloud_config=cloud_config.__dict__ if cloud_config else None
        )
        
        # Store source path in metadata
        job.metadata['source_path'] = source_path
        
        await self._save_job(job)
        self.active_jobs[job_id] = job
        
        return job
    
    async def start_export(self, job_id: str) -> bool:
        """Start processing an export job"""
        job = self.active_jobs.get(job_id)
        if not job:
            job = await self._load_job(job_id)
            if not job:
                return False
        
        if job.status != ExportStatus.PENDING:
            return False
        
        job.status = ExportStatus.PROCESSING
        job.started_at = datetime.now()
        await self._save_job(job)
        
        # Start processing in background
        asyncio.create_task(self._process_export_job(job))
        
        return True
    
    async def _process_export_job(self, job: ExportJob):
        """Process an export job"""
        try:
            self.logger.info(f"Starting export job {job.id}")
            
            # Get source file
            source_path = Path(job.metadata['source_path'])
            if not source_path.exists():
                raise FileNotFoundError(f"Source file not found: {source_path}")
            
            # Apply export settings
            settings = ExportSettings(**job.settings)
            
            # Generate output filename
            output_filename = f"{source_path.stem}_{job.id}.{settings.format.value}"
            output_path = self.temp_dir / output_filename
            
            # Export the file
            await self._export_file(source_path, output_path, settings, job)
            
            job.local_path = str(output_path)
            job.file_size = output_path.stat().st_size
            job.progress_percentage = 50.0
            await self._save_job(job)
            
            # Upload to cloud if configured
            if job.cloud_config:
                cloud_config = CloudStorageConfig(**job.cloud_config)
                upload_result = await self.cloud_manager.upload_file(
                    output_path,
                    cloud_config,
                    output_filename
                )
                
                if upload_result.get('success'):
                    job.cloud_url = upload_result['cloud_url']
                    job.download_url = upload_result['download_url']
                    job.metadata.update(upload_result)
                else:
                    raise Exception(f"Cloud upload failed: {upload_result.get('error')}")
            
            # Complete the job
            job.status = ExportStatus.COMPLETED
            job.completed_at = datetime.now()
            job.progress_percentage = 100.0
            
            await self._save_job(job)
            
            # Clean up temp file if uploaded to cloud
            if job.cloud_url and output_path.exists():
                output_path.unlink()
            
            self.logger.info(f"Export job {job.id} completed successfully")
            
        except Exception as e:
            self.logger.error(f"Export job {job.id} failed: {e}")
            job.status = ExportStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            await self._save_job(job)
    
    async def _export_file(
        self,
        source_path: Path,
        output_path: Path,
        settings: ExportSettings,
        job: ExportJob
    ):
        """Export file with specified settings"""
        if settings.format == ExportFormat.JSON:
            # Export project/video metadata as JSON
            await self._export_as_json(source_path, output_path, job)
        elif settings.format in [ExportFormat.MP4, ExportFormat.MOV, ExportFormat.AVI, ExportFormat.WEBM]:
            # Export as video
            await self._export_as_video(source_path, output_path, settings, job)
        elif settings.format == ExportFormat.GIF:
            # Export as GIF
            await self._export_as_gif(source_path, output_path, settings, job)
        elif settings.format in [ExportFormat.MP3, ExportFormat.WAV]:
            # Export audio only
            await self._export_as_audio(source_path, output_path, settings, job)
        else:
            raise ValueError(f"Unsupported export format: {settings.format}")
    
    async def _export_as_video(
        self,
        source_path: Path,
        output_path: Path,
        settings: ExportSettings,
        job: ExportJob
    ):
        """Export as video format using FFmpeg"""
        # Build FFmpeg command
        cmd = ['ffmpeg', '-i', str(source_path)]
        
        # Apply quality settings
        if settings.quality != VideoQuality.CUSTOM:
            preset = self.quality_presets[settings.quality]
            cmd.extend(['-s', preset['resolution']])
            cmd.extend(['-b:v', f"{preset['bitrate']}k"])
            cmd.extend(['-r', str(preset['fps'])])
        else:
            if settings.resolution:
                cmd.extend(['-s', settings.resolution])
            if settings.bitrate:
                cmd.extend(['-b:v', f"{settings.bitrate}k"])
            if settings.fps:
                cmd.extend(['-r', str(settings.fps)])
        
        # Set codecs
        if settings.codec:
            cmd.extend(['-c:v', settings.codec])
        if settings.audio_codec:
            cmd.extend(['-c:a', settings.audio_codec])
        
        # Container-specific options
        if settings.container_options:
            for key, value in settings.container_options.items():
                cmd.extend([f'-{key}', str(value)])
        
        # Output file
        cmd.extend(['-y', str(output_path)])  # -y to overwrite
        
        # Execute FFmpeg
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        # Monitor progress (simplified)
        job.progress_percentage = 25.0
        await self._save_job(job)
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"FFmpeg failed: {stderr.decode()}")
    
    async def _export_as_gif(
        self,
        source_path: Path,
        output_path: Path,
        settings: ExportSettings,
        job: ExportJob
    ):
        """Export as animated GIF"""
        # Build FFmpeg command for GIF
        cmd = [
            'ffmpeg', '-i', str(source_path),
            '-vf', 'fps=10,scale=480:-1:flags=lanczos',
            '-c:v', 'gif',
            '-y', str(output_path)
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        job.progress_percentage = 25.0
        await self._save_job(job)
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"GIF export failed: {stderr.decode()}")
    
    async def _export_as_audio(
        self,
        source_path: Path,
        output_path: Path,
        settings: ExportSettings,
        job: ExportJob
    ):
        """Export audio only"""
        cmd = [
            'ffmpeg', '-i', str(source_path),
            '-vn',  # No video
            '-acodec', settings.audio_codec or 'mp3',
            '-y', str(output_path)
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        job.progress_percentage = 25.0
        await self._save_job(job)
        
        stdout, stderr = await process.communicate()
        
        if process.returncode != 0:
            raise Exception(f"Audio export failed: {stderr.decode()}")
    
    async def _export_as_json(
        self,
        source_path: Path,
        output_path: Path,
        job: ExportJob
    ):
        """Export project/video data as JSON"""
        # This would export project configuration, timeline data, etc.
        # For now, create a basic metadata export
        metadata = {
            'export_id': job.id,
            'source_file': str(source_path),
            'export_time': datetime.now().isoformat(),
            'user_id': job.user_id,
            'project_id': job.project_id,
            'video_id': job.video_id,
            'settings': job.settings,
            'file_info': {
                'size': source_path.stat().st_size,
                'modified': datetime.fromtimestamp(source_path.stat().st_mtime).isoformat()
            }
        }
        
        async with aiofiles.open(output_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(metadata, indent=2, ensure_ascii=False))
    
    async def get_job(self, job_id: str) -> Optional[ExportJob]:
        """Get export job by ID"""
        if job_id in self.active_jobs:
            return self.active_jobs[job_id]
        
        return await self._load_job(job_id)
    
    async def get_user_jobs(self, user_id: str) -> List[ExportJob]:
        """Get all export jobs for a user"""
        jobs = []
        
        # Active jobs
        for job in self.active_jobs.values():
            if job.user_id == user_id:
                jobs.append(job)
        
        # Stored jobs
        for job_file in self.storage_dir.glob(f"*{user_id}*.json"):
            try:
                job = await self._load_job_from_file(job_file)
                if job and job.id not in self.active_jobs:
                    jobs.append(job)
            except Exception:
                continue
        
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)
    
    async def cancel_job(self, job_id: str) -> bool:
        """Cancel an export job"""
        job = await self.get_job(job_id)
        if not job or job.status not in [ExportStatus.PENDING, ExportStatus.PROCESSING]:
            return False
        
        job.status = ExportStatus.CANCELLED
        job.completed_at = datetime.now()
        await self._save_job(job)
        
        return True
    
    async def _save_job(self, job: ExportJob):
        """Save export job to disk"""
        job_file = self.storage_dir / f"export_{job.user_id}_{job.id}.json"
        
        async with aiofiles.open(job_file, 'w', encoding='utf-8') as f:
            await f.write(job.json(indent=2, ensure_ascii=False, default=str))
    
    async def _load_job(self, job_id: str) -> Optional[ExportJob]:
        """Load export job from disk"""
        for job_file in self.storage_dir.glob(f"*{job_id}*.json"):
            return await self._load_job_from_file(job_file)
        return None
    
    async def _load_job_from_file(self, job_file: Path) -> Optional[ExportJob]:
        """Load job from specific file"""
        try:
            async with aiofiles.open(job_file, 'r', encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
                return ExportJob(**data)
        except Exception as e:
            self.logger.error(f"Error loading job from {job_file}: {e}")
            return None


# Global instance
_export_manager = None

def get_export_manager() -> ExportManager:
    """Get the global export manager instance"""
    global _export_manager
    if _export_manager is None:
        _export_manager = ExportManager()
    return _export_manager