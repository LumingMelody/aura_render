"""
Export and Cloud Storage API Endpoints

REST API endpoints for video export and cloud storage operations.
"""

from fastapi import APIRouter, HTTPException, Depends, status
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

from export import (
    get_export_manager,
    ExportFormat,
    VideoQuality,
    CloudProvider,
    ExportStatus,
    ExportSettings,
    CloudStorageConfig,
    ExportJob
)

router = APIRouter(prefix="/api/export", tags=["export"])


class ExportSettingsRequest(BaseModel):
    """Request model for export settings"""
    format: ExportFormat
    quality: VideoQuality = VideoQuality.HIGH
    resolution: Optional[str] = None
    bitrate: Optional[int] = None
    fps: Optional[int] = None
    codec: Optional[str] = None
    audio_codec: Optional[str] = None
    container_options: Dict[str, Any] = Field(default_factory=dict)


class CloudStorageConfigRequest(BaseModel):
    """Request model for cloud storage configuration"""
    provider: CloudProvider
    bucket_name: str
    access_key: Optional[str] = None
    secret_key: Optional[str] = None
    region: Optional[str] = None
    endpoint_url: Optional[str] = None
    base_path: str = ""
    make_public: bool = False
    expiry_days: Optional[int] = None


class CreateExportJobRequest(BaseModel):
    """Request model for creating export jobs"""
    user_id: str
    source_path: str
    settings: ExportSettingsRequest
    cloud_config: Optional[CloudStorageConfigRequest] = None
    project_id: Optional[str] = None
    video_id: Optional[str] = None


class ExportJobResponse(BaseModel):
    """Response model for export jobs"""
    id: str
    user_id: str
    status: str
    created_at: datetime
    started_at: Optional[datetime]
    completed_at: Optional[datetime]
    progress_percentage: float
    local_path: Optional[str]
    cloud_url: Optional[str]
    download_url: Optional[str]
    file_size: Optional[int]
    error_message: Optional[str]


@router.post("/", response_model=ExportJobResponse)
async def create_export_job(request: CreateExportJobRequest):
    """Create a new export job"""
    try:
        export_manager = get_export_manager()
        
        # Convert request to export settings
        settings = ExportSettings(
            format=request.settings.format,
            quality=request.settings.quality,
            resolution=request.settings.resolution,
            bitrate=request.settings.bitrate,
            fps=request.settings.fps,
            codec=request.settings.codec,
            audio_codec=request.settings.audio_codec,
            container_options=request.settings.container_options
        )
        
        # Convert cloud config if provided
        cloud_config = None
        if request.cloud_config:
            cloud_config = CloudStorageConfig(
                provider=request.cloud_config.provider,
                bucket_name=request.cloud_config.bucket_name,
                access_key=request.cloud_config.access_key,
                secret_key=request.cloud_config.secret_key,
                region=request.cloud_config.region,
                endpoint_url=request.cloud_config.endpoint_url,
                base_path=request.cloud_config.base_path,
                make_public=request.cloud_config.make_public,
                expiry_days=request.cloud_config.expiry_days
            )
        
        # Create the export job
        job = await export_manager.create_export_job(
            user_id=request.user_id,
            settings=settings,
            source_path=request.source_path,
            cloud_config=cloud_config,
            project_id=request.project_id,
            video_id=request.video_id
        )
        
        # Auto-start the job
        await export_manager.start_export(job.id)
        
        return ExportJobResponse(
            id=job.id,
            user_id=job.user_id,
            status=job.status.value,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            progress_percentage=job.progress_percentage,
            local_path=job.local_path,
            cloud_url=job.cloud_url,
            download_url=job.download_url,
            file_size=job.file_size,
            error_message=job.error_message
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create export job: {str(e)}"
        )


@router.get("/{job_id}", response_model=ExportJobResponse)
async def get_export_job(job_id: str):
    """Get export job details"""
    try:
        export_manager = get_export_manager()
        
        job = await export_manager.get_job(job_id)
        if not job:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Export job not found"
            )
        
        return ExportJobResponse(
            id=job.id,
            user_id=job.user_id,
            status=job.status.value,
            created_at=job.created_at,
            started_at=job.started_at,
            completed_at=job.completed_at,
            progress_percentage=job.progress_percentage,
            local_path=job.local_path,
            cloud_url=job.cloud_url,
            download_url=job.download_url,
            file_size=job.file_size,
            error_message=job.error_message
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get export job: {str(e)}"
        )


@router.post("/{job_id}/cancel")
async def cancel_export_job(job_id: str):
    """Cancel an export job"""
    try:
        export_manager = get_export_manager()
        
        success = await export_manager.cancel_job(job_id)
        
        if not success:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Job cannot be cancelled (not found or invalid status)"
            )
        
        return {
            "success": True,
            "message": f"Export job {job_id} cancelled successfully",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to cancel export job: {str(e)}"
        )


@router.get("/user/{user_id}", response_model=List[ExportJobResponse])
async def get_user_export_jobs(user_id: str):
    """Get all export jobs for a user"""
    try:
        export_manager = get_export_manager()
        
        jobs = await export_manager.get_user_jobs(user_id)
        
        return [
            ExportJobResponse(
                id=job.id,
                user_id=job.user_id,
                status=job.status.value,
                created_at=job.created_at,
                started_at=job.started_at,
                completed_at=job.completed_at,
                progress_percentage=job.progress_percentage,
                local_path=job.local_path,
                cloud_url=job.cloud_url,
                download_url=job.download_url,
                file_size=job.file_size,
                error_message=job.error_message
            )
            for job in jobs
        ]
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user export jobs: {str(e)}"
        )


@router.get("/formats/list")
async def get_export_formats():
    """Get all available export formats"""
    return {
        "formats": [
            {
                "value": format.value,
                "label": format.value.upper(),
                "description": f"Export as {format.value.upper()} format"
            }
            for format in ExportFormat
        ]
    }


@router.get("/qualities/list") 
async def get_video_qualities():
    """Get all available video quality presets"""
    return {
        "qualities": [
            {
                "value": quality.value,
                "label": quality.value.replace("_", " ").title(),
                "description": {
                    "low": "480p - Smaller file size",
                    "medium": "720p - Good balance of quality and size",
                    "high": "1080p - High definition",
                    "ultra": "4K - Ultra high definition",
                    "custom": "Custom settings"
                }.get(quality.value, "Custom quality settings")
            }
            for quality in VideoQuality
        ]
    }


@router.get("/providers/list")
async def get_cloud_providers():
    """Get all supported cloud storage providers"""
    return {
        "providers": [
            {
                "value": provider.value,
                "label": provider.value.replace("_", " ").title(),
                "description": {
                    "aws_s3": "Amazon Web Services S3",
                    "google_cloud": "Google Cloud Storage",
                    "azure_blob": "Microsoft Azure Blob Storage",
                    "dropbox": "Dropbox Cloud Storage",
                    "local": "Local File System"
                }.get(provider.value, "Cloud storage provider")
            }
            for provider in CloudProvider
        ]
    }


# Convenience endpoints for common export operations

@router.post("/quick/mp4")
async def quick_mp4_export(
    user_id: str,
    source_path: str,
    quality: VideoQuality = VideoQuality.HIGH,
    project_id: Optional[str] = None
):
    """Quick MP4 export with standard settings"""
    try:
        export_manager = get_export_manager()
        
        settings = ExportSettings(
            format=ExportFormat.MP4,
            quality=quality
        )
        
        job = await export_manager.create_export_job(
            user_id=user_id,
            settings=settings,
            source_path=source_path,
            project_id=project_id
        )
        
        await export_manager.start_export(job.id)
        
        return {
            "success": True,
            "job_id": job.id,
            "message": f"MP4 export started with {quality.value} quality",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start MP4 export: {str(e)}"
        )


@router.post("/quick/gif")
async def quick_gif_export(
    user_id: str,
    source_path: str,
    project_id: Optional[str] = None
):
    """Quick GIF export with optimized settings"""
    try:
        export_manager = get_export_manager()
        
        settings = ExportSettings(
            format=ExportFormat.GIF,
            quality=VideoQuality.MEDIUM  # GIFs work best with medium quality
        )
        
        job = await export_manager.create_export_job(
            user_id=user_id,
            settings=settings,
            source_path=source_path,
            project_id=project_id
        )
        
        await export_manager.start_export(job.id)
        
        return {
            "success": True,
            "job_id": job.id,
            "message": "GIF export started",
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start GIF export: {str(e)}"
        )


@router.post("/quick/audio")
async def quick_audio_export(
    user_id: str,
    source_path: str,
    format: str = "mp3",
    project_id: Optional[str] = None
):
    """Quick audio-only export"""
    try:
        if format not in ["mp3", "wav"]:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio format must be 'mp3' or 'wav'"
            )
        
        export_manager = get_export_manager()
        
        settings = ExportSettings(
            format=ExportFormat.MP3 if format == "mp3" else ExportFormat.WAV,
            quality=VideoQuality.HIGH,
            audio_codec="mp3" if format == "mp3" else "pcm"
        )
        
        job = await export_manager.create_export_job(
            user_id=user_id,
            settings=settings,
            source_path=source_path,
            project_id=project_id
        )
        
        await export_manager.start_export(job.id)
        
        return {
            "success": True,
            "job_id": job.id,
            "message": f"Audio export started ({format.upper()})",
            "timestamp": datetime.now()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start audio export: {str(e)}"
        )