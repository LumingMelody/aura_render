"""
Backup and Disaster Recovery System

Comprehensive backup and disaster recovery system with automated backups,
point-in-time recovery, and multi-tier backup strategies.
"""

import asyncio
import os
import shutil
import json
import gzip
import tarfile
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, asdict
from enum import Enum
from pathlib import Path
import logging

import boto3
from botocore.exceptions import ClientError
import schedule

logger = logging.getLogger(__name__)


class BackupType(Enum):
    """Backup types"""
    FULL = "full"
    INCREMENTAL = "incremental"
    DIFFERENTIAL = "differential"
    SNAPSHOT = "snapshot"


class BackupStatus(Enum):
    """Backup operation status"""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class StorageLocation(Enum):
    """Backup storage locations"""
    LOCAL = "local"
    S3 = "s3"
    GCS = "gcs"
    AZURE = "azure"
    FTP = "ftp"


@dataclass
class BackupConfig:
    """Backup configuration"""
    name: str
    backup_type: BackupType
    storage_location: StorageLocation
    schedule_cron: Optional[str] = None
    retention_days: int = 30
    compression: bool = True
    encryption: bool = False
    include_patterns: List[str] = None
    exclude_patterns: List[str] = None
    storage_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.include_patterns is None:
            self.include_patterns = []
        if self.exclude_patterns is None:
            self.exclude_patterns = []
        if self.storage_config is None:
            self.storage_config = {}


@dataclass
class BackupJob:
    """Backup job information"""
    id: str
    config: BackupConfig
    status: BackupStatus
    created_at: datetime
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    size_bytes: int = 0
    files_count: int = 0
    error_message: Optional[str] = None
    storage_path: Optional[str] = None
    checksum: Optional[str] = None


@dataclass
class RestorePoint:
    """System restore point"""
    id: str
    backup_id: str
    created_at: datetime
    description: str
    database_backup: str
    files_backup: str
    config_backup: str
    size_bytes: int
    is_verified: bool = False


class BackupStorage:
    """Abstract backup storage interface"""
    
    async def store_backup(self, local_path: str, remote_path: str) -> bool:
        """Store backup file to remote location"""
        raise NotImplementedError
    
    async def retrieve_backup(self, remote_path: str, local_path: str) -> bool:
        """Retrieve backup file from remote location"""
        raise NotImplementedError
    
    async def list_backups(self, prefix: str = "") -> List[Dict[str, Any]]:
        """List available backups"""
        raise NotImplementedError
    
    async def delete_backup(self, remote_path: str) -> bool:
        """Delete backup file from remote location"""
        raise NotImplementedError


class LocalStorage(BackupStorage):
    """Local file system storage"""
    
    def __init__(self, base_path: str):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
    
    async def store_backup(self, local_path: str, remote_path: str) -> bool:
        try:
            dest_path = self.base_path / remote_path
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local_path, dest_path)
            return True
        except Exception as e:
            logger.error(f"Local storage error: {e}")
            return False
    
    async def retrieve_backup(self, remote_path: str, local_path: str) -> bool:
        try:
            src_path = self.base_path / remote_path
            if src_path.exists():
                shutil.copy2(src_path, local_path)
                return True
            return False
        except Exception as e:
            logger.error(f"Local retrieval error: {e}")
            return False
    
    async def list_backups(self, prefix: str = "") -> List[Dict[str, Any]]:
        try:
            backups = []
            search_path = self.base_path / prefix if prefix else self.base_path
            
            for file_path in search_path.rglob("*.backup"):
                stat = file_path.stat()
                backups.append({
                    "name": file_path.name,
                    "path": str(file_path.relative_to(self.base_path)),
                    "size": stat.st_size,
                    "modified": datetime.fromtimestamp(stat.st_mtime)
                })
            
            return sorted(backups, key=lambda x: x["modified"], reverse=True)
            
        except Exception as e:
            logger.error(f"List backups error: {e}")
            return []
    
    async def delete_backup(self, remote_path: str) -> bool:
        try:
            file_path = self.base_path / remote_path
            if file_path.exists():
                file_path.unlink()
                return True
            return False
        except Exception as e:
            logger.error(f"Delete backup error: {e}")
            return False


class S3Storage(BackupStorage):
    """Amazon S3 storage"""
    
    def __init__(self, bucket: str, access_key: str, secret_key: str, region: str = "us-east-1"):
        self.bucket = bucket
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region
        )
    
    async def store_backup(self, local_path: str, remote_path: str) -> bool:
        try:
            self.s3_client.upload_file(local_path, self.bucket, remote_path)
            return True
        except ClientError as e:
            logger.error(f"S3 upload error: {e}")
            return False
    
    async def retrieve_backup(self, remote_path: str, local_path: str) -> bool:
        try:
            self.s3_client.download_file(self.bucket, remote_path, local_path)
            return True
        except ClientError as e:
            logger.error(f"S3 download error: {e}")
            return False
    
    async def list_backups(self, prefix: str = "") -> List[Dict[str, Any]]:
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )
            
            backups = []
            for obj in response.get('Contents', []):
                backups.append({
                    "name": obj['Key'].split('/')[-1],
                    "path": obj['Key'],
                    "size": obj['Size'],
                    "modified": obj['LastModified']
                })
            
            return sorted(backups, key=lambda x: x["modified"], reverse=True)
            
        except ClientError as e:
            logger.error(f"S3 list error: {e}")
            return []
    
    async def delete_backup(self, remote_path: str) -> bool:
        try:
            self.s3_client.delete_object(Bucket=self.bucket, Key=remote_path)
            return True
        except ClientError as e:
            logger.error(f"S3 delete error: {e}")
            return False


class DisasterRecoveryManager:
    """Main disaster recovery and backup manager"""
    
    def __init__(self):
        self.backup_configs: Dict[str, BackupConfig] = {}
        self.storage_providers: Dict[StorageLocation, BackupStorage] = {}
        self.active_jobs: Dict[str, BackupJob] = {}
        self.restore_points: Dict[str, RestorePoint] = {}
        self.scheduler_running = False
        
        # Default backup paths
        self.database_backup_path = "/tmp/db_backup"
        self.files_backup_path = "/tmp/files_backup"
        self.config_backup_path = "/tmp/config_backup"
        
        # Initialize default configurations
        self._initialize_default_configs()
    
    def _initialize_default_configs(self):
        """Initialize default backup configurations"""
        self.backup_configs["daily_full"] = BackupConfig(
            name="daily_full",
            backup_type=BackupType.FULL,
            storage_location=StorageLocation.LOCAL,
            schedule_cron="0 2 * * *",  # Daily at 2 AM
            retention_days=7,
            compression=True,
            include_patterns=["database/*", "uploads/*", "config/*"],
            exclude_patterns=["*.log", "*.tmp", "cache/*"]
        )
        
        self.backup_configs["hourly_incremental"] = BackupConfig(
            name="hourly_incremental",
            backup_type=BackupType.INCREMENTAL,
            storage_location=StorageLocation.LOCAL,
            schedule_cron="0 * * * *",  # Every hour
            retention_days=2,
            compression=True,
            include_patterns=["database/*"],
            exclude_patterns=["*.log"]
        )
        
        self.backup_configs["weekly_s3"] = BackupConfig(
            name="weekly_s3",
            backup_type=BackupType.FULL,
            storage_location=StorageLocation.S3,
            schedule_cron="0 3 * * 0",  # Weekly on Sunday at 3 AM
            retention_days=90,
            compression=True,
            encryption=True,
            include_patterns=["*"]
        )
    
    def add_storage_provider(self, location: StorageLocation, provider: BackupStorage):
        """Add a storage provider"""
        self.storage_providers[location] = provider
    
    def add_backup_config(self, config: BackupConfig):
        """Add a backup configuration"""
        self.backup_configs[config.name] = config
    
    def remove_backup_config(self, name: str):
        """Remove a backup configuration"""
        self.backup_configs.pop(name, None)
    
    async def create_backup(self, config_name: str, description: str = "") -> str:
        """Create a backup job"""
        if config_name not in self.backup_configs:
            raise ValueError(f"Backup configuration '{config_name}' not found")
        
        config = self.backup_configs[config_name]
        job_id = f"backup_{int(datetime.now().timestamp())}_{config_name}"
        
        job = BackupJob(
            id=job_id,
            config=config,
            status=BackupStatus.PENDING,
            created_at=datetime.now()
        )
        
        self.active_jobs[job_id] = job
        
        # Start backup in background
        asyncio.create_task(self._execute_backup(job))
        
        logger.info(f"Backup job {job_id} created for config {config_name}")
        return job_id
    
    async def _execute_backup(self, job: BackupJob):
        """Execute a backup job"""
        try:
            job.status = BackupStatus.IN_PROGRESS
            job.started_at = datetime.now()
            
            # Create temporary backup directory
            temp_dir = Path(f"/tmp/{job.id}")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            backup_files = []
            
            # 1. Backup database
            db_backup_file = await self._backup_database(temp_dir)
            if db_backup_file:
                backup_files.append(db_backup_file)
            
            # 2. Backup files
            files_backup_file = await self._backup_files(temp_dir, job.config)
            if files_backup_file:
                backup_files.append(files_backup_file)
            
            # 3. Backup configuration
            config_backup_file = await self._backup_configuration(temp_dir)
            if config_backup_file:
                backup_files.append(config_backup_file)
            
            # 4. Create consolidated backup archive
            backup_archive = await self._create_backup_archive(
                backup_files, temp_dir, job.config.compression
            )
            
            if backup_archive:
                # 5. Calculate checksum
                job.checksum = await self._calculate_checksum(backup_archive)
                
                # 6. Store backup
                storage = self.storage_providers.get(job.config.storage_location)
                if storage:
                    remote_path = f"{job.config.name}/{job.id}.backup"
                    success = await storage.store_backup(str(backup_archive), remote_path)
                    
                    if success:
                        job.storage_path = remote_path
                        job.size_bytes = backup_archive.stat().st_size
                        job.status = BackupStatus.COMPLETED
                        job.completed_at = datetime.now()
                        
                        # Create restore point
                        await self._create_restore_point(job)
                        
                        logger.info(f"Backup job {job.id} completed successfully")
                    else:
                        job.status = BackupStatus.FAILED
                        job.error_message = "Failed to store backup"
                else:
                    job.status = BackupStatus.FAILED
                    job.error_message = "Storage provider not configured"
            else:
                job.status = BackupStatus.FAILED
                job.error_message = "Failed to create backup archive"
            
            # Cleanup temporary files
            shutil.rmtree(temp_dir, ignore_errors=True)
            
        except Exception as e:
            job.status = BackupStatus.FAILED
            job.error_message = str(e)
            job.completed_at = datetime.now()
            logger.error(f"Backup job {job.id} failed: {e}")
    
    async def _backup_database(self, temp_dir: Path) -> Optional[Path]:
        """Backup database"""
        try:
            backup_file = temp_dir / "database.sql"
            
            # Mock database backup - replace with real implementation
            # For PostgreSQL: pg_dump
            # For MySQL: mysqldump
            # For SQLite: .backup command
            
            with open(backup_file, 'w') as f:
                f.write("-- Mock database backup\n")
                f.write(f"-- Created at: {datetime.now()}\n")
                f.write("-- This is a placeholder for actual database backup\n")
            
            logger.info(f"Database backup created: {backup_file}")
            return backup_file
            
        except Exception as e:
            logger.error(f"Database backup failed: {e}")
            return None
    
    async def _backup_files(self, temp_dir: Path, config: BackupConfig) -> Optional[Path]:
        """Backup files based on include/exclude patterns"""
        try:
            files_archive = temp_dir / "files.tar"
            
            # Mock file backup - replace with real implementation
            with tarfile.open(files_archive, 'w') as tar:
                # Add a sample file
                sample_file = temp_dir / "sample.txt"
                with open(sample_file, 'w') as f:
                    f.write("Sample file for backup testing\n")
                
                tar.add(sample_file, arcname="sample.txt")
            
            logger.info(f"Files backup created: {files_archive}")
            return files_archive
            
        except Exception as e:
            logger.error(f"Files backup failed: {e}")
            return None
    
    async def _backup_configuration(self, temp_dir: Path) -> Optional[Path]:
        """Backup system configuration"""
        try:
            config_file = temp_dir / "config.json"
            
            # Mock configuration backup
            config_data = {
                "backup_time": datetime.now().isoformat(),
                "system_settings": {
                    "version": "1.0.0",
                    "environment": "production",
                    "features": ["ai_optimization", "video_generation"]
                },
                "user_settings": {
                    "default_quality": "standard",
                    "max_duration": 300
                }
            }
            
            with open(config_file, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            logger.info(f"Configuration backup created: {config_file}")
            return config_file
            
        except Exception as e:
            logger.error(f"Configuration backup failed: {e}")
            return None
    
    async def _create_backup_archive(self, files: List[Path], temp_dir: Path, 
                                   compress: bool = True) -> Optional[Path]:
        """Create consolidated backup archive"""
        try:
            archive_name = "backup.tar.gz" if compress else "backup.tar"
            archive_path = temp_dir / archive_name
            
            mode = 'w:gz' if compress else 'w'
            
            with tarfile.open(archive_path, mode) as tar:
                for file_path in files:
                    if file_path.exists():
                        tar.add(file_path, arcname=file_path.name)
            
            logger.info(f"Backup archive created: {archive_path}")
            return archive_path
            
        except Exception as e:
            logger.error(f"Archive creation failed: {e}")
            return None
    
    async def _calculate_checksum(self, file_path: Path) -> str:
        """Calculate file checksum"""
        sha256_hash = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        
        return sha256_hash.hexdigest()
    
    async def _create_restore_point(self, job: BackupJob):
        """Create a restore point from backup job"""
        restore_point = RestorePoint(
            id=f"restore_{job.id}",
            backup_id=job.id,
            created_at=job.completed_at,
            description=f"Automatic restore point from backup {job.id}",
            database_backup=job.storage_path,
            files_backup=job.storage_path,
            config_backup=job.storage_path,
            size_bytes=job.size_bytes
        )
        
        self.restore_points[restore_point.id] = restore_point
        logger.info(f"Restore point created: {restore_point.id}")
    
    async def restore_from_backup(self, backup_id: str, restore_options: Dict[str, bool] = None) -> bool:
        """Restore system from backup"""
        try:
            if restore_options is None:
                restore_options = {
                    "database": True,
                    "files": True,
                    "configuration": True
                }
            
            if backup_id not in self.active_jobs:
                raise ValueError(f"Backup {backup_id} not found")
            
            job = self.active_jobs[backup_id]
            
            if job.status != BackupStatus.COMPLETED:
                raise ValueError(f"Backup {backup_id} is not completed")
            
            # Get storage provider
            storage = self.storage_providers.get(job.config.storage_location)
            if not storage:
                raise ValueError("Storage provider not available")
            
            # Download backup
            temp_dir = Path(f"/tmp/restore_{backup_id}")
            temp_dir.mkdir(parents=True, exist_ok=True)
            
            backup_file = temp_dir / "backup.tar.gz"
            success = await storage.retrieve_backup(job.storage_path, str(backup_file))
            
            if not success:
                raise ValueError("Failed to retrieve backup")
            
            # Verify checksum
            if job.checksum:
                current_checksum = await self._calculate_checksum(backup_file)
                if current_checksum != job.checksum:
                    raise ValueError("Backup checksum verification failed")
            
            # Extract backup
            with tarfile.open(backup_file, 'r:gz') as tar:
                tar.extractall(temp_dir)
            
            # Perform restore operations
            restore_success = True
            
            if restore_options.get("database", False):
                restore_success &= await self._restore_database(temp_dir / "database.sql")
            
            if restore_options.get("files", False):
                restore_success &= await self._restore_files(temp_dir / "files.tar")
            
            if restore_options.get("configuration", False):
                restore_success &= await self._restore_configuration(temp_dir / "config.json")
            
            # Cleanup
            shutil.rmtree(temp_dir, ignore_errors=True)
            
            if restore_success:
                logger.info(f"Restore from backup {backup_id} completed successfully")
            else:
                logger.error(f"Restore from backup {backup_id} completed with errors")
            
            return restore_success
            
        except Exception as e:
            logger.error(f"Restore failed: {e}")
            return False
    
    async def _restore_database(self, backup_file: Path) -> bool:
        """Restore database from backup"""
        try:
            # Mock database restore - replace with real implementation
            if backup_file.exists():
                logger.info(f"Database restored from {backup_file}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Database restore failed: {e}")
            return False
    
    async def _restore_files(self, backup_file: Path) -> bool:
        """Restore files from backup"""
        try:
            # Mock files restore - replace with real implementation
            if backup_file.exists():
                with tarfile.open(backup_file, 'r') as tar:
                    tar.extractall("/tmp/restored_files")
                logger.info(f"Files restored from {backup_file}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Files restore failed: {e}")
            return False
    
    async def _restore_configuration(self, backup_file: Path) -> bool:
        """Restore configuration from backup"""
        try:
            # Mock configuration restore - replace with real implementation
            if backup_file.exists():
                with open(backup_file, 'r') as f:
                    config_data = json.load(f)
                logger.info(f"Configuration restored from {backup_file}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Configuration restore failed: {e}")
            return False
    
    async def list_backups(self, config_name: Optional[str] = None) -> List[Dict[str, Any]]:
        """List available backups"""
        backups = []
        
        for job in self.active_jobs.values():
            if config_name and job.config.name != config_name:
                continue
            
            backup_info = {
                "id": job.id,
                "config_name": job.config.name,
                "type": job.config.backup_type.value,
                "status": job.status.value,
                "created_at": job.created_at,
                "completed_at": job.completed_at,
                "size_bytes": job.size_bytes,
                "storage_location": job.config.storage_location.value,
                "checksum": job.checksum
            }
            backups.append(backup_info)
        
        return sorted(backups, key=lambda x: x["created_at"], reverse=True)
    
    async def delete_backup(self, backup_id: str) -> bool:
        """Delete a backup"""
        try:
            if backup_id not in self.active_jobs:
                return False
            
            job = self.active_jobs[backup_id]
            
            # Delete from storage
            if job.storage_path:
                storage = self.storage_providers.get(job.config.storage_location)
                if storage:
                    await storage.delete_backup(job.storage_path)
            
            # Remove from active jobs
            del self.active_jobs[backup_id]
            
            # Remove related restore points
            restore_points_to_remove = [
                rp_id for rp_id, rp in self.restore_points.items()
                if rp.backup_id == backup_id
            ]
            
            for rp_id in restore_points_to_remove:
                del self.restore_points[rp_id]
            
            logger.info(f"Backup {backup_id} deleted successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete backup {backup_id}: {e}")
            return False
    
    async def cleanup_old_backups(self):
        """Clean up old backups based on retention policies"""
        try:
            for job in list(self.active_jobs.values()):
                if job.status == BackupStatus.COMPLETED and job.completed_at:
                    age_days = (datetime.now() - job.completed_at).days
                    
                    if age_days > job.config.retention_days:
                        logger.info(f"Cleaning up old backup {job.id} (age: {age_days} days)")
                        await self.delete_backup(job.id)
            
        except Exception as e:
            logger.error(f"Backup cleanup failed: {e}")
    
    def get_backup_status(self, backup_id: str) -> Optional[BackupJob]:
        """Get backup job status"""
        return self.active_jobs.get(backup_id)
    
    def get_restore_points(self) -> List[RestorePoint]:
        """Get available restore points"""
        return sorted(self.restore_points.values(), key=lambda x: x.created_at, reverse=True)
    
    async def start_scheduler(self):
        """Start backup scheduler"""
        if self.scheduler_running:
            return
        
        self.scheduler_running = True
        
        # Schedule backup tasks
        for config in self.backup_configs.values():
            if config.schedule_cron:
                # This is a simplified scheduler
                # In production, use a proper cron scheduler like APScheduler
                asyncio.create_task(self._schedule_backup(config))
        
        # Schedule cleanup task
        asyncio.create_task(self._schedule_cleanup())
        
        logger.info("Backup scheduler started")
    
    async def stop_scheduler(self):
        """Stop backup scheduler"""
        self.scheduler_running = False
        logger.info("Backup scheduler stopped")
    
    async def _schedule_backup(self, config: BackupConfig):
        """Schedule backup execution"""
        while self.scheduler_running:
            try:
                # Simple daily backup at 2 AM - replace with proper cron scheduling
                if config.schedule_cron == "0 2 * * *":
                    now = datetime.now()
                    if now.hour == 2 and now.minute < 5:  # Run within first 5 minutes of 2 AM
                        await self.create_backup(config.name, "Scheduled backup")
                        await asyncio.sleep(300)  # Wait 5 minutes to avoid duplicate runs
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Scheduled backup error: {e}")
                await asyncio.sleep(60)
    
    async def _schedule_cleanup(self):
        """Schedule cleanup execution"""
        while self.scheduler_running:
            try:
                # Run cleanup daily at 3 AM
                now = datetime.now()
                if now.hour == 3 and now.minute < 5:
                    await self.cleanup_old_backups()
                    await asyncio.sleep(300)  # Wait 5 minutes
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Scheduled cleanup error: {e}")
                await asyncio.sleep(60)


# Global disaster recovery manager
disaster_recovery = DisasterRecoveryManager()


def get_disaster_recovery() -> DisasterRecoveryManager:
    """Get global disaster recovery manager"""
    return disaster_recovery


async def initialize_disaster_recovery(config: Dict[str, Any]):
    """Initialize disaster recovery system"""
    
    # Add local storage
    local_storage = LocalStorage(config.get("local_backup_path", "/backups"))
    disaster_recovery.add_storage_provider(StorageLocation.LOCAL, local_storage)
    
    # Add S3 storage if configured
    s3_config = config.get("s3")
    if s3_config:
        s3_storage = S3Storage(
            bucket=s3_config["bucket"],
            access_key=s3_config["access_key"],
            secret_key=s3_config["secret_key"],
            region=s3_config.get("region", "us-east-1")
        )
        disaster_recovery.add_storage_provider(StorageLocation.S3, s3_storage)
    
    # Start scheduler
    await disaster_recovery.start_scheduler()
    
    logger.info("Disaster recovery system initialized")