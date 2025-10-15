"""
Task Management API Endpoints

FastAPI endpoints for Celery task management:
- Task submission with priorities
- Status monitoring
- Queue management
- Worker control
"""

from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import APIRouter, HTTPException, status, Depends, Query
from pydantic import BaseModel, Field

from task_queue.task_manager import get_task_manager, TaskPriority, TaskStatus
from task_queue.workers import get_worker_manager
from config import Settings, get_settings
import logging

logger = logging.getLogger(__name__)

# Create router
router = APIRouter(prefix="/tasks", tags=["Task Management"])

# =============================
# Pydantic Models
# =============================

class AsyncVideoGenerationRequest(BaseModel):
    """Async video generation request"""
    theme_id: str = Field(..., description="视频主题", min_length=1, max_length=200)
    keywords_id: List[str] = Field(..., description="关键词列表", min_items=1, max_items=10)
    target_duration_id: int = Field(..., description="目标时长（秒）", ge=5, le=3600)
    user_description_id: str = Field(..., description="用户描述", min_length=1, max_length=1000)
    priority: str = Field("normal", description="任务优先级: low, normal, high, urgent")
    config: Optional[Dict[str, Any]] = Field(None, description="生成配置参数")
    
    class Config:
        json_schema_extra = {
            "example": {
                "theme_id": "产品宣传",
                "keywords_id": ["科技", "创新", "未来"],
                "target_duration_id": 60,
                "user_description_id": "一个展示AI技术发展的60秒宣传视频",
                "priority": "high",
                "config": {
                    "quality": "high",
                    "format": "mp4",
                    "resolution": "1920x1080"
                }
            }
        }

class AsyncTaskResponse(BaseModel):
    """Async task submission response"""
    task_id: str
    status: str
    priority: str
    estimated_duration: Optional[int]
    message: str
    timestamp: datetime

class TaskStatusResponse(BaseModel):
    """Task status response"""
    task_id: str
    status: str
    priority: str
    progress: float
    message: str
    created_at: datetime
    updated_at: datetime
    estimated_duration: Optional[int]
    actual_duration: Optional[int]
    result: Optional[Dict[str, Any]]
    error: Optional[str]

class QueueStatusResponse(BaseModel):
    """Queue status response"""
    timestamp: datetime
    total_active_tasks: int
    task_counts_by_priority: Dict[str, int]
    worker_count: int
    queue_health: str

class WorkerControlRequest(BaseModel):
    """Worker control request"""
    worker_id: str
    action: str = Field(..., description="操作: start, stop, restart")
    queue: Optional[str] = Field("default", description="队列名称")
    concurrency: Optional[int] = Field(4, description="并发数")

# =============================
# Task Management Endpoints
# =============================

@router.post("/video/async", response_model=AsyncTaskResponse)
async def submit_async_video_generation(
    request: AsyncVideoGenerationRequest,
    settings: Settings = Depends(get_settings)
):
    """提交异步视频生成任务"""
    
    try:
        # Convert priority string to enum
        try:
            priority = TaskPriority[request.priority.upper()]
        except KeyError:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid priority: {request.priority}. Must be one of: low, normal, high, urgent"
            )
        
        # Prepare user input
        user_input = {
            "theme_id": request.theme_id,
            "keywords_id": request.keywords_id,
            "target_duration_id": request.target_duration_id,
            "user_description_id": request.user_description_id
        }
        
        # Submit task
        task_manager = get_task_manager(settings)
        task_id = await task_manager.submit_video_generation_task(
            user_input=user_input,
            priority=priority,
            config=request.config
        )
        
        # Get task info for response
        task_info = await task_manager.get_task_status(task_id)
        
        return AsyncTaskResponse(
            task_id=task_id,
            status=task_info.status.value if task_info else "submitted",
            priority=priority.name.lower(),
            estimated_duration=task_info.estimated_duration if task_info else None,
            message=f"视频生成任务已提交，优先级: {priority.name}",
            timestamp=datetime.utcnow()
        )
        
    except Exception as e:
        logger.error(f"Failed to submit async video generation task: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"任务提交失败: {str(e)}"
        )

# =============================
# ⚠️ 已删除的接口
# =============================
# 以下接口已被删除，统一使用外部 API 服务进行状态回调：
# - /tasks/status/{task_id} - 获取任务状态（已删除）
# - /tasks/cancel/{task_id} - 取消任务（已删除）
# - /tasks/history - 获取任务历史（已删除）
#
# 现在所有任务状态更新通过 APIService 的 update_task_status 进行回调

# =============================
# Queue Management Endpoints
# =============================

@router.get("/queue/status", response_model=QueueStatusResponse)
async def get_queue_status(
    settings: Settings = Depends(get_settings)
):
    """获取任务队列状态"""
    
    try:
        task_manager = get_task_manager(settings)
        queue_status = await task_manager.get_queue_status()
        
        # Determine queue health
        total_tasks = queue_status.get('total_active_tasks', 0)
        worker_count = queue_status.get('worker_count', 0)
        
        if worker_count == 0:
            health = "no_workers"
        elif total_tasks > worker_count * 10:  # More than 10 tasks per worker
            health = "overloaded"
        elif total_tasks > worker_count * 5:   # More than 5 tasks per worker
            health = "busy"
        else:
            health = "healthy"
        
        return QueueStatusResponse(
            timestamp=datetime.utcnow(),
            total_active_tasks=total_tasks,
            task_counts_by_priority=queue_status.get('task_counts_by_priority', {}),
            worker_count=worker_count,
            queue_health=health
        )
        
    except Exception as e:
        logger.error(f"Failed to get queue status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取队列状态失败: {str(e)}"
        )

# =============================
# Worker Management Endpoints
# =============================

@router.post("/workers/control")
async def control_worker(
    request: WorkerControlRequest,
    settings: Settings = Depends(get_settings)
):
    """控制 Worker (启动/停止/重启)"""
    
    try:
        worker_manager = get_worker_manager(settings)
        
        if request.action == "start":
            success = worker_manager.start_worker(
                worker_id=request.worker_id,
                queue=request.queue,
                concurrency=request.concurrency
            )
            message = f"Worker {request.worker_id} {'started' if success else 'failed to start'}"
            
        elif request.action == "stop":
            success = worker_manager.stop_worker(request.worker_id)
            message = f"Worker {request.worker_id} {'stopped' if success else 'failed to stop'}"
            
        elif request.action == "restart":
            success = worker_manager.restart_worker(request.worker_id)
            message = f"Worker {request.worker_id} {'restarted' if success else 'failed to restart'}"
            
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid action: {request.action}. Must be one of: start, stop, restart"
            )
        
        if not success:
            raise HTTPException(
                status_code=500,
                detail=f"Worker operation failed: {message}"
            )
        
        return {
            "message": message,
            "worker_id": request.worker_id,
            "action": request.action,
            "timestamp": datetime.utcnow()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to control worker: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Worker控制失败: {str(e)}"
        )

@router.get("/workers/status")
async def get_workers_status(
    worker_id: Optional[str] = Query(None, description="特定Worker ID"),
    settings: Settings = Depends(get_settings)
):
    """获取 Worker 状态"""
    
    try:
        worker_manager = get_worker_manager(settings)
        status = worker_manager.get_worker_status(worker_id)
        
        return {
            "workers": status,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to get worker status: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"获取Worker状态失败: {str(e)}"
        )

@router.post("/workers/autoscale")
async def autoscale_workers(
    target_workers: int = Query(..., description="目标Worker数量", ge=0, le=10),
    queue: str = Query("default", description="队列名称"),
    settings: Settings = Depends(get_settings)
):
    """自动扩缩容 Workers"""
    
    try:
        worker_manager = get_worker_manager(settings)
        result = worker_manager.auto_scale(
            target_workers=target_workers,
            queue=queue
        )
        
        return {
            "message": f"Auto-scaling completed for queue {queue}",
            "target_workers": target_workers,
            "actions": result,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to auto-scale workers: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"自动扩缩容失败: {str(e)}"
        )

# =============================
# System Maintenance Endpoints
# =============================

@router.post("/maintenance/cleanup")
async def cleanup_old_tasks(
    hours: int = Query(24, description="清理多少小时前的任务", ge=1, le=168),
    settings: Settings = Depends(get_settings)
):
    """清理旧的已完成任务"""
    
    try:
        task_manager = get_task_manager(settings)
        cleaned_count = await task_manager.cleanup_completed_tasks(older_than_hours=hours)
        
        return {
            "message": f"清理了 {cleaned_count} 个旧任务",
            "cleaned_count": cleaned_count,
            "hours_threshold": hours,
            "timestamp": datetime.utcnow()
        }
        
    except Exception as e:
        logger.error(f"Failed to cleanup old tasks: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"清理任务失败: {str(e)}"
        )