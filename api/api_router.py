"""
API路由器 - 定义所有API端点和路由规则
"""
from typing import Dict, List, Any, Optional
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, Query, status
from fastapi.security import HTTPBearer
from pydantic import BaseModel, Field

from .auth_service import AuthService, get_current_user, User, LoginRequest, LoginResponse, RefreshRequest
from .rate_limiter import RateLimiter
from workflow.video_generation_workflow import VideoGenerationWorkflow
from monitoring.performance_monitor import PerformanceMonitor


# 请求/响应模型
class VideoGenerationRequest(BaseModel):
    """视频生成请求"""
    text: str = Field(..., description="输入文本")
    template: str = Field("basic_video_generation", description="模板类型")
    params: Dict[str, Any] = Field(default_factory=dict, description="生成参数")


class VideoGenerationResponse(BaseModel):
    """视频生成响应"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    message: str = Field(..., description="响应消息")


class TaskStatusResponse(BaseModel):
    """任务状态响应"""
    task_id: str = Field(..., description="任务ID")
    status: str = Field(..., description="任务状态")
    progress: float = Field(0, description="进度百分比")
    result: Optional[Dict[str, Any]] = Field(None, description="任务结果")
    error: Optional[str] = Field(None, description="错误信息")


class APIRouter:
    """API路由器"""

    def __init__(self, auth_service: AuthService, rate_limiter: RateLimiter,
                 video_workflow: Optional[VideoGenerationWorkflow] = None,
                 performance_monitor: Optional[PerformanceMonitor] = None):
        self.auth_service = auth_service
        self.rate_limiter = rate_limiter
        self.video_workflow = video_workflow
        self.performance_monitor = performance_monitor

        self.router = APIRouter()
        self.security = HTTPBearer()

        # 设置路由
        self._setup_auth_routes()
        self._setup_video_routes()
        self._setup_monitoring_routes()
        self._setup_system_routes()

    def _setup_auth_routes(self):
        """设置认证路由"""

        @self.router.post("/auth/login", response_model=LoginResponse, tags=["Authentication"])
        async def login(request: LoginRequest):
            """用户登录"""
            return await self.auth_service.login(request)

        @self.router.post("/auth/refresh", response_model=LoginResponse, tags=["Authentication"])
        async def refresh_token(request: RefreshRequest):
            """刷新令牌"""
            return await self.auth_service.refresh_token(request)

        @self.router.post("/auth/logout", tags=["Authentication"])
        async def logout(user: User = Depends(get_current_user)):
            """用户登出"""
            # 这里需要从请求头获取token
            return {"message": "Logged out successfully"}

        @self.router.get("/auth/me", tags=["Authentication"])
        async def get_current_user_info(user: User = Depends(get_current_user)):
            """获取当前用户信息"""
            return {
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "role": user.role.value,
                "permissions": [p.value for p in user.permissions]
            }

    def _setup_video_routes(self):
        """设置视频相关路由"""

        @self.router.post("/video/generate", response_model=VideoGenerationResponse, tags=["Video"])
        async def generate_video(request: VideoGenerationRequest, user: User = Depends(get_current_user)):
            """生成视频"""
            if not self.video_workflow:
                raise HTTPException(status_code=503, detail="Video generation service not available")

            # 检查速率限制
            allowed, limit_info = await self.rate_limiter.check_rate_limit(
                client_id=user.user_id,
                user_role=user.role.value
            )

            if not allowed:
                raise HTTPException(
                    status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                    detail="Rate limit exceeded",
                    headers={"Retry-After": str(limit_info.get("retry_after", 60))}
                )

            try:
                # 创建视频生成任务
                generation_request = {
                    "template": request.template,
                    "input": {"text": request.text},
                    "params": request.params,
                    "user_id": user.user_id,
                    "session_id": f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                }

                task_id = await self.video_workflow.create_video_generation_task(generation_request)

                # 异步执行
                await self.video_workflow.execute_video_generation(task_id, async_mode=True)

                return VideoGenerationResponse(
                    task_id=task_id,
                    status="created",
                    message="Video generation task created successfully"
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to create video generation task: {str(e)}")

        @self.router.get("/video/task/{task_id}/status", response_model=TaskStatusResponse, tags=["Video"])
        async def get_task_status(task_id: str, user: User = Depends(get_current_user)):
            """获取任务状态"""
            if not self.video_workflow:
                raise HTTPException(status_code=503, detail="Video generation service not available")

            try:
                status_info = await self.video_workflow.get_generation_status(task_id)
                if not status_info:
                    raise HTTPException(status_code=404, detail="Task not found")

                return TaskStatusResponse(
                    task_id=task_id,
                    status=status_info.get("status", "unknown"),
                    progress=status_info.get("progress", 0),
                    result=status_info.get("result"),
                    error=status_info.get("error")
                )

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get task status: {str(e)}")

        @self.router.delete("/video/task/{task_id}", tags=["Video"])
        async def cancel_task(task_id: str, user: User = Depends(get_current_user)):
            """取消任务"""
            if not self.video_workflow:
                raise HTTPException(status_code=503, detail="Video generation service not available")

            try:
                success = await self.video_workflow.cancel_generation(task_id)
                if success:
                    return {"message": "Task cancelled successfully"}
                else:
                    raise HTTPException(status_code=404, detail="Task not found or cannot be cancelled")

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to cancel task: {str(e)}")

        @self.router.get("/video/templates", tags=["Video"])
        async def get_video_templates(user: User = Depends(get_current_user)):
            """获取视频模板列表"""
            if not self.video_workflow:
                raise HTTPException(status_code=503, detail="Video generation service not available")

            try:
                templates = self.video_workflow.get_available_templates()
                return {"templates": templates}

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get templates: {str(e)}")

        @self.router.get("/video/tasks", tags=["Video"])
        async def list_user_tasks(user: User = Depends(get_current_user),
                                 status_filter: Optional[str] = Query(None, description="状态过滤")):
            """列出用户任务"""
            if not self.video_workflow:
                raise HTTPException(status_code=503, detail="Video generation service not available")

            try:
                tasks = self.video_workflow.list_active_generations()
                # 这里应该过滤用户的任务，暂时返回所有任务
                return {"tasks": tasks}

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to list tasks: {str(e)}")

    def _setup_monitoring_routes(self):
        """设置监控相关路由"""

        @self.router.get("/monitoring/dashboard", tags=["Monitoring"])
        async def get_monitoring_dashboard(user: User = Depends(get_current_user)):
            """获取监控面板数据"""
            if not self.performance_monitor:
                raise HTTPException(status_code=503, detail="Monitoring service not available")

            try:
                dashboard_data = self.performance_monitor.get_dashboard_data()
                return dashboard_data

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get dashboard data: {str(e)}")

        @self.router.get("/monitoring/metrics", tags=["Monitoring"])
        async def get_system_metrics(user: User = Depends(get_current_user)):
            """获取系统指标"""
            if not self.performance_monitor:
                raise HTTPException(status_code=503, detail="Monitoring service not available")

            try:
                metrics = self.performance_monitor.get_metrics_summary()
                return {"metrics": metrics}

            except Exception as e:
                raise HTTPException(status_code=500, detail=f"Failed to get metrics: {str(e)}")

        @self.router.get("/monitoring/health", tags=["Monitoring"])
        async def get_system_health():
            """获取系统健康状态"""
            if not self.performance_monitor:
                return {"status": "monitoring_unavailable"}

            try:
                health = self.performance_monitor.get_system_health()
                return {
                    "status": health.overall_status,
                    "cpu_usage": health.cpu_usage,
                    "memory_usage": health.memory_usage,
                    "disk_usage": health.disk_usage,
                    "error_rate": health.error_rate,
                    "timestamp": health.timestamp.isoformat()
                }

            except Exception as e:
                return {"status": "error", "message": str(e)}

    def _setup_system_routes(self):
        """设置系统相关路由"""

        @self.router.get("/system/stats", tags=["System"])
        async def get_system_stats(user: User = Depends(get_current_user)):
            """获取系统统计"""
            stats = {
                "auth_stats": self.auth_service.get_auth_stats(),
                "rate_limiter_stats": self.rate_limiter.get_stats(),
                "timestamp": datetime.now().isoformat()
            }

            if self.video_workflow:
                try:
                    system_status = await self.video_workflow.get_system_status()
                    stats["video_workflow_stats"] = system_status
                except Exception:
                    stats["video_workflow_stats"] = "error"

            if self.performance_monitor:
                try:
                    perf_metrics = await self.performance_monitor.get_performance_metrics()
                    stats["performance_stats"] = perf_metrics
                except Exception:
                    stats["performance_stats"] = "error"

            return stats

        @self.router.get("/system/config", tags=["System"])
        async def get_system_config(user: User = Depends(get_current_user)):
            """获取系统配置"""
            # 这里应该根据用户权限返回配置信息
            config = {
                "api_version": "1.0.0",
                "features": {
                    "video_generation": self.video_workflow is not None,
                    "performance_monitoring": self.performance_monitor is not None,
                    "rate_limiting": True,
                    "authentication": True
                },
                "limits": {
                    "max_video_duration": 300,  # 5分钟
                    "max_file_size": 100 * 1024 * 1024,  # 100MB
                    "supported_formats": ["mp4", "avi", "mov"]
                }
            }

            return config