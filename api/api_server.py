"""
API服务器 - 基于FastAPI的高性能异步API服务
"""
from typing import Dict, List, Any, Optional, Callable
import asyncio
from datetime import datetime
import uvicorn
from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import time
import json

from .auth_service import AuthService, AuthMiddleware
from .rate_limiter import RateLimiter
from .websocket_manager import WebSocketManager
from .api_router import APIRouter
from monitoring.performance_monitor import PerformanceMonitor
from workflow.video_generation_workflow import VideoGenerationWorkflow


class APIConfig(BaseModel):
    """API服务配置"""
    title: str = "Aura Render API"
    description: str = "智能视频生成系统API服务"
    version: str = "1.0.0"
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    enable_cors: bool = True
    cors_origins: List[str] = ["*"]
    enable_compression: bool = True
    enable_docs: bool = True
    docs_url: str = "/docs"
    openapi_url: str = "/openapi.json"
    max_request_size: int = 100 * 1024 * 1024  # 100MB
    request_timeout: float = 300.0  # 5分钟


class HealthResponse(BaseModel):
    """健康检查响应"""
    status: str = Field(..., description="服务状态")
    timestamp: str = Field(..., description="检查时间")
    version: str = Field(..., description="API版本")
    uptime_seconds: float = Field(..., description="运行时间(秒)")
    dependencies: Dict[str, str] = Field(..., description="依赖服务状态")


class ErrorResponse(BaseModel):
    """错误响应"""
    error: str = Field(..., description="错误类型")
    message: str = Field(..., description="错误消息")
    details: Optional[Dict[str, Any]] = Field(None, description="错误详情")
    timestamp: str = Field(..., description="错误时间")
    request_id: Optional[str] = Field(None, description="请求ID")


class APIServer:
    """API服务器"""

    def __init__(self, config: APIConfig,
                 performance_monitor: Optional[PerformanceMonitor] = None,
                 video_workflow: Optional[VideoGenerationWorkflow] = None):
        self.config = config
        self.performance_monitor = performance_monitor
        self.video_workflow = video_workflow

        # 创建FastAPI实例
        self.app = FastAPI(
            title=config.title,
            description=config.description,
            version=config.version,
            docs_url=config.docs_url if config.enable_docs else None,
            openapi_url=config.openapi_url if config.enable_docs else None
        )

        # 核心服务
        self.auth_service = AuthService()
        self.rate_limiter = RateLimiter()
        self.websocket_manager = WebSocketManager()
        self.api_router = APIRouter(
            auth_service=self.auth_service,
            rate_limiter=self.rate_limiter,
            video_workflow=video_workflow,
            performance_monitor=performance_monitor
        )

        # 服务状态
        self.start_time = datetime.now()
        self.request_count = 0

        # 设置中间件和路由
        self._setup_middleware()
        self._setup_routes()
        self._setup_error_handlers()

    def _setup_middleware(self):
        """设置中间件"""
        # CORS中间件
        if self.config.enable_cors:
            self.app.add_middleware(
                CORSMiddleware,
                allow_origins=self.config.cors_origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"]
            )

        # 压缩中间件
        if self.config.enable_compression:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)

        # 认证中间件
        self.app.add_middleware(AuthMiddleware, auth_service=self.auth_service)

        # 请求日志中间件
        @self.app.middleware("http")
        async def request_logging_middleware(request: Request, call_next):
            start_time = time.time()
            request_id = f"req_{int(start_time * 1000)}"

            # 记录请求开始
            print(f"📥 {request.method} {request.url.path} - Request ID: {request_id}")

            # 处理请求
            response = await call_next(request)

            # 计算处理时间
            process_time = (time.time() - start_time) * 1000

            # 记录性能指标
            if self.performance_monitor:
                self.performance_monitor.record_request(
                    endpoint=request.url.path,
                    response_time_ms=process_time,
                    success=response.status_code < 400
                )

            # 更新请求计数
            self.request_count += 1

            # 添加响应头
            response.headers["X-Process-Time"] = str(process_time)
            response.headers["X-Request-ID"] = request_id

            print(f"📤 {request.method} {request.url.path} - {response.status_code} ({process_time:.2f}ms)")

            return response

    def _setup_routes(self):
        """设置路由"""
        # 健康检查
        @self.app.get("/health", response_model=HealthResponse, tags=["System"])
        async def health_check():
            """系统健康检查"""
            uptime = (datetime.now() - self.start_time).total_seconds()

            # 检查依赖服务
            dependencies = {}

            if self.video_workflow:
                try:
                    system_status = await self.video_workflow.get_system_status()
                    dependencies["video_workflow"] = "healthy" if system_status["initialized"] else "error"
                except Exception:
                    dependencies["video_workflow"] = "error"

            if self.performance_monitor:
                try:
                    health = self.performance_monitor.get_system_health()
                    dependencies["performance_monitor"] = health.overall_status
                except Exception:
                    dependencies["performance_monitor"] = "error"

            return HealthResponse(
                status="healthy",
                timestamp=datetime.now().isoformat(),
                version=self.config.version,
                uptime_seconds=uptime,
                dependencies=dependencies
            )

        # 系统信息
        @self.app.get("/info", tags=["System"])
        async def system_info():
            """获取系统信息"""
            uptime = (datetime.now() - self.start_time).total_seconds()

            info = {
                "service": self.config.title,
                "version": self.config.version,
                "uptime_seconds": uptime,
                "uptime_formatted": self._format_uptime(uptime),
                "request_count": self.request_count,
                "start_time": self.start_time.isoformat(),
                "debug_mode": self.config.debug
            }

            # 添加性能统计
            if self.performance_monitor:
                try:
                    dashboard_data = self.performance_monitor.get_dashboard_data()
                    info["performance"] = dashboard_data
                except Exception as e:
                    info["performance_error"] = str(e)

            return info

        # WebSocket端点
        @self.app.websocket("/ws/{client_id}")
        async def websocket_endpoint(websocket, client_id: str):
            """WebSocket连接端点"""
            await self.websocket_manager.connect(websocket, client_id)
            try:
                while True:
                    data = await websocket.receive_text()
                    await self.websocket_manager.handle_message(client_id, data)
            except Exception as e:
                print(f"❌ WebSocket error for client {client_id}: {e}")
            finally:
                await self.websocket_manager.disconnect(client_id)

        # 包含API路由
        self.app.include_router(self.api_router.router, prefix="/api/v1")

    def _setup_error_handlers(self):
        """设置错误处理器"""

        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """HTTP异常处理"""
            return JSONResponse(
                status_code=exc.status_code,
                content=ErrorResponse(
                    error="HTTP_ERROR",
                    message=exc.detail,
                    timestamp=datetime.now().isoformat(),
                    request_id=request.headers.get("X-Request-ID")
                ).dict()
            )

        @self.app.exception_handler(ValueError)
        async def value_error_handler(request: Request, exc: ValueError):
            """值错误处理"""
            return JSONResponse(
                status_code=400,
                content=ErrorResponse(
                    error="VALIDATION_ERROR",
                    message=str(exc),
                    timestamp=datetime.now().isoformat(),
                    request_id=request.headers.get("X-Request-ID")
                ).dict()
            )

        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """通用异常处理"""
            error_details = {
                "type": type(exc).__name__,
                "args": str(exc.args) if exc.args else None
            } if self.config.debug else None

            return JSONResponse(
                status_code=500,
                content=ErrorResponse(
                    error="INTERNAL_ERROR",
                    message="An internal server error occurred",
                    details=error_details,
                    timestamp=datetime.now().isoformat(),
                    request_id=request.headers.get("X-Request-ID")
                ).dict()
            )

    def _format_uptime(self, uptime_seconds: float) -> str:
        """格式化运行时间"""
        days = int(uptime_seconds // 86400)
        hours = int((uptime_seconds % 86400) // 3600)
        minutes = int((uptime_seconds % 3600) // 60)
        seconds = int(uptime_seconds % 60)

        parts = []
        if days > 0:
            parts.append(f"{days}d")
        if hours > 0:
            parts.append(f"{hours}h")
        if minutes > 0:
            parts.append(f"{minutes}m")
        parts.append(f"{seconds}s")

        return " ".join(parts)

    async def start(self):
        """启动API服务器"""
        print(f"🚀 Starting API server on {self.config.host}:{self.config.port}")

        # 启动相关服务
        if self.video_workflow and not self.video_workflow.initialized:
            await self.video_workflow.initialize()

        if self.performance_monitor and not self.performance_monitor.is_running:
            await self.performance_monitor.start()

        # 启动WebSocket管理器
        await self.websocket_manager.start()

        print(f"✅ API server started successfully")

        if self.config.enable_docs:
            print(f"📖 API documentation available at: http://{self.config.host}:{self.config.port}{self.config.docs_url}")

    async def stop(self):
        """停止API服务器"""
        print("🛑 Stopping API server...")

        # 停止WebSocket管理器
        await self.websocket_manager.stop()

        # 停止相关服务
        if self.performance_monitor and self.performance_monitor.is_running:
            await self.performance_monitor.stop()

        if self.video_workflow and self.video_workflow.initialized:
            await self.video_workflow.shutdown()

        print("✅ API server stopped")

    def run(self):
        """运行API服务器"""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug,
            access_log=self.config.debug,
            timeout_keep_alive=30
        )

    async def run_async(self):
        """异步运行API服务器"""
        config = uvicorn.Config(
            self.app,
            host=self.config.host,
            port=self.config.port,
            debug=self.config.debug,
            access_log=self.config.debug
        )
        server = uvicorn.Server(config)

        # 启动前初始化
        await self.start()

        try:
            await server.serve()
        finally:
            await self.stop()

    def get_openapi_schema(self) -> Dict[str, Any]:
        """获取OpenAPI架构"""
        return self.app.openapi()

    def get_route_list(self) -> List[Dict[str, Any]]:
        """获取路由列表"""
        routes = []
        for route in self.app.routes:
            if hasattr(route, 'methods') and hasattr(route, 'path'):
                routes.append({
                    'path': route.path,
                    'methods': list(route.methods),
                    'name': getattr(route, 'name', None)
                })
        return routes

    def add_custom_route(self, path: str, endpoint: Callable, methods: List[str] = None,
                        tags: List[str] = None, **kwargs):
        """添加自定义路由"""
        if methods is None:
            methods = ["GET"]

        for method in methods:
            method_lower = method.lower()
            if hasattr(self.app, method_lower):
                decorator = getattr(self.app, method_lower)
                decorator(path, tags=tags, **kwargs)(endpoint)

    def get_metrics(self) -> Dict[str, Any]:
        """获取API服务器指标"""
        uptime = (datetime.now() - self.start_time).total_seconds()

        metrics = {
            "uptime_seconds": uptime,
            "request_count": self.request_count,
            "requests_per_second": self.request_count / uptime if uptime > 0 else 0,
            "active_websocket_connections": len(self.websocket_manager.connections),
            "rate_limiter_stats": self.rate_limiter.get_stats()
        }

        if self.performance_monitor:
            try:
                perf_metrics = self.performance_monitor.get_dashboard_data()
                metrics["performance"] = perf_metrics
            except Exception:
                metrics["performance"] = "error"

        return metrics