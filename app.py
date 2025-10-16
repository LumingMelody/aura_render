#!/usr/bin/env python3
"""
Aura Render FastAPI Application

Main FastAPI application with proper configuration management,
error handling, and API endpoints for the video generation pipeline.
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# 加载环境变量，确保所有模块都能访问
from dotenv import load_dotenv
from pathlib import Path

# 获取项目根目录并加载.env文件
project_root = Path(__file__).parent
env_path = project_root / '.env'
load_dotenv(dotenv_path=env_path)

# 验证关键环境变量已加载
import os
if not os.getenv('OSS_ACCESS_KEY_ID'):
    print("⚠️ 警告: OSS配置未加载，图生视频将使用base64方式")

import asyncio
import json
import logging
import os
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime
from contextlib import asynccontextmanager

from fastapi import FastAPI, BackgroundTasks, HTTPException, Depends, status
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, model_validator
from sqlalchemy.orm import Session
import httpx

from config import settings, get_settings
from database import get_db, init_db, TaskService, TaskStatus
from materials_api import materials_router
# from ai_api import ai_router  # ❌ 已删除 /ai/generate 接口
from render_api import render_router

# Import Celery task management
from api.task_endpoints import router as task_router
from task_queue.task_manager import get_task_manager

# Import system management APIs (保留有用的系统管理接口)
# from api.image_endpoints import router as image_router  # ❌ 已删除 /image/generate 接口
from api.templates_endpoints import router as templates_router
from api.analytics_endpoints import router as analytics_router
# from api.batch_endpoints import router as batch_router  # ❌ 已删除批量处理接口
from api.auth_endpoints import router as auth_router
from api.export_endpoints import router as export_router
from api.websocket_endpoints import router as websocket_router
# from api.ai_optimization_endpoints import router as ai_optimization_router  # ❌ 已删除AI优化接口

# =============================
# 日志系统初始化（必须在其他导入之前配置）
# =============================
from utils.logger import setup_logging, get_logger, LogCategory
from pathlib import Path

# ⚠️ 重要：先配置第三方库日志级别，再初始化日志系统
logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)  # 只记录警告和错误
logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.dialects").setLevel(logging.WARNING)
logging.getLogger("sqlalchemy.orm").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("uvicorn.access").setLevel(logging.WARNING)  # 减少访问日志

# 配置日志系统
log_dir = Path(__file__).parent / "logs"
setup_logging(
    log_dir=log_dir,
    log_level=logging.INFO if not settings.is_development else logging.DEBUG,
    enable_console=True,
    enable_json=True,
    enable_performance=True,
    max_file_size=100 * 1024 * 1024  # 100MB
)

# 使用增强的日志系统
logger = get_logger("aura_render.app").with_context(category=LogCategory.SYSTEM)

# =============================
# Application Lifespan Events
# =============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events"""
    # Startup
    logger.info("🚀 Aura Render API starting up...")
    logger.info(f"📍 Environment: {'Development' if settings.is_development else 'Production'}")
    # logger.info(f"🔧 AI Model: {settings.ai.qwen_model_name}")

    # Initialize database
    try:
        init_db()
        logger.info("✅ Database initialized")
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")

    # ⚠️ 不再在 startup 时初始化 VGP 系统
    # VGP 系统将在第一次使用时自动初始化（在后台任务中）
    # 这样可以避免阻塞第一次 API 请求
    logger.info("ℹ️  VGP workflow system will be initialized on first use")

    yield

    # Shutdown
    logger.info("👋 Aura Render API shutting down...")

    # VGP workflow system shutdown (if initialized)
    try:
        from vgp_api import workflow_system
        if workflow_system:
            logger.info("🛑 Shutting down VGP workflow system...")
            await workflow_system.shutdown()
            logger.info("✅ VGP workflow system shutdown complete")
        else:
            logger.info("ℹ️  VGP workflow system was not initialized, no shutdown needed")
    except Exception as e:
        logger.error(f"❌ VGP workflow shutdown failed: {e}")

# =============================
# FastAPI App Configuration
# =============================

app = FastAPI(
    title=settings.api.title,
    description=settings.api.description,
    version=settings.api.version,
    docs_url=settings.api.docs_url if settings.is_development else None,
    redoc_url=settings.api.redoc_url if settings.is_development else None,
    lifespan=lifespan
)

# Add CORS middleware for development
if settings.is_development:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, specify exact origins
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Include API routers
app.include_router(materials_router)
# app.include_router(ai_router)  # ❌ 已删除
app.include_router(render_router)
app.include_router(task_router)  # Add Celery task management endpoints
# app.include_router(image_router)  # ❌ 已删除 image generation endpoints
app.include_router(templates_router)  # Add templates system endpoints
app.include_router(analytics_router)  # Add analytics endpoints
# app.include_router(batch_router)  # ❌ 已删除 batch processing endpoints
app.include_router(auth_router)  # Add authentication endpoints
app.include_router(export_router)  # Add export and cloud storage endpoints
app.include_router(websocket_router)  # Add WebSocket endpoints
# app.include_router(ai_optimization_router)  # ❌ 已删除 AI optimization endpoints

# ✨ 新增：VGP新工作流API
try:
    from vgp_api import vgp_router
    app.include_router(vgp_router)  # Add VGP new workflow endpoints
    logger.info("✅ VGP新工作流API已加载: /vgp/generate")
except ImportError as e:
    logger.warning(f"⚠️ VGP新工作流API加载失败: {e}")

# =============================
# Pydantic Models
# =============================

# 新增多模态输入模型
class ReferenceMedia(BaseModel):
    """参考媒体输入模型"""
    url: str = Field(..., description="媒体文件路径或URL")
    type: str = Field(..., description="媒体类型 (style_reference, content_reference, etc.)")
    weight: float = Field(default=1.0, description="权重 (0.0-1.0)", ge=0.0, le=1.0)

class ReferenceMediaGroup(BaseModel):
    """参考媒体组模型"""
    reference_videos: Optional[List[ReferenceMedia]] = Field(default=None, description="参考视频列表", max_items=3)
    reference_images: Optional[List[ReferenceMedia]] = Field(default=None, description="参考图片列表", max_items=5)
    product_images: Optional[List[ReferenceMedia]] = Field(default=None, description="产品图片列表", max_items=10)

class ConversationContext(BaseModel):
    """对话上下文模型"""
    conversation_id: str = Field(..., description="对话ID")
    message_id: str = Field(..., description="消息ID")
    is_regeneration: bool = Field(default=False, description="是否为重新生成")
    previous_results: Optional[Dict[str, Any]] = Field(default=None, description="之前的生成结果")

class VideoGenerationRequest(BaseModel):
    """Video generation request model - 支持多模态输入"""
    # 支持标准字段名
    theme: Optional[str] = Field(None, description="视频主题", min_length=1, max_length=200)
    keywords: Optional[List[str]] = Field(None, description="关键词列表", min_items=1, max_items=10)
    target_duration: Optional[int] = Field(None, description="目标时长（秒）", ge=5, le=3600)
    user_description: Optional[str] = Field(None, description="用户描述", min_length=1, max_length=1000)

    # 向后兼容的_id字段名（可选）
    theme_id: Optional[str] = Field(None, description="视频主题（别名）", min_length=1, max_length=200)
    keywords_id: Optional[List[str]] = Field(None, description="关键词列表（别名）", min_items=1, max_items=10)
    target_duration_id: Optional[int] = Field(None, description="目标时长（秒，别名）", ge=5, le=3600)
    user_description_id: Optional[str] = Field(None, description="用户描述（别名）", min_length=1, max_length=1000)
    template: Optional[str] = Field(default="vgp_new_pipeline", description="VGP管道")

    # 新增多模态输入支持
    reference_media: Optional[ReferenceMediaGroup] = Field(default=None, description="参考媒体输入")
    conversation_context: Optional[ConversationContext] = Field(default=None, description="对话上下文")

    @model_validator(mode='after')
    def validate_required_fields(self):
        """验证必需字段，确保每对字段至少有一个存在"""
        # 验证theme字段
        if not self.theme and not self.theme_id:
            raise ValueError("theme 或 theme_id 必须提供其中之一")

        # 验证keywords字段
        if not self.keywords and not self.keywords_id:
            raise ValueError("keywords 或 keywords_id 必须提供其中之一")

        # 验证target_duration字段
        if self.target_duration is None and self.target_duration_id is None:
            raise ValueError("target_duration 或 target_duration_id 必须提供其中之一")

        # 验证user_description字段
        if not self.user_description and not self.user_description_id:
            raise ValueError("user_description 或 user_description_id 必须提供其中之一")

        # 规范化字段：如果提供了标准字段名，自动设置_id字段
        if self.theme and not self.theme_id:
            self.theme_id = self.theme
        elif self.theme_id and not self.theme:
            self.theme = self.theme_id

        if self.keywords and not self.keywords_id:
            self.keywords_id = self.keywords
        elif self.keywords_id and not self.keywords:
            self.keywords = self.keywords_id

        if self.target_duration is not None and self.target_duration_id is None:
            self.target_duration_id = self.target_duration
        elif self.target_duration_id is not None and self.target_duration is None:
            self.target_duration = self.target_duration_id

        if self.user_description and not self.user_description_id:
            self.user_description_id = self.user_description
        elif self.user_description_id and not self.user_description:
            self.user_description = self.user_description_id

        return self

    class Config:
        json_schema_extra = {
            "example": {
                "theme_id": "AI产品展示",
                "keywords_id": ["人工智能", "创新", "技术"],
                "target_duration_id": 60,
                "user_description_id": "制作一个60秒的AI产品宣传视频",
                "reference_media": {
                    "reference_videos": [
                        {"url": "path/to/reference.mp4", "type": "style_reference", "weight": 0.7}
                    ],
                    "reference_images": [
                        {"url": "path/to/style.jpg", "type": "style_guide", "weight": 0.8}
                    ],
                    "product_images": [
                        {"url": "path/to/product.jpg", "type": "main_product", "weight": 1.0}
                    ]
                },
                "conversation_context": {
                    "conversation_id": "conv_12345",
                    "message_id": "msg_001",
                    "is_regeneration": False,
                    "previous_results": None
                }
            }
        }


class TaskResponse(BaseModel):
    """Task response model"""
    task_id: str
    status: str
    message: str
    timestamp: datetime


class HealthResponse(BaseModel):
    """Health check response model"""
    status: str
    timestamp: datetime
    version: str
    environment: str


# =============================
# VGP Node Management System
# =============================

# Import VGP Protocol Tools
from video_generate_protocol.vgp_protocol import VGPProtocol, VGPDocument, VGPNodeData, NodeStatus
from video_generate_protocol.vgp_data_transformer import (
    VGPDataMapper, VGPDataFixer,
    prepare_context_for_node, transform_node_result
)

# Import VGP nodes (16个标准节点)
from video_generate_protocol.nodes.video_type_identification_node import VideoTypeIdentificationNode
from video_generate_protocol.nodes.emotion_analysis_node import EmotionAnalysisNode
from video_generate_protocol.nodes.shot_block_generation_node import ShotBlockGenerationNode
from video_generate_protocol.nodes.bgm_anchor_planning_node import BGMAanchorPlanningNode
from video_generate_protocol.nodes.bgm_composition_node import BGMCompositionNode
from video_generate_protocol.nodes.asset_request_node import AssetRequestNode
from video_generate_protocol.nodes.audio_processing_node import AudioProcessingNode
from video_generate_protocol.nodes.sfx_integration_node import SFXIntegrationNode
from video_generate_protocol.nodes.transition_selection_node import TransitionSelectionNode
from video_generate_protocol.nodes.filter_application_node import FilterApplicationNode
from video_generate_protocol.nodes.dynamic_effects_node import DynamicEffectsNode
from video_generate_protocol.nodes.aux_media_insertion_node import AuxMediaInsertionNode
from video_generate_protocol.nodes.aux_text_insertion_node import AuxTextInsertionNode
from video_generate_protocol.nodes.subtitle_node import SubtitleNode
from video_generate_protocol.nodes.intro_outro_node import IntroOutroNode
from video_generate_protocol.nodes.timeline_integration_node import TimelineIntegrationNode


def extract_node_outputs(node_id: str, node_output: dict) -> dict:
    """
    从节点输出中提取特定字段，供下游节点使用

    Args:
        node_id: 节点ID
        node_output: 节点输出数据

    Returns:
        提取的输出字段字典
    """
    extracted = {}

    # 根据节点类型提取特定输出字段
    if node_id == "video_type_identification":
        # 提取视频类型和结构模板
        if "video_type_id" in node_output:
            extracted["video_type_id"] = node_output["video_type_id"]
        if "structure_template_id" in node_output:
            extracted["structure_template_id"] = node_output["structure_template_id"]

    elif node_id == "emotion_analysis":
        # 提取情感分析结果
        if "emotions_id" in node_output:
            extracted["emotions_id"] = node_output["emotions_id"]
        # 也可能在不同的键下
        if "emotions" in node_output:
            extracted["emotions_id"] = node_output["emotions"]

    elif node_id == "shot_block_generation":
        # 提取分镜数据
        if "shot_blocks_id" in node_output:
            extracted["shot_blocks_id"] = node_output["shot_blocks_id"]
        if "shot_blocks" in node_output:
            extracted["shot_blocks_id"] = node_output["shot_blocks"]

    elif node_id == "bgm_anchor_planning":
        if "bgm_anchors" in node_output:
            extracted["bgm_anchors"] = node_output["bgm_anchors"]
        if "bgm_timeline" in node_output:
            extracted["bgm_timeline"] = node_output["bgm_timeline"]

    elif node_id == "bgm_composition":
        if "bgm_tracks" in node_output:
            extracted["bgm_tracks"] = node_output["bgm_tracks"]
        if "bgm_tracks_id" in node_output:
            extracted["bgm_tracks_id"] = node_output["bgm_tracks_id"]
        if "bgm_composition_id" in node_output:
            extracted["bgm_composition_id"] = node_output["bgm_composition_id"]

    elif node_id == "audio_processing":
        if "audio_track_id" in node_output:
            extracted["audio_track_id"] = node_output["audio_track_id"]
        if "audio_file_path" in node_output:
            extracted["audio_file_path"] = node_output["audio_file_path"]

    elif node_id == "transition_selection":
        if "transitions" in node_output:
            extracted["transitions"] = node_output["transitions"]
        if "preliminary_sequence_id" in node_output:
            extracted["preliminary_sequence_id"] = node_output["preliminary_sequence_id"]

    elif node_id == "filter_application":
        if "filters" in node_output:
            extracted["filters"] = node_output["filters"]
        if "transition_sequence_id" in node_output:
            extracted["transition_sequence_id"] = node_output["transition_sequence_id"]

    elif node_id == "dynamic_effects":
        if "effects" in node_output:
            extracted["effects"] = node_output["effects"]
        if "filter_sequence_id" in node_output:
            extracted["filter_sequence_id"] = node_output["filter_sequence_id"]

    elif node_id == "timeline_integration":
        if "video_clips" in node_output:
            extracted["video_clips"] = node_output["video_clips"]
        if "audio_tracks" in node_output:
            extracted["audio_tracks"] = node_output["audio_tracks"]

    # 通用提取：如果输出字段以_id结尾，提取它
    for key, value in node_output.items():
        if key.endswith("_id") and key not in extracted:
            extracted[key] = value

    return extracted


class VGPNodeManager:
    """VGP Standard Node Manager - 管理18个标准VGP节点"""

    def __init__(self):
        self.nodes = {}
        self.node_instances = {}
        self.vgp_protocol = VGPProtocol()  # VGP协议管理器
        self.data_mapper = VGPDataMapper()  # 数据映射器
        self.data_fixer = VGPDataFixer()  # 数据修复器
        self._initialize_vgp_nodes()

    def _initialize_vgp_nodes(self):
        """初始化所有VGP标准节点实例"""
        # 定义VGP节点执行顺序和配置
        self.vgp_node_sequence = [
            ('video_type_identification', VideoTypeIdentificationNode, '视频类型识别'),
            ('emotion_analysis', EmotionAnalysisNode, '情感基调分析'),
            ('shot_block_generation', ShotBlockGenerationNode, '分镜块生成'),
            ('bgm_anchor_planning', BGMAanchorPlanningNode, 'BGM锚点规划'),
            ('bgm_composition', BGMCompositionNode, 'BGM合成查找'),
            ('asset_request', AssetRequestNode, '素材需求解析'),
            ('audio_processing', AudioProcessingNode, '音频处理'),
            ('sfx_integration', SFXIntegrationNode, '音效添加'),
            ('transition_selection', TransitionSelectionNode, '转场选择'),
            ('filter_application', FilterApplicationNode, '滤镜应用'),
            ('dynamic_effects', DynamicEffectsNode, '动态特效添加'),
            ('aux_media_insertion', AuxMediaInsertionNode, '额外媒体插入'),
            ('aux_text_insertion', AuxTextInsertionNode, '装饰文字插入'),
            ('subtitle_generation', SubtitleNode, '字幕生成'),
            ('intro_outro', IntroOutroNode, '片头片尾生成'),
            ('timeline_integration', TimelineIntegrationNode, '最终时间线整合')
        ]

        # 创建节点实例
        for node_id, node_class, description in self.vgp_node_sequence:
            try:
                # 创建节点实例
                instance = node_class(node_id=f"{node_id}_{uuid.uuid4().hex[:8]}", name=description)
                self.node_instances[node_id] = instance

                # 注册节点信息
                self.nodes[node_id] = {
                    'name': node_class.__name__,
                    'description': description,
                    'status': 'available',
                    'instance': instance
                }
                logger.info(f"✅ VGP节点已加载: {node_id} - {description}")

            except Exception as e:
                logger.error(f"❌ VGP节点加载失败: {node_id} - {e}")
                self.nodes[node_id] = {
                    'name': node_class.__name__,
                    'description': f"{description} (加载失败)",
                    'status': 'error',
                    'error': str(e)
                }

        logger.info(f"🎯 VGP节点系统初始化完成: {len(self.node_instances)}/{len(self.vgp_node_sequence)} 个节点可用")
        logger.info(f"✅ Loaded {len(self.nodes)} nodes")
    
    def get_available_nodes(self) -> Dict[str, Any]:
        """Get list of available nodes"""
        return {k: v for k, v in self.nodes.items() if v['status'] == 'available'}
    
    async def execute_node(self, node_id: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行VGP标准节点 - 带数据修复和转换"""
        if node_id not in self.nodes:
            raise ValueError(f"VGP节点不存在: {node_id}")

        node_info = self.nodes[node_id]
        if node_info['status'] != 'available':
            raise RuntimeError(f"VGP节点不可用: {node_id} - {node_info.get('error', '未知错误')}")

        logger.info(f"🎬 执行VGP节点: {node_id} - {node_info['description']}")

        try:
            # 获取节点实例
            node_instance = self.node_instances[node_id]

            # DEBUG: 打印输入 context
            print(f"\n🎬 [execute_node] Starting node: {node_id}")
            print(f"🎬 [execute_node] Input context keys: {list(context.keys())}")
            for key in ['theme_id', 'keywords_id', 'target_duration_id', 'user_description_id']:
                if key in context:
                    print(f"🎬 [execute_node] context[{key}] = {context[key]}")

            # 修复上下文中的数据问题
            fixed_context = self.data_fixer.fix_all_node_outputs(context)
            print(f"🎬 [execute_node] Fixed context keys: {list(fixed_context.keys())}")

            # 为节点准备输入数据
            prepared_input = prepare_context_for_node(node_id, fixed_context)
            print(f"🎬 [execute_node] Prepared input keys: {list(prepared_input.keys())}")

            # 合并准备好的输入到上下文
            execution_context = {**fixed_context, **prepared_input}
            print(f"🎬 [execute_node] Execution context keys: {list(execution_context.keys())}")
            for key in ['theme_id', 'keywords_id', 'target_duration_id', 'user_description_id']:
                if key in execution_context:
                    print(f"🎬 [execute_node] execution_context[{key}] = {execution_context[key]}")
            print(f"🎬 [execute_node] Calling node.generate()...\n")

            # 执行节点的generate方法
            result = await node_instance.generate(execution_context)

            # 转换输出为标准格式
            standardized_result = transform_node_result(node_id, result)

            logger.info(f"✅ VGP节点执行成功: {node_id}")
            logger.debug(f"📋 节点输出: {standardized_result}")

            # 返回标准化的结果，包含节点ID标识
            return {node_id: standardized_result}

        except Exception as e:
            error_msg = f"VGP节点执行失败 {node_id}: {str(e)}"
            logger.error(error_msg)

            # 返回错误信息但不阻断流程（某些节点失败可以继续）
            return {
                node_id: {
                    'error': error_msg,
                    'node_id': node_id,
                    'status': 'failed'
                }
            }




# Global VGP node manager
node_manager = VGPNodeManager()

# =============================
# Utility Functions
# =============================

async def send_callback(task_id: str, node_index: int, status: str, message: str):
    """Send callback notification"""
    # Skip callback in development mode or when callback URL is not configured
    if settings.is_development or not settings.external.callback_url or \
       settings.external.callback_url.startswith("http://192.168.10.16"):
        logger.info(f"📋 Task {task_id} - Node {node_index}: {status} - {message}")
        return
    
    task_status = 3 if status == "failed" else 2
    
    payload = {
        "taskId": task_id,
        "taskStatus": task_status,
        "key": f"node{node_index}",
        "val": status,
        "remark": message
    }
    
    try:
        async with httpx.AsyncClient(timeout=settings.performance.http_timeout) as client:
            response = await client.post(settings.external.callback_url, json=payload)
            if response.status_code == 200:
                logger.info(f"✅ Callback sent for task {task_id}")
            else:
                logger.warning(f"⚠️ Callback failed with status {response.status_code}")
    except Exception as e:
        logger.error(f"❌ Callback error for task {task_id}: {e}")


# =============================
# API Endpoints
# =============================

@app.get("/")
async def read_index():
    """Serve main web application"""
    return FileResponse('static/index.html')

@app.get("/admin")
async def read_admin():
    """Serve admin dashboard"""
    return FileResponse('static/admin.html')

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now(),
        version=settings.api.version,
        environment="development" if settings.is_development else "production"
    )


@app.get("/nodes")
async def list_nodes():
    """List available processing nodes"""
    nodes = node_manager.get_available_nodes()
    return {
        "nodes": nodes,
        "count": len(nodes),
        "timestamp": datetime.now()
    }


@app.post("/generate", response_model=TaskResponse)
async def generate_video(
    request: VideoGenerationRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db),
    settings: Any = Depends(get_settings)
):
    """Generate video from user input"""
    try:
        # Create task in database
        task = TaskService.create_task(
            db=db,
            theme=request.theme_id,
            keywords=request.keywords_id,
            target_duration=request.target_duration_id,
            user_description=request.user_description_id
        )
        
        logger.info(f"🚀 Starting video generation task: {task.task_id}")
        
        # Add background task for processing
        background_tasks.add_task(
            process_video_generation,
            task_id=task.task_id,
            request=request
        )
        
        return TaskResponse(
            task_id=task.task_id,
            status="started",
            message="视频生成任务已启动",
            timestamp=datetime.now()
        )
    except Exception as e:
        logger.error(f"❌ Failed to create task: {e}")
        raise HTTPException(status_code=500, detail="Failed to create task")


@app.get("/task/{task_id}/status")
async def get_task_status(task_id: str, db: Session = Depends(get_db)):
    """Get task status from database"""
    task = TaskService.get_task_by_id(db, task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Task not found")
    
    return task.to_dict()


@app.get("/tasks")
async def list_tasks(
    status: Optional[str] = None,
    limit: int = 20,
    db: Session = Depends(get_db)
):
    """List recent tasks, optionally filtered by status"""
    try:
        if status:
            # Convert string status to TaskStatus enum
            task_status = TaskStatus(status.lower())
            tasks = TaskService.get_tasks_by_status(db, task_status, limit)
        else:
            tasks = TaskService.get_recent_tasks(db, limit)
        
        return {
            "tasks": [task.to_dict() for task in tasks],
            "count": len(tasks),
            "timestamp": datetime.now()
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"Invalid status: {status}")


@app.get("/database/stats")
async def get_database_stats(db: Session = Depends(get_db)):
    """Get database statistics"""
    from database.base import get_db_stats
    from database.models import Task, Project
    
    try:
        total_tasks = db.query(Task).count()
        total_projects = db.query(Project).count()
        
        pending_tasks = db.query(Task).filter(Task.status == TaskStatus.PENDING).count()
        processing_tasks = db.query(Task).filter(Task.status == TaskStatus.PROCESSING).count()
        completed_tasks = db.query(Task).filter(Task.status == TaskStatus.COMPLETED).count()
        failed_tasks = db.query(Task).filter(Task.status == TaskStatus.FAILED).count()
        
        return {
            "database": get_db_stats(),
            "tasks": {
                "total": total_tasks,
                "pending": pending_tasks,
                "processing": processing_tasks,
                "completed": completed_tasks,
                "failed": failed_tasks
            },
            "projects": {
                "total": total_projects
            },
            "timestamp": datetime.now()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get stats: {e}")


# =============================
# Helper Functions
# =============================

def generate_vgp_summary(results: dict) -> dict:
    """生成VGP分析摘要"""
    try:
        # 安全提取数据，处理可能的复杂对象
        def safe_extract(data, *keys, default=None):
            """安全提取嵌套字典数据"""
            current = data
            for key in keys:
                if isinstance(current, dict) and key in current:
                    current = current[key]
                else:
                    return default
            return current

        summary = {
            "video_type": safe_extract(results, 'video_type_identification', 'video_type', default='未知'),
            "emotions": safe_extract(results, 'emotion_analysis', 'emotions', default={}),
            "shot_count": len(safe_extract(results, 'shot_block_generation', 'shot_blocks', default=[])),
            "asset_count": len(safe_extract(results, 'asset_request', 'assets', default=[])),
            "subtitle_count": len(safe_extract(results, 'subtitle_generation', 'subtitles', default=[]))
        }
        return summary
    except Exception as e:
        print(f"生成VGP摘要失败: {e}")
        return {"error": str(e)}

def serialize_results(results: dict) -> dict:
    """序列化VGP结果为可JSON化的格式"""
    visited = set()

    def serialize_object(obj):
        """递归序列化对象,防止循环引用"""
        # 检测循环引用
        obj_id = id(obj)
        if obj_id in visited:
            return f"<circular_ref:{type(obj).__name__}@{hex(obj_id)[-6:]}>"

        # 基本类型直接返回
        if isinstance(obj, (str, int, float, bool, type(None))):
            return obj

        # 标记为已访问
        visited.add(obj_id)

        try:
            if hasattr(obj, '__dict__'):
                # 如果是自定义对象,转换为字典
                return {k: serialize_object(v) for k, v in obj.__dict__.items() if not k.startswith('_')}
            elif isinstance(obj, dict):
                return {k: serialize_object(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [serialize_object(item) for item in obj]
            else:
                # 其他类型转换为字符串
                return str(obj)
        finally:
            # 处理完后移除标记,允许同一对象在不同分支中出现
            visited.discard(obj_id)

    return serialize_object(results)


# =============================
# Multimodal Processing Functions
# =============================

async def process_reference_media(reference_media: ReferenceMediaGroup) -> List[Dict[str, Any]]:
    """处理参考媒体输入"""
    from multimodal.analyzers.reference_video_analyzer import ReferenceVideoAnalyzer

    processed_media = []

    try:
        # 初始化分析器
        video_analyzer = ReferenceVideoAnalyzer()

        # 处理参考视频
        if reference_media.reference_videos:
            logger.info(f"🎬 处理 {len(reference_media.reference_videos)} 个参考视频")

            for video_ref in reference_media.reference_videos:
                try:
                    analysis_result = await video_analyzer.analyze_reference_video(
                        video_ref.url,
                        video_ref.type,
                        video_ref.weight
                    )

                    processed_media.append({
                        "media_type": "video",
                        "original_input": video_ref.dict(),
                        "analysis_result": analysis_result,
                        "processing_status": "success"
                    })

                    logger.info(f"✅ 视频分析完成: {video_ref.url}")

                except Exception as e:
                    logger.error(f"❌ 视频分析失败 {video_ref.url}: {e}")
                    processed_media.append({
                        "media_type": "video",
                        "original_input": video_ref.dict(),
                        "error": str(e),
                        "processing_status": "failed"
                    })

        # 处理参考图片 - Day 2功能实现
        if reference_media.reference_images:
            logger.info(f"🖼️  处理 {len(reference_media.reference_images)} 个参考图片")

            from multimodal.analyzers.reference_image_analyzer import ReferenceImageAnalyzer
            image_analyzer = ReferenceImageAnalyzer()

            for image_ref in reference_media.reference_images:
                try:
                    analysis_result = await image_analyzer.analyze_reference_image(
                        image_ref.url,
                        image_ref.type,
                        image_ref.weight
                    )

                    processed_media.append({
                        "media_type": "image",
                        "original_input": image_ref.dict(),
                        "analysis_result": analysis_result,
                        "processing_status": "success"
                    })

                    logger.info(f"✅ 图片分析完成: {image_ref.url}")

                except Exception as e:
                    logger.error(f"❌ 图片分析失败 {image_ref.url}: {e}")
                    processed_media.append({
                        "media_type": "image",
                        "original_input": image_ref.dict(),
                        "error": str(e),
                        "processing_status": "failed"
                    })

        # 处理产品图片 - Day 2功能实现
        if reference_media.product_images:
            logger.info(f"📦 处理 {len(reference_media.product_images)} 个产品图片")

            # 复用图片分析器进行产品图分析
            if 'image_analyzer' not in locals():
                from multimodal.analyzers.reference_image_analyzer import ReferenceImageAnalyzer
                image_analyzer = ReferenceImageAnalyzer()

            for product_ref in reference_media.product_images:
                try:
                    analysis_result = await image_analyzer.analyze_reference_image(
                        product_ref.url,
                        product_ref.type,
                        product_ref.weight
                    )

                    processed_media.append({
                        "media_type": "product_image",
                        "original_input": product_ref.dict(),
                        "analysis_result": analysis_result,
                        "processing_status": "success"
                    })

                    logger.info(f"✅ 产品图分析完成: {product_ref.url}")

                except Exception as e:
                    logger.error(f"❌ 产品图分析失败 {product_ref.url}: {e}")
                    processed_media.append({
                        "media_type": "product_image",
                        "original_input": product_ref.dict(),
                        "error": str(e),
                        "processing_status": "failed"
                    })

        logger.info(f"🎯 多模态处理完成: {len(processed_media)} 个媒体文件")
        return processed_media

    except Exception as e:
        logger.error(f"❌ 多模态处理失败: {e}")
        return [{
            "error": str(e),
            "processing_status": "failed"
        }]

# =============================
# Background Tasks
# =============================

async def process_video_generation(task_id: str, request: VideoGenerationRequest):
    """Process video generation in background with database persistence"""
    from database.base import SessionLocal
    
    db = SessionLocal()
    try:
        # Update task status to processing
        TaskService.update_task_status(
            db, task_id, TaskStatus.PROCESSING, 
            progress=0.0, message="开始处理视频生成任务"
        )
        
        logger.info(f"🚀 Starting background processing for task {task_id}")
        print(f"🚀 Starting background processing for task {task_id}")
        
        # Convert request to context with multimodal support
        context = {
            "theme_id": request.theme_id,
            "keywords_id": request.keywords_id,
            "target_duration_id": request.target_duration_id,
            "user_description_id": request.user_description_id
        }

        # 处理多模态输入
        if request.reference_media:
            processed_media = await process_reference_media(request.reference_media)
            context["reference_media"] = processed_media
            logger.info(f"🎬 多模态输入处理完成: {len(processed_media)} 个媒体文件")

            # Day 3: 多模态融合处理
            if processed_media:
                try:
                    from multimodal.fusion.multimodal_fusion_engine import MultiModalFusionEngine

                    fusion_engine = MultiModalFusionEngine()

                    # 分类分析结果
                    categorized_media = {
                        'video_analyses': [],
                        'image_analyses': [],
                        'product_analyses': []
                    }

                    for media in processed_media:
                        if media.get('processing_status') == 'success':
                            analysis_result = media.get('analysis_result')
                            if analysis_result:
                                media_type = media.get('media_type', '')
                                if media_type == 'video':
                                    categorized_media['video_analyses'].append(analysis_result)
                                elif media_type == 'image':
                                    categorized_media['image_analyses'].append(analysis_result)
                                elif media_type == 'product_image':
                                    categorized_media['product_analyses'].append(analysis_result)

                    # 执行多模态融合
                    fusion_result = await fusion_engine.fuse_multimodal_inputs(categorized_media)
                    context["multimodal_fusion"] = fusion_result

                    logger.info(f"🔄 多模态融合完成，置信度: {fusion_result.get('confidence_score', 0):.2f}")

                    # 更新任务进度
                    TaskService.update_task_status(
                        db, task_id, TaskStatus.PROCESSING,
                        progress=10.0, message=f"多模态融合完成 (置信度: {fusion_result.get('confidence_score', 0):.2f})"
                    )

                except Exception as e:
                    logger.error(f"❌ 多模态融合失败: {e}")
                    context["multimodal_fusion"] = {"error": str(e), "fallback": True}

        # 处理对话上下文 - 智能对话修改功能
        conversation_result = None
        if request.conversation_context:
            try:
                from conversation.conversation_manager import conversation_manager

                # 处理对话请求，进行意图分析
                conversation_result = await conversation_manager.process_conversation_request(
                    {
                        "theme_id": request.theme_id,
                        "keywords_id": request.keywords_id,
                        "target_duration_id": request.target_duration_id,
                        "user_description_id": request.user_description_id
                    },
                    request.conversation_context.dict()
                )

                context["conversation_context"] = request.conversation_context.dict()
                context["is_regeneration"] = request.conversation_context.is_regeneration

                # 如果识别到增量修改模式
                if conversation_result.get("incremental_mode"):
                    logger.info(f"🔄 增量修改模式: {conversation_result.get('intent_analysis')}")
                    context["incremental_mode"] = True
                    context["modifications"] = conversation_result.get("modifications")
                    context["nodes_to_update"] = conversation_result.get("nodes_to_update", [])
                    context["skip_nodes"] = conversation_result.get("skip_nodes", [])

                    # 更新任务消息
                    TaskService.update_task_status(
                        db, task_id, TaskStatus.PROCESSING,
                        progress=5.0,
                        message=f"智能对话分析: {conversation_result.get('intent_analysis', {}).get('modification_type', '调整')}"
                    )
                else:
                    logger.info(f"💬 对话上下文: {request.conversation_context.conversation_id}")

            except Exception as e:
                logger.warning(f"对话管理器处理失败，使用基础模式: {e}")
                context["conversation_context"] = request.conversation_context.dict()
                context["is_regeneration"] = request.conversation_context.is_regeneration

        # 如果是重新生成且没有增量修改，加载之前的结果
        if context.get("is_regeneration") and not context.get("incremental_mode"):
            if request.conversation_context and request.conversation_context.previous_results:
                context.update(request.conversation_context.previous_results)
                logger.info("🔄 重新生成模式：已加载之前的结果")
        
        logger.info(f"🎯 Task context: {context}")
        print(f"🎯 Processing request: {request.theme_id} - {request.target_duration_id}s")

        # 创建VGP文档来记录整个流程
        vgp_document = node_manager.vgp_protocol.create_document({
            'task_id': task_id,
            'theme': request.theme_id,
            'keywords': request.keywords_id,
            'duration': request.target_duration_id,
            'description': request.user_description_id
        })
        vgp_document.task_id = task_id

        # Execute VGP complete pipeline (expanded to 16-node pipeline)
        results = {}
        # 使用完整的16个VGP标准节点序列
        complete_vgp_nodes = [
            'video_type_identification',  # 视频类型识别
            'emotion_analysis',          # 情感基调分析
            'shot_block_generation',     # 分镜块生成
            'bgm_anchor_planning',       # BGM锚点规划
            'bgm_composition',           # BGM合成查找
            'asset_request',             # 素材需求解析
            'audio_processing',          # 音频处理
            'sfx_integration',           # 音效添加
            'transition_selection',      # 转场选择
            'filter_application',        # 滤镜应用
            'dynamic_effects',           # 动态特效添加
            'aux_media_insertion',       # 额外媒体插入
            'aux_text_insertion',        # 装饰文字插入
            'subtitle_generation',       # 字幕生成
            'intro_outro',               # 片头片尾生成
            'timeline_integration'       # 最终时间线整合
        ]
        
        # 智能节点执行 - 支持增量修改和节点跳过
        nodes_to_execute = complete_vgp_nodes.copy()
        skip_nodes = context.get("skip_nodes", [])
        nodes_to_update = context.get("nodes_to_update", [])
        incremental_mode = context.get("incremental_mode", False)

        # 如果是增量模式，过滤节点列表
        if incremental_mode and skip_nodes:
            nodes_to_execute = [node for node in complete_vgp_nodes if node not in skip_nodes]
            logger.info(f"🔄 增量模式：跳过节点 {skip_nodes}")

        for i, node_id in enumerate(nodes_to_execute, 1):
            try:
                progress = (i - 1) / len(nodes_to_execute) * 100

                # 检查是否需要更新此节点
                node_status = "更新" if node_id in nodes_to_update else "执行"
                status_msg = f"正在{node_status}{node_id}"

                TaskService.update_task_status(
                    db, task_id, TaskStatus.PROCESSING,
                    progress=progress, message=status_msg
                )

                await send_callback(task_id, i, "processing", status_msg)
                print(f"⚡ {node_status.title()} node {i}/{len(nodes_to_execute)}: {node_id}")

                # Update context with previous results
                context.update(results)

                # 增量模式：如果有之前的结果且节点不需要更新，尝试复用
                if incremental_mode and node_id not in nodes_to_update:
                    previous_result = context.get("modifications", {}).get(node_id)
                    if previous_result:
                        logger.info(f"🔄 复用节点结果: {node_id}")
                        results.update({node_id: previous_result})
                        continue

                # Execute node
                node_result = await node_manager.execute_node(node_id, context)
                results.update(node_result)

                # Extract node outputs to context for downstream nodes
                if node_id in node_result:
                    node_output = node_result[node_id]
                    if isinstance(node_output, dict):
                        # Extract specific output fields based on node type
                        extracted_outputs = extract_node_outputs(node_id, node_output)
                        logger.info(f"🔍 Node {node_id} extracted outputs: {list(extracted_outputs.keys())}")
                        print(f"🔍 Node {node_id} extracted outputs: {list(extracted_outputs.keys())}")
                        context.update(extracted_outputs)
                        logger.info(f"🎯 Context keys after {node_id}: {list(context.keys())}")
                        print(f"🎯 Context keys after {node_id}: {list(context.keys())}")

                # 记录到VGP文档
                node_manager.vgp_protocol.add_node(
                    vgp_document,
                    node_type=node_id,
                    input_data=context.copy(),
                    output_data=node_result.get(node_id, {})
                )

                await send_callback(task_id, i, "completed", f"{node_id}执行完成")
                logger.info(f"✅ Node {node_id} completed")
                print(f"✅ Node {i}/{len(complete_vgp_nodes)} completed: {node_id}")
                
            except Exception as node_error:
                error_msg = f"节点{node_id}执行失败: {str(node_error)}"
                TaskService.update_task_status(
                    db, task_id, TaskStatus.FAILED,
                    error_message=error_msg
                )
                await send_callback(task_id, i, "failed", error_msg)
                logger.error(error_msg)
                print(f"❌ Node {node_id} failed: {node_error}")
                raise

        # VGP分析完成后，生成VGP分析摘要
        vgp_summary = generate_vgp_summary(results)
        print(f"📋 VGP分析摘要: {vgp_summary}")

        # 进度更新：VGP分析完成，开始视频生成
        TaskService.update_task_status(
            db, task_id, TaskStatus.PROCESSING,
            progress=70.0, message="VGP分析完成，开始视频生成"
        )

        # 使用VGP分析结果生成视频 - 完整分镜到视频流程
        try:
            print("🎬 开始使用VGP分析结果生成视频...")

            # 阶段1: 从分镜块生成关键帧图片
            shot_blocks = results.get('shot_block_generation', {}).get('shot_blocks_id', [])
            if not shot_blocks:
                raise ValueError("未找到分镜块数据，无法生成关键帧")

            print(f"📋 找到 {len(shot_blocks)} 个分镜块，开始生成关键帧...")

            # 进度更新
            TaskService.update_task_status(
                db, task_id, TaskStatus.PROCESSING,
                progress=75.0, message=f"生成 {len(shot_blocks)} 个关键帧图片"
            )

            # 生成关键帧
            keyframes = await generate_keyframes_from_shot_blocks(shot_blocks, context)
            print(f"🎨 生成了 {len(keyframes)} 个关键帧图片")

            # 阶段2: 处理帧复用逻辑
            processed_keyframes = process_frame_reuse_logic(keyframes)
            print(f"🔄 处理帧复用后共 {len(processed_keyframes)} 个帧")

            # 阶段3: 使用关键帧生成视频片段
            from video_generate_protocol.nodes.qwen_integration import StoryboardToVideoProcessor

            # 获取千问API密钥
            import os
            qwen_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('AI__DASHSCOPE_API_KEY')
            if not qwen_key:
                raise ValueError("缺少千问API密钥，请设置 DASHSCOPE_API_KEY 环境变量")

            video_processor = StoryboardToVideoProcessor(qwen_key)

            # 进度更新
            TaskService.update_task_status(
                db, task_id, TaskStatus.PROCESSING,
                progress=80.0, message="使用关键帧生成视频片段"
            )

            # 生成视频片段
            video_clips = await video_processor.process_storyboard_frames(
                processed_keyframes,
                f"/tmp/video_clips_{task_id}"
            )
            print(f"🎥 生成了 {len(video_clips)} 个视频片段")

            # 阶段4: 合并视频片段
            if video_clips:
                TaskService.update_task_status(
                    db, task_id, TaskStatus.PROCESSING,
                    progress=90.0, message="使用IMS合并视频片段"
                )

                final_video_path = f"/tmp/final_video_{task_id}.mp4"
                merge_result = await video_processor.merge_clips(video_clips, final_video_path)

                # merge_result可能包含video_url(IMS)或local_path(ffmpeg降级)
                video_output = merge_result.get("video_url") or merge_result.get("local_path")

                results['video_generation'] = {
                    "success": merge_result.get("success", False),
                    "video_url": merge_result.get("video_url"),  # IMS返回的URL
                    "video_path": merge_result.get("local_path"),  # ffmpeg返回的本地路径
                    "job_id": merge_result.get("job_id"),  # IMS任务ID
                    "duration_seconds": int(request.target_duration_id),
                    "generation_mode": "storyboard_to_video_ims" if merge_result.get("video_url") else "storyboard_to_video_ffmpeg",
                    "segments_count": len(video_clips),
                    "keyframes_count": len(processed_keyframes),
                    "metadata": {
                        "vgp_analysis": results,
                        "shot_blocks": shot_blocks,
                        "keyframes": [kf.get('image_url') or kf.get('image_path') for kf in processed_keyframes],
                        "clips": video_clips
                    }
                }
                print(f"🎉 视频生成完成: {len(video_clips)} 个片段合并为最终视频")
                print(f"📍 视频输出: {video_output}")
            else:
                raise Exception("没有成功生成任何视频片段")

        except Exception as video_error:
            print(f"⚠️ 视频生成失败，但VGP分析成功: {video_error}")
            results['video_generation_error'] = str(video_error)

        # 保存VGP文档到项目根目录
        vgp_dir = Path(__file__).parent / "vgp_documents"
        vgp_dir.mkdir(exist_ok=True)
        vgp_file_path = str(vgp_dir / f"{task_id}.vgp.json")

        # 更新VGP文档的最终输出
        vgp_document.final_output = results.get('video_generation', {})

        # 验证并保存VGP文档
        try:
            node_manager.vgp_protocol.save(vgp_document, vgp_file_path)
            logger.info(f"📄 VGP document saved: {vgp_file_path}")
            print(f"📄 VGP document saved: {vgp_file_path}")
        except Exception as e:
            logger.warning(f"Failed to save VGP document: {e}")

        # 序列化结果为可JSON化的格式
        serialized_results = serialize_results(results)
        serialized_results['vgp_document_path'] = vgp_file_path

        # 保存对话结果（如果有对话上下文）
        if request.conversation_context and conversation_result:
            try:
                from conversation.conversation_manager import conversation_manager
                conversation_manager.save_generation_result(
                    request.conversation_context.conversation_id,
                    task_id,
                    serialized_results
                )
                logger.info(f"💬 保存对话生成结果: {request.conversation_context.conversation_id}")
            except Exception as e:
                logger.warning(f"保存对话结果失败: {e}")

        # Task completed successfully
        TaskService.update_task_status(
            db, task_id, TaskStatus.COMPLETED,
            progress=100.0,
            message=f"任务完成：{'增量修改' if context.get('incremental_mode') else 'VGP分析+视频生成'}",
            result=serialized_results
        )
        
        # Final success callback
        await send_callback(task_id, 0, "completed", "视频生成任务完成")
        logger.info(f"🎉 Task {task_id} completed successfully")
        print(f"🎉 Task {task_id} completed successfully!")
        print(f"📊 Final results: {results}")
        
    except Exception as e:
        error_msg = f"任务执行失败: {str(e)}"
        TaskService.update_task_status(
            db, task_id, TaskStatus.FAILED,
            error_message=error_msg
        )
        await send_callback(task_id, 0, "failed", error_msg)
        logger.error(f"❌ Task {task_id} failed: {e}")
        print(f"❌ Task {task_id} failed: {e}")
    finally:
        db.close()


# =============================
# Exception Handlers
# =============================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"❌ Unhandled exception: {exc}")
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error" if not settings.is_development else str(exc),
            "timestamp": datetime.now().isoformat()
        }
    )


# Application events are now handled by the lifespan context manager above


# =============================
# 视频生成辅助函数
# =============================

async def generate_keyframes_from_shot_blocks(shot_blocks: List[Dict], context: Dict) -> List[Dict]:
    """从分镜块生成关键帧图片"""
    try:
        # ✅ 优先检查本地是否有可用的关键帧文件
        local_keyframe_dir = "/Users/luming/Downloads/aura_render/aura_render_temp"
        local_keyframes = []
        use_local_keyframes = False

        if os.path.exists(local_keyframe_dir):
            # 检查是否有关键帧1.png到关键帧N.png
            for i in range(1, len(shot_blocks) + 1):
                keyframe_path = os.path.join(local_keyframe_dir, f"关键帧{i}.png")
                if os.path.exists(keyframe_path):
                    local_keyframes.append(keyframe_path)

            if len(local_keyframes) >= len(shot_blocks):
                use_local_keyframes = True
                print(f"✅ 发现本地关键帧文件 {len(local_keyframes)} 个，将直接使用本地关键帧")

        # 如果有足够的本地关键帧，直接使用
        if use_local_keyframes:
            keyframes = []
            for i, (shot_block, keyframe_path) in enumerate(zip(shot_blocks, local_keyframes[:len(shot_blocks)])):
                keyframe = {
                    "frame_id": f"frame_{i+1:03d}",
                    "segment_id": i,
                    "image_path": keyframe_path,  # 使用本地路径
                    "shot_block": shot_block,
                    "generation_mode": "local_file",  # 标记来源
                    "is_reused": False,
                    "duration": shot_block.get('duration', 3.0)
                }
                keyframes.append(keyframe)
                print(f"✅ 加载本地关键帧 {i+1}/{len(shot_blocks)}: {keyframe_path}")

            print(f"\n{'='*80}")
            print(f"🖼️  使用的本地关键帧 (共{len(keyframes)}个):")
            print(f"{'='*80}")
            for idx, kf in enumerate(keyframes, 1):
                print(f"{idx}. {kf.get('image_path', 'N/A')}")
            print(f"{'='*80}\n")

            return keyframes

        # 没有足够的本地关键帧，需要生成
        print(f"⚠️ 未找到足够的本地关键帧文件，将调用API生成")

        from video_generate_protocol.nodes.image_generation_node import (
            ImageGenerationNode,
            ImageGenerationTask,
            ImageGenerationNodeRequest
        )

        # 初始化图像生成节点
        image_node = ImageGenerationNode({
            "dalle_api_key": os.getenv("OPENAI_API_KEY"),
            "stability_api_key": os.getenv("STABILITY_API_KEY")
        })

        # 提取参考图片信息
        reference_image = None
        reference_media = context.get("reference_media")

        # 适配两种数据结构：
        # 1. 新格式（来自/vgp/generate）: {"product_images": [...], "videos": [...]}
        # 2. 旧格式（来自/generate）: [{media_type: ..., ...}, ...]

        all_media = []
        if reference_media:
            if isinstance(reference_media, dict):
                # 新格式：从字典中提取所有媒体
                if "product_images" in reference_media:
                    product_images = reference_media.get("product_images", [])
                    if isinstance(product_images, list):
                        # 将产品图片添加到媒体列表，并标记类型
                        for img in product_images:
                            if isinstance(img, dict):
                                img_copy = img.copy()
                                img_copy["media_type"] = "product_image"
                                all_media.append(img_copy)

                if "videos" in reference_media:
                    videos = reference_media.get("videos", [])
                    if isinstance(videos, list):
                        for vid in videos:
                            if isinstance(vid, dict):
                                vid_copy = vid.copy()
                                vid_copy["media_type"] = "video"
                                all_media.append(vid_copy)

            elif isinstance(reference_media, list):
                # 旧格式：直接使用列表
                all_media = reference_media

        # 从所有媒体中查找产品图片
        for media in all_media:
            if isinstance(media, dict):
                # 新格式：直接从media中获取url
                if media.get("media_type") == "product_image":
                    reference_image = media.get("url")
                    if reference_image:
                        print(f"✅ 使用产品参考图片: {reference_image}")
                        break

                # 旧格式：从original_input或其他字段获取
                if media.get("media_type") == "product_image" and media.get("processing_status") == "success":
                    if "original_input" in media and "url" in media["original_input"]:
                        reference_image = media["original_input"]["url"]
                    else:
                        reference_image = media.get("original_url") or media.get("local_path")
                    if reference_image:
                        print(f"✅ 使用产品参考图片: {reference_image}")
                        break

        keyframes = []
        tasks = []

        # 为每个分镜块创建图像生成任务
        for i, shot_block in enumerate(shot_blocks):
            visual_desc = shot_block.get('visual_description', '画面内容')
            shot_type = shot_block.get('shot_type', '中景')

            # 构建提示词，结合参考图片信息
            if reference_image:
                prompt = f"{shot_type}镜头：{visual_desc}，参考产品图片的风格和元素"
            else:
                prompt = f"{shot_type}镜头：{visual_desc}，高质量商业视频风格"

            task = ImageGenerationTask(
                prompt=prompt,
                style="realistic",
                quality="high",
                aspect_ratio="3:2",
                reference_image=reference_image,
                metadata={
                    "shot_block_index": i,
                    "shot_type": shot_type,
                    "duration": shot_block.get('duration', 3.0)
                }
            )
            tasks.append(task)

        # 检查是否有可用的API密钥
        dashscope_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("AI__DASHSCOPE_API_KEY")
        openai_key = os.getenv("OPENAI_API_KEY")
        stability_key = os.getenv("STABILITY_API_KEY")

        has_api_key = bool(openai_key or stability_key or dashscope_key)

        if not has_api_key:
            print("⚠️ 未配置图片生成API密钥(DASHSCOPE_API_KEY, OPENAI_API_KEY 或 STABILITY_API_KEY)")
            print("⚠️ 将使用占位图片生成视频")

            # 创建占位图片
            import tempfile
            from PIL import Image, ImageDraw, ImageFont

            for i, shot_block in enumerate(shot_blocks):
                # 创建占位图
                img = Image.new('RGB', (1280, 720), color=(30, 30, 30))
                draw = ImageDraw.Draw(img)

                # 添加文本
                text = f"镜头 {i+1}\n{shot_block.get('shot_type', '中景')}\n{shot_block.get('visual_description', '')[:50]}..."
                draw.text((640, 360), text, fill=(255, 255, 255), anchor='mm')

                # 保存
                temp_path = tempfile.mktemp(suffix='.jpg', prefix='keyframe_')
                img.save(temp_path, quality=85)

                keyframe = {
                    "frame_id": f"frame_{i+1:03d}",
                    "segment_id": i,
                    "image_path": temp_path,
                    "shot_block": shot_block,
                    "generation_mode": "placeholder",
                    "is_reused": False,
                    "duration": shot_block.get('duration', 3.0)
                }
                keyframes.append(keyframe)
                print(f"📝 创建占位关键帧 {i+1}: {temp_path}")
        elif dashscope_key:
            # 使用通义万相生成图片 - 直接保存URL不下载
            print("🎨 使用通义万相(Wanx)生成关键帧图片...")
            try:
                import dashscope
                from dashscope import ImageSynthesis

                dashscope.api_key = dashscope_key

                for i, shot_block in enumerate(shot_blocks):
                    visual_desc = shot_block.get('visual_description', '画面内容')
                    shot_type = shot_block.get('shot_type', '中景')

                    # 构建提示词
                    if reference_image:
                        prompt = f"{shot_type}镜头：{visual_desc}，参考产品图片的风格和元素，高质量商业视频风格"
                    else:
                        prompt = f"{shot_type}镜头：{visual_desc}，高质量商业视频风格"

                    try:
                        # 调用通义万相API
                        rsp = ImageSynthesis.call(
                            model='wanx-v1',
                            prompt=prompt,
                            n=1,
                            size='1280*720'
                        )

                        if rsp.status_code == 200 and rsp.output and rsp.output.results:
                            image_url = rsp.output.results[0].url

                            # 直接保存URL，不下载图片（万相URL可直接用于图生视频）
                            keyframe = {
                                "frame_id": f"frame_{i+1:03d}",
                                "segment_id": i,
                                "image_url": image_url,  # 保存URL而非本地路径
                                "shot_block": shot_block,
                                "generation_mode": "wanx_text_to_image",
                                "is_reused": False,
                                "duration": shot_block.get('duration', 3.0)
                            }
                            keyframes.append(keyframe)
                            print(f"✅ 通义万相生成关键帧 {i+1}/{len(shot_blocks)}: {image_url}")
                        else:
                            print(f"⚠️ 关键帧 {i+1} 生成失败: {rsp.message}")
                    except Exception as e:
                        print(f"❌ 关键帧 {i+1} 生成失败: {e}")

            except Exception as e:
                print(f"❌ 通义万相图片生成失败: {e}")
                logger.error(f"通义万相图片生成失败: {e}", exc_info=True)
        else:
            # 批量生成图片
            request = ImageGenerationNodeRequest(
                tasks=tasks,
                provider_preference="dalle",  # 优先使用DALL-E
                batch_mode=True
            )

            try:
                response = await image_node.process(request)

                # 处理生成结果
                for i, (shot_block, generated_image) in enumerate(zip(shot_blocks, response.generated_images)):
                    if generated_image and generated_image.image_path:
                        keyframe = {
                            "frame_id": f"frame_{i+1:03d}",
                            "segment_id": i,
                            "image_path": generated_image.image_path,
                            "shot_block": shot_block,
                            "generation_mode": "text_to_image",
                            "is_reused": False,
                            "duration": shot_block.get('duration', 3.0)
                        }
                        keyframes.append(keyframe)
                        print(f"✅ 生成关键帧 {i+1}: {generated_image.image_path}")
                    else:
                        print(f"❌ 关键帧 {i+1} 生成失败")
            except Exception as e:
                print(f"❌ 图片生成失败: {e}")
                logger.error(f"图片生成失败: {e}", exc_info=True)

        return keyframes

    except Exception as e:
        print(f"❌ 关键帧生成失败: {e}")
        # 返回空的关键帧列表，或者使用fallback逻辑
        return []


def process_frame_reuse_logic(keyframes: List[Dict]) -> List[Dict]:
    """处理帧复用逻辑 - 尾帧复用为下一段的首帧"""
    if len(keyframes) <= 1:
        return keyframes

    processed_frames = []

    for i in range(len(keyframes)):
        frame = keyframes[i]
        processed_frames.append(frame)

        # 检查是否需要复用当前帧作为下一段的首帧
        # 逻辑：每个奇数位置的帧（即每段的尾帧）复用为下一段的首帧
        if i < len(keyframes) - 1 and i % 2 == 1:
            reused_frame = {
                "frame_id": f"{frame['frame_id']}_reused",
                "segment_id": frame["segment_id"] + 1,
                "shot_block": frame["shot_block"],
                "generation_mode": frame["generation_mode"],
                "is_reused": True,
                "source_frame_id": frame["frame_id"],
                "duration": keyframes[i+1].get('duration', 3.0) if i+1 < len(keyframes) else 3.0
            }

            # 同时支持image_url和image_path（万相用URL，占位图用path）
            if "image_url" in frame:
                reused_frame["image_url"] = frame["image_url"]
            if "image_path" in frame:
                reused_frame["image_path"] = frame["image_path"]

            processed_frames.append(reused_frame)
            print(f"🔄 复用帧: {frame['frame_id']} -> {reused_frame['frame_id']}")

    return processed_frames


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app:app",
        host=settings.api.host,
        port=settings.api.port,
        reload=settings.is_development
    )