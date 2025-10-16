"""
VGP新工作流API - 专用于 vgp_new_pipeline
提供 /vgp/generate 接口
完全复用 /generate 的处理逻辑，只使用不同的节点序列
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends
import logging

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid

# 导入数据库相关
from sqlalchemy.orm import Session
from database import get_db, TaskService, TaskStatus

# 创建路由
vgp_router = APIRouter(prefix="/vgp", tags=["VGP Workflow"])


class ReferenceMedia(BaseModel):
    """参考媒体"""
    product_images: Optional[List[Dict[str, Any]]] = None
    videos: Optional[List[Dict[str, Any]]] = None


class VGPGenerateRequest(BaseModel):
    """VGP新工作流生成请求"""
    # 核心输入
    theme_id: str = Field(..., description="主题ID，如：产品展示、教学视频等")
    user_description_id: str = Field(..., description="用户的详细描述")
    target_duration_id: int = Field(default=30, description="目标时长（秒）", ge=5, le=300)
    keywords_id: List[str] = Field(default_factory=list, description="关键词列表")

    # 参考媒体
    reference_media: Optional[ReferenceMedia] = None

    # 工作流模板（默认使用新工作流）
    template: str = Field(default="vgp_new_pipeline", description="工作流模板名称")

    # 会话信息（可选）
    session_id: Optional[str] = Field(None, description="会话ID，用于关联多次请求")
    user_id: Optional[str] = Field(None, description="用户ID，用于用户行为分析")

    # 任务状态回调字段（用于集成到外部系统）
    tenant_id: Optional[str] = Field(None, description="租户ID，用于多租户系统")
    id: Optional[str] = Field(None, description="业务ID，用于关联业务记录")

    @field_validator('template')
    @classmethod
    def validate_template(cls, v):
        allowed = ['vgp_new_pipeline', 'vgp_full_pipeline', 'basic_video_generation']
        if v not in allowed:
            raise ValueError(f'模板必须是以下之一: {", ".join(allowed)}')
        return v


class VGPGenerateResponse(BaseModel):
    """VGP生成响应"""
    success: bool
    instance_id: str
    task_id: Optional[str] = None
    message: str
    status: str = "submitted"
    estimated_time: Optional[float] = None


class VGPStatusResponse(BaseModel):
    """VGP任务状态响应"""
    instance_id: str
    status: str
    progress: Optional[float] = None
    current_node: Optional[str] = None
    execution_time: Optional[float] = None
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None


# ============== 后台处理函数（使用DAG并行执行引擎）==============
async def process_vgp_video_generation(task_id: str, request: 'VGPGenerateRequest'):
    """
    VGP视频生成后台处理 - 使用DAG并行执行引擎
    支持节点依赖关系和并行执行
    """
    from database.base import SessionLocal
    from pathlib import Path
    from vgp_dag_executor import VGPDAGExecutor
    import time

    # 导入 app.py 中的函数和管理器
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    # 从 app 模块导入需要的函数
    from app import (
        node_manager, send_callback, generate_vgp_summary, serialize_results,
        extract_node_outputs, generate_keyframes_from_shot_blocks, process_frame_reuse_logic
    )

    # 初始化 API 服务（用于状态更新）
    api_service = None
    tenant_id = request.tenant_id
    business_id = request.id
    if tenant_id:
        try:
            from api_service.api_service import APIService
            api_service = APIService()
            logger.info(f"✅ [VGP] API服务初始化成功 (tenant_id={tenant_id}, business_id={business_id})")
        except Exception as e:
            logger.info(f"⚠️ [VGP] API服务初始化失败: {e}")

    db = SessionLocal()
    try:
        # 1️⃣ 任务开始 - 更新状态为运行中 (status="0")
        if api_service and tenant_id:
            try:
                api_service.update_task_status(task_id, "0", tenant_id, business_id=business_id)
                logger.info(f"✅ [VGP] 任务状态更新为运行中: task_id={task_id}")
            except Exception as status_error:
                logger.info(f"⚠️ [VGP] 更新运行状态时出错: {status_error}")

        # 更新数据库任务状态为处理中
        TaskService.update_task_status(
            db, task_id, TaskStatus.PROCESSING,
            progress=0.0, message="开始VGP视频生成任务"
        )

        logger.info(f"🚀 [VGP] Starting background processing for task {task_id}")

        # 构建上下文（与老接口一致）
        context = {
            "theme_id": request.theme_id,
            "keywords_id": request.keywords_id,
            "target_duration_id": request.target_duration_id,
            "user_description_id": request.user_description_id,
            "reference_media": request.reference_media.dict() if request.reference_media else {}
        }

        logger.info(f"🎯 [VGP] Processing request: {request.theme_id} - {request.target_duration_id}s")

        # 创建VGP文档
        vgp_document = node_manager.vgp_protocol.create_document({
            'task_id': task_id,
            'theme': request.theme_id,
            'keywords': request.keywords_id,
            'duration': request.target_duration_id,
            'description': request.user_description_id
        })
        vgp_document.task_id = task_id

        # ✨ 使用DAG执行引擎（支持并行和依赖关系）
        dag_executor = VGPDAGExecutor()

        # 打印DAG结构
        logger.info(f"\n" + dag_executor.visualize_dag())
        logger.info(f"")

        results = {}
        completed_count = 0

        # 定义节点执行器函数
        async def execute_single_node(node_name: str, exec_context: dict) -> dict:
            """执行单个节点并返回结果"""
            nonlocal completed_count

            try:
                # 执行节点
                node_result = await node_manager.execute_node(node_name, exec_context)

                # 提取输出
                if node_name in node_result:
                    node_output = node_result[node_name]
                    if isinstance(node_output, dict):
                        extracted_outputs = extract_node_outputs(node_name, node_output)
                        node_result.update(extracted_outputs)
                        logger.info(f"🔍 [VGP] Node {node_name} outputs: {list(extracted_outputs.keys())}")

                # 记录到VGP文档
                node_manager.vgp_protocol.add_node(
                    vgp_document,
                    node_type=node_name,
                    input_data=exec_context.copy(),
                    output_data=node_result.get(node_name, {})
                )

                return node_result

            except Exception as e:
                logger.info(f"❌ [VGP] Node {node_name} execution failed: {e}")
                raise

        # 定义进度回调
        async def on_node_progress(node_id: int, status: str, message: str):
            """节点进度回调"""
            nonlocal completed_count

            if status == 'completed':
                completed_count += 1

            progress = (completed_count / 16) * 100
            status_msg = f"DAG进度: {completed_count}/16 节点 - {message}"

            TaskService.update_task_status(
                db, task_id, TaskStatus.PROCESSING,
                progress=progress, message=status_msg
            )

            await send_callback(task_id, node_id, status, message)

        # 执行DAG工作流
        logger.info(f"🚀 [VGP] Starting DAG execution with parallel nodes...")
        node_results = await dag_executor.execute_dag(
            node_executor=execute_single_node,
            context=context,
            on_progress=on_node_progress
        )

        # 合并所有节点结果
        for node_id, node_result in node_results.items():
            if isinstance(node_result, dict):
                results.update(node_result)

        logger.info(f"📊 [VGP] DAG execution summary:")
        summary = dag_executor.get_execution_summary()
        for key, value in summary.items():
            logger.info(f"   {key}: {value}")

        # 生成VGP摘要
        vgp_summary = generate_vgp_summary(results)
        logger.info(f"📋 [VGP] Analysis summary: {vgp_summary}")

        # ✨ 视频生成现在在 Node 5 (asset_request) 中完成
        # Node 5 会生成 video_clips 并传递给后续节点
        # Node 16 (timeline_integration) 会进行最终合成
        logger.info(f"📊 [VGP] Video generation completed in Node 5, final composition in Node 16")

        # 从 Node 5 的输出中获取视频生成结果
        asset_request_result = results.get('asset_request', {})
        video_clips = asset_request_result.get('video_clips', [])
        video_generation_success = asset_request_result.get('video_generation_success', False)

        logger.info(f"🎥 [VGP] Node 5 generated {len(video_clips)} video clips")
        logger.info(f"✅ [VGP] Video generation status: {'Success' if video_generation_success else 'Failed'}")

        # 从 Node 16 的输出中获取最终合成结果
        timeline_result = results.get('timeline_integration', {})
        final_video_url = timeline_result.get('final_video_url')
        final_video_path = timeline_result.get('final_video_path')

        if final_video_url or final_video_path:
            results['video_generation'] = {
                "success": True,
                "video_url": final_video_url,
                "video_path": final_video_path,
                "duration_seconds": int(request.target_duration_id),
                "generation_mode": "vgp_new_pipeline",
                "segments_count": len(video_clips),
            }
            logger.info(f"🎉 [VGP] Final video composition completed")
        else:
            results['video_generation_error'] = "No final video generated in Node 16"

        # 保存VGP文档
        vgp_dir = Path(__file__).parent / "vgp_documents"
        vgp_dir.mkdir(exist_ok=True)
        vgp_file_path = str(vgp_dir / f"{task_id}.vgp.json")

        vgp_document.final_output = results.get('video_generation', {})

        try:
            node_manager.vgp_protocol.save(vgp_document, vgp_file_path)
            logger.info(f"📄 [VGP] Document saved: {vgp_file_path}")
        except Exception as e:
            logger.info(f"⚠️ [VGP] Failed to save document: {e}")

        # 序列化结果
        serialized_results = serialize_results(results)
        serialized_results['vgp_document_path'] = vgp_file_path

        # 2️⃣ 任务完成 - 先创建资源，再更新状态为完成 (status="1")
        if api_service and tenant_id:
            try:
                resource_id = None
                # 第一步：如果有视频URL，先创建资源记录
                if final_video_url:
                    try:
                        resource_result = api_service.create_resource(
                            resource_type=1,  # 1=视频类型
                            name=f"VGP视频-{request.theme_id}",
                            path=final_video_url,
                            local_full_path="",
                            file_type="mp4",
                            size=0,
                            tenant_id=tenant_id
                        )
                        if resource_result:
                            resource_id = resource_result.get('resource_id')
                        logger.info(f"✅ [VGP] 资源创建成功: {final_video_url}, resource_id={resource_id}")
                    except Exception as resource_error:
                        logger.info(f"⚠️ [VGP] 创建资源记录时出错: {resource_error}")

                # 第二步：更新任务状态为完成，传入 resource_id
                api_service.update_task_status(task_id, "1", tenant_id,
                                               business_id=business_id,
                                               resource_id=resource_id)
                logger.info(f"✅ [VGP] 任务状态更新为完成: task_id={task_id}")

            except Exception as status_error:
                logger.info(f"⚠️ [VGP] 更新完成状态时出错: {status_error}")

        # 更新数据库任务状态为完成
        TaskService.update_task_status(
            db, task_id, TaskStatus.COMPLETED,
            progress=100.0,
            message="VGP视频生成完成",
            result=serialized_results
        )

        await send_callback(task_id, 0, "completed", "VGP视频生成任务完成")
        logger.info(f"🎉 [VGP] Task {task_id} completed successfully!")

    except Exception as e:
        # 3️⃣ 任务失败 - 更新状态为失败 (status="2")
        if api_service and tenant_id:
            try:
                api_service.update_task_status(task_id, "2", tenant_id, business_id=business_id)
                logger.info(f"✅ [VGP] 任务状态更新为失败: task_id={task_id}")
            except Exception as status_error:
                logger.info(f"⚠️ [VGP] 更新失败状态时出错: {status_error}")

        error_msg = f"VGP任务执行失败: {str(e)}"
        TaskService.update_task_status(
            db, task_id, TaskStatus.FAILED,
            error_message=error_msg
        )
        await send_callback(task_id, 0, "failed", error_msg)
        logger.info(f"❌ [VGP] Task {task_id} failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


@vgp_router.post("/generate", response_model=VGPGenerateResponse)
async def generate_video(
    request: VGPGenerateRequest,
    background_tasks: BackgroundTasks,
    db: Session = Depends(get_db)
):
    """
    VGP新工作流视频生成 - 完全复用 /generate 的逻辑

    与 /generate 的唯一区别：执行不同的节点序列（VGP新工作流16节点）
    处理流程完全相同：创建任务 → 立即返回 → 后台执行
    """
    try:
        # 步骤1: 创建数据库任务（与 /generate 完全相同）
        task = TaskService.create_task(
            db=db,
            theme=request.theme_id,
            keywords=request.keywords_id,
            target_duration=request.target_duration_id,
            user_description=request.user_description_id
        )

        logger.info(f"🚀 [VGP] Starting video generation task: {task.task_id}")

        # 步骤2: 添加后台任务处理（与 /generate 完全相同）
        background_tasks.add_task(
            process_vgp_video_generation,
            task_id=task.task_id,
            request=request
        )

        # 步骤3: 立即返回响应（与 /generate 完全相同的模式）
        return VGPGenerateResponse(
            success=True,
            instance_id=task.task_id,
            task_id=task.task_id,
            message=f"VGP视频生成任务已启动（模板: {request.template}）",
            status="started",
            estimated_time=request.target_duration_id * 2
        )

    except Exception as e:
        logger.info(f"❌ [VGP] Failed to create task: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create VGP task: {str(e)}")


@vgp_router.get("/templates")
async def get_available_templates():
    """获取可用的工作流模板 - 返回静态配置"""
    return {
        "templates": [
            "vgp_new_pipeline",
            "vgp_full_pipeline",
            "basic_video_generation"
        ],
        "recommended": "vgp_new_pipeline",
        "description": {
            "vgp_new_pipeline": "新版VGP工作流，优化的16节点架构，素材生成集中化",
            "vgp_full_pipeline": "旧版VGP工作流，保留用于兼容（暂不可用）",
            "basic_video_generation": "基础视频生成（使用/generate接口）"
        }
    }


@vgp_router.get("/system/health")
async def health_check(db: Session = Depends(get_db)):
    """健康检查 - 检查数据库和节点管理器"""
    try:
        # 检查数据库连接
        from database.models import Task
        db.query(Task).first()

        # 检查节点管理器
        from app import node_manager

        return {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_info": {
                "database": "connected",
                "node_manager": "available" if node_manager else "unavailable",
                "vgp_nodes_count": 16,
                "api_version": "1.0.0"
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# 如果直接运行此文件，启动FastAPI服务器
if __name__ == "__main__":
    import uvicorn
    from fastapi import FastAPI

    app = FastAPI(title="VGP新工作流API", version="1.0.0")
    app.include_router(vgp_router)

    logger.info(f"="*60)
    logger.info(f"🎬 VGP新工作流API服务")
    logger.info(f"="*60)
    logger.info(f"接口地址: http://localhost:8000")
    logger.info(f"文档地址: http://localhost:8000/docs")
    logger.info(f"="*60)

    uvicorn.run(app, host="0.0.0.0", port=8000)
