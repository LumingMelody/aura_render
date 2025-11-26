"""
VGP新工作流API - 专用于 vgp_new_pipeline
提供 /vgp/generate 接口
完全复用 /generate 的处理逻辑，只使用不同的节点序列
"""
from fastapi import APIRouter, HTTPException, BackgroundTasks, Depends, Request
import logging

logger = logging.getLogger(__name__)

from pydantic import BaseModel, Field, field_validator
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import json

# 导入数据库相关
from sqlalchemy.orm import Session
from database import get_db, TaskService, TaskStatus

# 导入Qwen LLM用于意图识别
from llm.qwen import QwenLLM

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

    # ✨ 新增：控制是否启用Coze图片搜索
    enable_coze_search: bool = Field(
        default=True,
        description="是否在未提供product_images时自动调用Coze搜索图片（True=启用，False=禁用）"
    )

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


# ============== AI意图识别辅助函数 ==============

def needs_ai_analysis(keywords: List[str]) -> bool:
    """
    判断是否需要AI分析用户意图

    Args:
        keywords: 关键词列表

    Returns:
        True if需要AI分析，False otherwise
    """
    if not keywords or len(keywords) == 0:
        return True

    # 明显的形容词列表（如果第一个关键词是这些，说明可能没有正确提取产品名）
    adjectives = {
        '4K', '高清', '智能', '便携', '高端', '专业', '创新', '科技',
        '时尚', '现代', '精致', '轻薄', '强大', '优质', '先进', '卓越',
        '产品', '展示', '视频', '广告', '宣传', '介绍'
    }

    # 第一个关键词是形容词 → 需要AI
    if keywords[0] in adjectives:
        logger.info(f"🤖 [AI意图识别] 第一个关键词'{keywords[0]}'是形容词，需要AI分析")
        return True

    # 关键词太少 → 需要AI
    if len(keywords) < 2:
        logger.info(f"🤖 [AI意图识别] 关键词太少({len(keywords)}个)，需要AI分析")
        return True

    return False


async def analyze_user_intent(
    user_description: str,
    keywords: List[str],
    qwen_client: Optional[QwenLLM] = None
) -> Optional[Dict[str, Any]]:
    """
    使用Qwen AI分析用户意图并提取结构化信息

    Args:
        user_description: 用户的完整描述
        keywords: 初步提取的关键词列表
        qwen_client: Qwen客户端实例（可选）

    Returns:
        包含product_name, product_attributes等的字典，失败返回None
    """
    try:
        # 创建Qwen客户端（如果没有提供）
        if qwen_client is None:
            qwen_client = QwenLLM(model_name="qwen-max", timeout=30)

        prompt = f"""你是一个视频生成需求分析专家。用户想要生成产品展示视频，请从描述中精准提取关键信息。

用户描述：{user_description}
初步关键词：{', '.join(keywords)}

请以JSON格式返回（必须是纯JSON，不要包含任何markdown标记）：
{{
  "product_name": "核心产品名称（如：投影仪、手机、耳机、音箱）",
  "product_attributes": ["产品特性关键词列表，如：4K、高清、智能、便携"],
  "video_style": "视频风格（如：科技感、温馨、动感、专业）",
  "key_selling_points": ["核心卖点列表"]
}}

关键要求：
1. product_name必须是具体的产品名词（投影仪、手机等），不能是形容词
2. 如果描述中没有明确产品名，设置为null
3. product_attributes是修饰产品的特性（4K、智能、高清等）
4. 只返回JSON，不要任何额外文字

示例：
输入："帮我生成一个10秒的产品展示视频，突出智能投影仪的4K高清特点"
输出：{{"product_name": "投影仪", "product_attributes": ["智能", "4K", "高清"], "video_style": "科技感", "key_selling_points": ["4K高清显示", "智能功能"]}}"""

        logger.info(f"🤖 [AI意图识别] 开始调用Qwen分析...")

        # 调用Qwen（同步调用，因为QwenLLM.generate是同步的）
        response = qwen_client.generate(
            prompt=prompt,
            max_tokens=500,
            temperature=0.1  # 低温度，确保输出稳定
        )

        if not response:
            logger.warning(f"🤖 [AI意图识别] Qwen返回空响应")
            return None

        logger.info(f"🤖 [AI意图识别] Qwen原始响应: {response[:200]}...")

        # 解析JSON
        # 清理可能的markdown标记
        response_text = response.strip()
        if response_text.startswith('```json'):
            response_text = response_text[7:]
        if response_text.startswith('```'):
            response_text = response_text[3:]
        if response_text.endswith('```'):
            response_text = response_text[:-3]
        response_text = response_text.strip()

        intent_data = json.loads(response_text)

        logger.info(f"🤖 [AI意图识别] ✅ 解析成功: {intent_data}")
        return intent_data

    except json.JSONDecodeError as e:
        logger.error(f"🤖 [AI意图识别] ❌ JSON解析失败: {e}, 响应: {response[:200]}")
        return None
    except Exception as e:
        logger.error(f"🤖 [AI意图识别] ❌ 分析失败: {e}")
        return None


# ============== 后台处理函数（使用DAG并行执行引擎）==============
async def process_vgp_video_generation(task_id: str, request: 'VGPGenerateRequest', conversation_context: dict = None):
    """
    VGP视频生成后台处理 - 使用DAG并行执行引擎
    支持节点依赖关系和并行执行
    支持对话式增量修改
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

    # ✅ 初始化素材库客户端（用于BGM和视频素材匹配）
    if tenant_id:
        try:
            from materials_supplies.material_library_client import init_material_library_client
            import os

            # 从环境变量获取Authorization (如果没有设置，使用空字符串)
            auth_token = os.getenv("MATERIAL_LIBRARY_AUTH", "")

            init_material_library_client(tenant_id, auth_token)
            logger.info(f"✅ [VGP] 素材库客户端初始化成功 (tenant_id={tenant_id})")
        except Exception as e:
            logger.warning(f"⚠️ [VGP] 素材库客户端初始化失败: {e}")

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

        # ✨ 新增：如果没有产品图片，且用户启用了Coze搜索，则调用 Coze 搜索图片
        has_product_images = False
        if request.reference_media and request.reference_media.product_images:
            has_product_images = len(request.reference_media.product_images) > 0

        # 调试日志
        logger.info(f"🔍 [VGP] 检查产品图片: reference_media={request.reference_media is not None}, has_product_images={has_product_images}")
        logger.info(f"🔍 [VGP] Coze搜索开关: enable_coze_search={request.enable_coze_search}")

        if not has_product_images and request.enable_coze_search:
            logger.info("🔍 [VGP] 未检测到产品图片，且已启用Coze搜索，开始从 Coze 搜索图片...")

            try:
                from core.cliptemplate.coze.image_search import search_reference_image_from_coze

                # 使用 user_description 作为搜索查询
                image_url = await search_reference_image_from_coze(request.user_description_id)

                if image_url:
                    logger.info(f"✅ [VGP] Coze 搜索到图片: {image_url}")

                    # 添加到 context，格式为 {"product_images": [{"url": "..."}]}
                    context["reference_media"] = {
                        "product_images": [{"url": image_url}]
                    }

                    logger.info(f"🎯 [VGP] 已添加 Coze 搜索图片到 reference_media")
                else:
                    logger.warning("⚠️ [VGP] Coze 未搜索到图片")

            except Exception as e:
                logger.error(f"❌ [VGP] Coze 图片搜索失败: {e}")
                # 继续流程，不阻断

        elif not has_product_images and not request.enable_coze_search:
            logger.info("ℹ️ [VGP] 未检测到产品图片，但用户已禁用Coze搜索，跳过图片搜索")
        elif has_product_images:
            logger.info(f"✅ [VGP] 已有产品图片 ({len(request.reference_media.product_images)} 张)，跳过Coze搜索")

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
            result=serialized_results,
            output_url=final_video_url  # ✅ 将视频URL保存到output_url字段
        )

        # 💬 如果有会话上下文，保存生成结果到对话管理器
        if request.session_id:
            try:
                from conversation.conversation_manager import conversation_manager

                # 保存生成结果
                conversation_manager.save_generation_result(
                    conversation_id=request.session_id,
                    task_id=task_id,
                    result={
                        "video_url": final_video_url,
                        "video_path": final_video_path,
                        "theme": request.theme_id,
                        "duration": request.target_duration_id,
                        "keywords": request.keywords_id,
                        "vgp_summary": vgp_summary,
                        "serialized_results": serialized_results
                    }
                )

                logger.info(f"💬 [VGP] Generation result saved to conversation: {request.session_id}")

            except Exception as conv_error:
                logger.warning(f"⚠️ [VGP] Failed to save to conversation manager: {conv_error}")

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
    VGP新工作流视频生成 - 支持对话式增量修改

    与 /generate 的区别：
    1. 执行不同的节点序列（VGP新工作流16节点）
    2. 支持通过 session_id 实现对话式视频编辑
    """
    try:
        # 步骤0: 如果提供了session_id，进行对话分析
        conversation_context = None
        if request.session_id:
            from conversation.conversation_manager import conversation_manager

            # 生成消息ID
            import uuid
            message_id = str(uuid.uuid4())

            # 检查是否是同一会话的后续请求（判断是否是编辑）
            previous_generation = None
            try:
                previous_generation = conversation_manager.history_manager.get_previous_generation(
                    request.session_id
                )
            except:
                pass

            is_regeneration = previous_generation is not None

            # 处理对话请求
            conversation_context = await conversation_manager.process_conversation_request(
                request_data=request.dict(),
                conversation_context={
                    "conversation_id": request.session_id,
                    "message_id": message_id,
                    "is_regeneration": is_regeneration
                }
            )

            logger.info(f"📝 [VGP] Conversation context: {conversation_context}")

        # 步骤0.5: AI意图识别兜底（如果前端提取的关键词有问题）
        if needs_ai_analysis(request.keywords_id):
            logger.info(f"🤖 [VGP] 触发AI意图识别 - 原始关键词: {request.keywords_id}")

            intent_result = await analyze_user_intent(
                user_description=request.user_description_id,
                keywords=request.keywords_id
            )

            if intent_result and intent_result.get('product_name'):
                # 重新组织关键词：产品名在前，属性在后
                optimized_keywords = [intent_result['product_name']]
                if intent_result.get('product_attributes'):
                    optimized_keywords.extend(intent_result['product_attributes'])

                # 去重
                optimized_keywords = list(dict.fromkeys(optimized_keywords))

                logger.info(f"🤖 [VGP] ✅ AI优化关键词: {request.keywords_id} → {optimized_keywords}")
                request.keywords_id = optimized_keywords

                # 可选：也可以优化theme
                if intent_result.get('video_style'):
                    logger.info(f"🤖 [VGP] 视频风格建议: {intent_result['video_style']}")
            else:
                logger.warning(f"🤖 [VGP] ⚠️ AI意图识别未返回有效结果，使用原始关键词")
        else:
            logger.info(f"✅ [VGP] 关键词提取正常，跳过AI分析: {request.keywords_id}")

        # 步骤1: 创建数据库任务
        task = TaskService.create_task(
            db=db,
            theme=request.theme_id,
            keywords=request.keywords_id,
            target_duration=request.target_duration_id,
            user_description=request.user_description_id
        )

        logger.info(f"🚀 [VGP] Starting video generation task: {task.task_id}")

        # 将 conversation_context 传递给后台任务
        if conversation_context:
            # 暂存到任务元数据或其他地方
            # 这里简单起见，我们将在后台处理函数中重新获取
            pass

        # 步骤2: 添加后台任务处理
        background_tasks.add_task(
            process_vgp_video_generation,
            task_id=task.task_id,
            request=request,
            conversation_context=conversation_context
        )

        # 步骤3: 立即返回响应
        return VGPGenerateResponse(
            success=True,
            instance_id=task.task_id,
            task_id=task.task_id,
            message=f"VGP视频生成任务已启动（模板: {request.template}）" +
                    (f"，会话ID: {request.session_id}" if request.session_id else ""),
            status="started",
            estimated_time=request.target_duration_id * 2
        )

    except Exception as e:
        logger.error(f"❌ [VGP] Failed to create task: {e}")
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


# ============== 对话管理相关API ==============

@vgp_router.get("/conversation/{session_id}/context")
async def get_conversation_context(session_id: str):
    """
    获取对话上下文信息

    Args:
        session_id: 会话ID

    Returns:
        对话的上下文信息，包括消息数量、生成数量等
    """
    try:
        from conversation.conversation_manager import conversation_manager

        context = conversation_manager.history_manager.get_conversation_context(session_id)

        return {
            "success": True,
            "context": context,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ [VGP] Failed to get conversation context: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation context: {str(e)}")


@vgp_router.get("/conversation/{session_id}/history")
async def get_conversation_history(session_id: str):
    """
    获取对话历史

    Args:
        session_id: 会话ID

    Returns:
        完整的对话历史，包括所有消息和生成结果
    """
    try:
        from conversation.conversation_manager import conversation_manager

        conversation = conversation_manager.history_manager.get_or_create_conversation(session_id)

        return {
            "success": True,
            "conversation_id": session_id,
            "messages": conversation.messages,
            "generation_history": conversation.generation_history,
            "current_context": conversation.current_context,
            "created_at": conversation.created_at.isoformat(),
            "updated_at": conversation.updated_at.isoformat(),
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"❌ [VGP] Failed to get conversation history: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get conversation history: {str(e)}")


@vgp_router.get("/conversation/{session_id}/latest-video")
async def get_latest_video(session_id: str):
    """
    获取会话中最新生成的视频

    Args:
        session_id: 会话ID

    Returns:
        最新的视频生成结果
    """
    try:
        from conversation.conversation_manager import conversation_manager

        previous_generation = conversation_manager.history_manager.get_previous_generation(session_id)

        if not previous_generation:
            raise HTTPException(status_code=404, detail="No video found for this session")

        return {
            "success": True,
            "video": previous_generation,
            "timestamp": datetime.now().isoformat()
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [VGP] Failed to get latest video: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get latest video: {str(e)}")


# ============== Timeline提交到IMS ==============

class IMSClip(BaseModel):
    """IMS视频片段"""
    MediaURL: str
    TimelineIn: float
    TimelineOut: float
    In: float
    Out: float
    Volume: float = 100.0
    Speed: float = 1.0
    Effects: Optional[List[Dict[str, Any]]] = None


class IMSVideoTrack(BaseModel):
    """IMS视频轨道"""
    VideoTrackClips: List[IMSClip]


class IMSSubmitRequest(BaseModel):
    """IMS提交请求"""
    VideoTracks: List[IMSVideoTrack]
    OutputConfig: Optional[Dict[str, Any]] = None


class IMSSubmitResponse(BaseModel):
    """IMS提交响应"""
    success: bool
    message: str
    job_id: Optional[str] = None
    timeline_data: Optional[Dict[str, Any]] = None


@vgp_router.post("/submit", response_model=IMSSubmitResponse)
async def submit_timeline_to_ims(request: IMSSubmitRequest):
    """
    提交已编辑的Timeline到阿里云IMS进行云端剪辑

    这个接口接收前端视频编辑器编辑好的Timeline数据（IMS格式），
    然后调用阿里云IMS API提交剪辑任务。

    Args:
        request: IMS格式的Timeline数据

    Returns:
        提交结果，包含任务ID和状态
    """
    try:
        logger.info("=" * 80)
        logger.info("[VGP] 收到IMS Timeline提交请求")
        logger.info("=" * 80)

        # 验证数据
        if not request.VideoTracks:
            raise HTTPException(status_code=400, detail="VideoTracks不能为空")

        # 打印接收到的数据
        logger.info(f"[VGP] 视频轨道数量: {len(request.VideoTracks)}")
        for i, track in enumerate(request.VideoTracks):
            logger.info(f"\n[VGP] 轨道 {i + 1}:")
            logger.info(f"  片段数量: {len(track.VideoTrackClips)}")
            for j, clip in enumerate(track.VideoTrackClips):
                logger.info(f"\n  片段 {j + 1}:")
                logger.info(f"    MediaURL: {clip.MediaURL}")
                logger.info(f"    时间轴: {clip.TimelineIn}s - {clip.TimelineOut}s")
                logger.info(f"    素材裁剪: In={clip.In}s, Out={clip.Out}s")
                logger.info(f"    音量: {clip.Volume}")
                logger.info(f"    速���: {clip.Speed}")
                if clip.Effects:
                    logger.info(f"    特效数量: {len(clip.Effects)}")

        # 调用阿里云IMS API进行云端剪辑
        logger.info("\n" + "=" * 80)
        logger.info("[VGP] 开始提交到阿里云IMS")
        logger.info("=" * 80)

        import os
        import json
        from alibabacloud_ice20201109 import client as ice_client, models as ice_models
        from alibabacloud_tea_openapi import models as open_api_models

        # 检查是否配置了阿里云凭证
        access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")

        if not access_key_id or not access_key_secret:
            logger.warning("[VGP] ⚠️ 未配置阿里云凭证，使用模拟模式")
            # 模拟模式
            response = IMSSubmitResponse(
                success=True,
                message="Timeline已成功提交到IMS（模拟模式 - 未配置阿里云凭证）",
                job_id=f"mock_ims_job_{uuid.uuid4().hex[:8]}",
                timeline_data=request.dict()
            )
            logger.info(f"\n✅ [VGP] 模拟提交成功，任务ID: {response.job_id}\n")
            return response

        # 初始化IMS客户端
        config = open_api_models.Config(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            region_id='cn-shanghai',
            endpoint='ice.cn-shanghai.aliyuncs.com'
        )
        ims_client = ice_client.Client(config)

        # 构建Timeline（将Pydantic模型转换为字典）
        timeline = {
            "VideoTracks": [
                {
                    "VideoTrackClips": [
                        {
                            "MediaURL": clip.MediaURL,
                            "TimelineIn": clip.TimelineIn,
                            "TimelineOut": clip.TimelineOut,
                            "In": clip.In,
                            "Out": clip.Out,
                            "Volume": clip.Volume,
                            "Speed": clip.Speed,
                            "Effects": clip.Effects if clip.Effects else []
                        }
                        for clip in track.VideoTrackClips
                    ]
                }
                for track in request.VideoTracks
            ]
        }

        # 构建输出配置
        if request.OutputConfig:
            output_config = request.OutputConfig
        else:
            # 默认输出配置
            import time
            timestamp = int(time.time())
            output_config = {
                "MediaURL": f"https://ai-movie-cloud-v2.oss-cn-shanghai.aliyuncs.com/edited_videos/video_{timestamp}.mp4",
                "Width": 1280,
                "Height": 720
            }

        logger.info(f"[VGP] Timeline: {json.dumps(timeline, indent=2, ensure_ascii=False)}")
        logger.info(f"[VGP] OutputConfig: {json.dumps(output_config, indent=2, ensure_ascii=False)}")

        # 提交剪辑任务
        submit_request = ice_models.SubmitMediaProducingJobRequest(
            timeline=json.dumps(timeline, ensure_ascii=False),
            output_media_config=json.dumps(output_config, ensure_ascii=False)
        )

        submit_response = ims_client.submit_media_producing_job(submit_request)

        if submit_response.status_code == 200:
            job_id = submit_response.body.job_id
            logger.info(f"✅ [VGP] IMS任务已提交成功")
            logger.info(f"   JobId: {job_id}")
            logger.info(f"   输出URL: {output_config.get('MediaURL')}")

            response = IMSSubmitResponse(
                success=True,
                message=f"Timeline已成功提交到阿里云IMS，任务ID: {job_id}",
                job_id=job_id,
                timeline_data={
                    "timeline": timeline,
                    "output_config": output_config
                }
            )

            logger.info(f"\n✅ [VGP] 提交成功，任务ID: {response.job_id}\n")
            return response
        else:
            raise Exception(f"IMS API返回错误: status_code={submit_response.status_code}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [VGP] 提交失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"提交到IMS失败: {str(e)}"
        )


class IMSJobStatusResponse(BaseModel):
    """IMS任务状态响应"""
    success: bool
    job_id: str
    status: str  # Init, Running, Success, Failed
    message: Optional[str] = None
    video_url: Optional[str] = None
    progress: Optional[float] = None


class VideoUploadResponse(BaseModel):
    """视频上传响应"""
    success: bool
    url: str
    message: str


@vgp_router.post("/upload-video", response_model=VideoUploadResponse)
async def upload_video_to_oss(request: Request):
    """
    上传视频到OSS

    接收前端发送的视频文件，上传到阿里云OSS，返回公网URL

    Args:
        request: FastAPI Request对象，从中读取原始二进制数据

    Returns:
        上传后的OSS公网URL
    """
    try:
        import tempfile
        import os
        from pathlib import Path
        from utils.oss_uploader import get_oss_uploader

        logger.info("[VGP] 收到视频上传请求")

        # 读取原始请求体（二进制数据）
        file_data = await request.body()

        if not file_data:
            raise HTTPException(status_code=400, detail="未收到视频文件")

        # 创建临时文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as temp_file:
            temp_file.write(file_data)
            temp_path = temp_file.name

        try:
            # 上传到OSS
            logger.info(f"[VGP] 正在上传视频到OSS... (大小: {len(file_data) / 1024 / 1024:.2f} MB)")

            uploader = get_oss_uploader()
            oss_url = uploader.upload_video(temp_path)

            logger.info(f"[VGP] ✅ 视频上传成功: {oss_url}")

            return VideoUploadResponse(
                success=True,
                url=oss_url,
                message="视频上传成功"
            )

        finally:
            # 删除临时文件
            if os.path.exists(temp_path):
                os.remove(temp_path)

    except Exception as e:
        logger.error(f"❌ [VGP] 视频上传失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"视频上传失败: {str(e)}"
        )


@vgp_router.get("/submit/{job_id}/status", response_model=IMSJobStatusResponse)
async def get_ims_job_status(job_id: str):
    """
    查询IMS剪辑任务的状态

    Args:
        job_id: IMS任务ID

    Returns:
        任务状态信息
    """
    try:
        logger.info(f"[VGP] 查询IMS任务状态: {job_id}")

        import os
        from alibabacloud_ice20201109 import client as ice_client, models as ice_models
        from alibabacloud_tea_openapi import models as open_api_models

        # 检查是否是模拟任务
        if job_id.startswith("mock_"):
            return IMSJobStatusResponse(
                success=True,
                job_id=job_id,
                status="Success",
                message="模拟任务（未实际提交到IMS）",
                video_url="https://example.com/mock_video.mp4",
                progress=100.0
            )

        # 检查是否配置了阿里云凭证
        access_key_id = os.getenv("OSS_ACCESS_KEY_ID")
        access_key_secret = os.getenv("OSS_ACCESS_KEY_SECRET")

        if not access_key_id or not access_key_secret:
            raise HTTPException(
                status_code=400,
                detail="未配置阿里云凭证，无法查询任务状态"
            )

        # 初始化IMS客户端
        config = open_api_models.Config(
            access_key_id=access_key_id,
            access_key_secret=access_key_secret,
            region_id='cn-shanghai',
            endpoint='ice.cn-shanghai.aliyuncs.com'
        )
        ims_client = ice_client.Client(config)

        # 查询任务状态
        request = ice_models.GetMediaProducingJobRequest(job_id=job_id)
        response = ims_client.get_media_producing_job(request)

        if response.status_code == 200:
            job = response.body.media_producing_job
            status = job.status

            # 计算进度
            progress = 0.0
            if status == "Init":
                progress = 10.0
            elif status == "Running":
                progress = 50.0
            elif status == "Success":
                progress = 100.0
            elif status == "Failed":
                progress = 0.0

            result = IMSJobStatusResponse(
                success=True,
                job_id=job_id,
                status=status,
                message=getattr(job, 'message', None),
                video_url=getattr(job, 'media_url', None) if status == "Success" else None,
                progress=progress
            )

            logger.info(f"[VGP] 任务状态: {status}, 进度: {progress}%")
            return result
        else:
            raise Exception(f"查询任务状态失败: status_code={response.status_code}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [VGP] 查询任务状态失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"查询任务状态失败: {str(e)}"
        )


# ============== 图片上传接口 ==============

@vgp_router.post("/upload-image", summary="上传图片到OSS")
async def upload_image(
    request: Request,
    db: Session = Depends(get_db)
):
    """
    上传图片到阿里云OSS

    接收表单数据中的图片文件，上传到OSS并返回公网URL
    """
    import os
    from fastapi import UploadFile, File, Form
    from utils.oss_uploader import get_oss_uploader
    import tempfile

    try:
        # 获取表单数据
        form = await request.form()
        file = form.get("file")

        if not file:
            raise HTTPException(status_code=400, detail="未找到上传的文件")

        # 检查是否配置了OSS
        if not os.getenv("OSS_ACCESS_KEY_ID"):
            raise HTTPException(
                status_code=503,
                detail="OSS未配置，请联系管理员配置 OSS_ACCESS_KEY_ID 和 OSS_ACCESS_KEY_SECRET"
            )

        # 验证文件类型
        allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp", "image/gif"]
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400,
                detail=f"不支持的文件类型: {file.content_type}，仅支持 jpg, png, webp, gif"
            )

        # 验证文件大小（最大5MB）
        content = await file.read()
        if len(content) > 5 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="文件大小不能超过5MB")

        # 保存到临时文件
        suffix = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp_file:
            tmp_file.write(content)
            tmp_path = tmp_file.name

        try:
            # 上传到OSS
            uploader = get_oss_uploader()
            url = uploader.upload_image(tmp_path)

            logger.info(f"✅ [VGP] 图片上传成功: {file.filename} -> {url}")

            return {
                "success": True,
                "url": url,
                "filename": file.filename,
                "size": len(content),
                "message": "图片上传成功"
            }

        finally:
            # 清理临时文件
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"❌ [VGP] 图片上传失败: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"图片上传失败: {str(e)}"
        )


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
