"""
AI Service API Endpoints

FastAPI endpoints for enhanced AI services including vision analysis,
quality assessment, and intelligent content generation.
"""

from typing import Dict, Any, List, Optional, Union
from datetime import datetime
from fastapi import APIRouter, HTTPException, Depends, File, UploadFile, Form
from pydantic import BaseModel, Field
import json
import base64
from pathlib import Path

from ai_service import (
    get_enhanced_qwen_service,
    get_prompt_manager, 
    get_context_manager,
    get_quality_assessor,
    PromptTemplate,
    PromptType,
    DifficultyLevel,
    ContentType,
    ContextType,
    ContextPriority
)
from ai_service.vision_analyzer import get_vision_analyzer, AnalysisType
from config import get_settings


# Pydantic models for API
class AIGenerationRequest(BaseModel):
    """AI生成请求"""
    prompt: str = Field(..., description="提示词内容")
    template_id: Optional[str] = Field(None, description="提示词模板ID")
    template_variables: Optional[Dict[str, Any]] = Field(None, description="模板变量")
    parse_json: bool = Field(False, description="是否解析JSON输出")
    json_schema: Optional[Dict[str, Any]] = Field(None, description="JSON模式验证")
    context: Optional[Dict[str, Any]] = Field(None, description="生成上下文")
    session_id: Optional[str] = Field(None, description="会话ID")
    
    class Config:
        json_schema_extra = {
            "example": {
                "prompt": "生成一个关于科技创新的视频脚本",
                "template_id": "video_script_generation",
                "template_variables": {
                    "topic": "人工智能",
                    "duration": "60",
                    "target_audience": "年轻人",
                    "style": "科技感"
                },
                "parse_json": True,
                "context": {
                    "user_id": "user123",
                    "project_type": "科技视频"
                }
            }
        }


class VisionAnalysisRequest(BaseModel):
    """视觉分析请求"""
    image_url: Optional[str] = Field(None, description="图像URL")
    image_base64: Optional[str] = Field(None, description="Base64编码的图像")
    analysis_type: str = Field(..., description="分析类型")
    context: Optional[Dict[str, Any]] = Field(None, description="分析上下文")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_url": "https://example.com/image.jpg",
                "analysis_type": "scene_analysis",
                "context": {
                    "project_context": "商业视频",
                    "target_style": "现代简约"
                }
            }
        }


class QualityAssessmentRequest(BaseModel):
    """质量评估请求"""
    content: Union[str, Dict[str, Any]] = Field(..., description="待评估内容")
    content_type: str = Field(..., description="内容类型")
    context: Optional[Dict[str, Any]] = Field(None, description="评估上下文")
    
    class Config:
        json_schema_extra = {
            "example": {
                "content": {
                    "title": "AI科技视频",
                    "scenes": [{"description": "展示AI应用场景"}]
                },
                "content_type": "video_script",
                "context": {
                    "target_audience": "技术从业者",
                    "requirements": ["专业", "简洁", "有吸引力"]
                }
            }
        }


class PromptTemplateRequest(BaseModel):
    """提示词模板请求"""
    name: str = Field(..., description="模板名称")
    type: str = Field(..., description="模板类型")
    content: str = Field(..., description="模板内容")
    variables: List[str] = Field(default_factory=list, description="变量列表")
    description: Optional[str] = Field(None, description="模板描述")
    tags: List[str] = Field(default_factory=list, description="标签")
    difficulty: str = Field(default="simple", description="难度级别")


class ContextUpdateRequest(BaseModel):
    """上下文更新请求"""
    session_id: str = Field(..., description="会话ID")
    context_type: str = Field(..., description="上下文类型")
    content: Dict[str, Any] = Field(..., description="上下文内容")
    priority: str = Field(default="medium", description="优先级")
    ttl: Optional[int] = Field(None, description="生存时间(秒)")


# Create router
ai_router = APIRouter(prefix="/ai", tags=["AI Services"])


@ai_router.post("/generate", response_model=Dict[str, Any])
async def generate_content(request: AIGenerationRequest):
    """生成AI内容"""
    try:
        qwen_service = get_enhanced_qwen_service()
        context_manager = get_context_manager()
        
        # 构建提示词
        final_prompt = request.prompt
        if request.template_id:
            prompt_manager = get_prompt_manager()
            variables = request.template_variables or {}
            
            # 检查是否需要上下文增强
            if request.session_id:
                context_templates = []
                if variables.get('use_context'):
                    context_templates.append('style_consistency_check')
                
                final_prompt = prompt_manager.build_contextual_prompt(
                    base_template_id=request.template_id,
                    variables=variables,
                    context_templates=context_templates,
                    system_template_id=variables.get('system_template')
                )
            else:
                final_prompt = prompt_manager.render_prompt(request.template_id, variables)
            
            if not final_prompt:
                raise HTTPException(status_code=400, detail="Failed to render prompt template")
        
        # 更新会话上下文
        enhanced_context = request.context or {}
        if request.session_id:
            enhanced_context['session_id'] = request.session_id
            enhanced_context['task_type'] = 'content_generation'
            
            # 获取相关历史上下文
            relevant_contexts = context_manager.get_relevant_contexts(
                query_context={'type': 'conversation', 'content': final_prompt},
                max_results=5
            )
            if relevant_contexts:
                enhanced_context['conversation_history'] = [
                    {'role': 'assistant', 'content': str(ctx.content)} 
                    for ctx in relevant_contexts[-3:]
                ]
        
        # 调用AI服务
        response = await qwen_service.generate_async(
            prompt=final_prompt,
            parse_json=request.parse_json,
            json_schema=request.json_schema,
            context=enhanced_context
        )
        
        # 记录对话历史
        if request.session_id:
            context_manager.update_conversation(
                session_id=request.session_id,
                role='user',
                content=request.prompt,
                metadata={'template_id': request.template_id}
            )
            context_manager.update_conversation(
                session_id=request.session_id,
                role='assistant',
                content=str(response.content),
                metadata={
                    'model': response.model_used,
                    'confidence': getattr(response, 'confidence', None),
                    'processing_time': response.response_time
                }
            )
        
        return {
            "status": response.status.value,
            "content": response.content,
            "metadata": {
                "model_used": response.model_used,
                "processing_time": response.response_time,
                "retry_count": response.retry_count,
                "timestamp": datetime.now().isoformat()
            },
            "error_message": response.error_message
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@ai_router.post("/vision/analyze", response_model=Dict[str, Any])
async def analyze_vision(request: VisionAnalysisRequest):
    """视觉分析"""
    try:
        vision_analyzer = get_vision_analyzer()
        
        # 验证输入
        if not request.image_url and not request.image_base64:
            raise HTTPException(status_code=400, detail="Either image_url or image_base64 must be provided")
        
        # 验证分析类型
        try:
            analysis_type = AnalysisType(request.analysis_type)
        except ValueError:
            valid_types = [t.value for t in AnalysisType]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid analysis_type. Must be one of: {valid_types}"
            )
        
        # 准备图像数据
        image_data = request.image_url if request.image_url else request.image_base64
        
        # 执行分析
        result = await vision_analyzer.analyze_image(
            image_data=image_data,
            analysis_type=analysis_type,
            context=request.context
        )
        
        return {
            "analysis_type": result.analysis_type.value,
            "confidence": result.confidence,
            "results": result.results,
            "metadata": {
                "processing_time": result.processing_time,
                "timestamp": result.timestamp.isoformat(),
                **result.metadata
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Vision analysis failed: {str(e)}")


@ai_router.post("/vision/analyze-file", response_model=Dict[str, Any])
async def analyze_vision_file(
    file: UploadFile = File(...),
    analysis_type: str = Form(...),
    context: Optional[str] = Form(None)
):
    """通过文件上传进行视觉分析"""
    try:
        # 验证文件类型
        allowed_types = ['image/jpeg', 'image/jpg', 'image/png', 'image/gif', 'image/webp']
        if file.content_type not in allowed_types:
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type: {file.content_type}. Allowed: {allowed_types}"
            )
        
        # 验证分析类型
        try:
            analysis_type_enum = AnalysisType(analysis_type)
        except ValueError:
            valid_types = [t.value for t in AnalysisType]
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid analysis_type. Must be one of: {valid_types}"
            )
        
        # 读取文件数据
        file_content = await file.read()
        
        # 解析上下文
        context_dict = None
        if context:
            try:
                context_dict = json.loads(context)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid JSON format for context")
        
        # 执行分析
        vision_analyzer = get_vision_analyzer()
        result = await vision_analyzer.analyze_image(
            image_data=file_content,
            analysis_type=analysis_type_enum,
            context=context_dict
        )
        
        return {
            "analysis_type": result.analysis_type.value,
            "confidence": result.confidence,
            "results": result.results,
            "metadata": {
                "processing_time": result.processing_time,
                "timestamp": result.timestamp.isoformat(),
                "file_info": {
                    "filename": file.filename,
                    "content_type": file.content_type,
                    "size": len(file_content)
                },
                **result.metadata
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"File vision analysis failed: {str(e)}")


@ai_router.post("/quality/assess", response_model=Dict[str, Any])
async def assess_quality(request: QualityAssessmentRequest):
    """内容质量评估"""
    try:
        quality_assessor = get_quality_assessor()
        
        # 验证内容类型
        try:
            content_type = ContentType(request.content_type)
        except ValueError:
            valid_types = [t.value for t in ContentType]
            raise HTTPException(
                status_code=400,
                detail=f"Invalid content_type. Must be one of: {valid_types}"
            )
        
        # 执行质量评估
        assessment = quality_assessor.assess_content(
            content=request.content,
            content_type=content_type,
            context=request.context
        )
        
        # 转换评估结果
        dimension_scores = {
            dim.value: {
                "score": score.score,
                "reasoning": score.reasoning,
                "confidence": score.confidence,
                "details": score.details
            }
            for dim, score in assessment.dimension_scores.items()
        }
        
        return {
            "content_type": assessment.content_type.value,
            "overall_score": assessment.overall_score,
            "dimension_scores": dimension_scores,
            "feedback": assessment.feedback,
            "suggestions": assessment.suggestions,
            "metadata": {
                "assessment_time": assessment.assessment_time.isoformat(),
                **assessment.metadata
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Quality assessment failed: {str(e)}")


@ai_router.get("/prompts/templates", response_model=List[Dict[str, Any]])
async def list_prompt_templates(
    type_filter: Optional[str] = None,
    tag_filter: Optional[str] = None,
    difficulty_filter: Optional[str] = None
):
    """列出提示词模板"""
    try:
        prompt_manager = get_prompt_manager()
        
        # 解析过滤器
        prompt_type = PromptType(type_filter) if type_filter else None
        difficulty = DifficultyLevel(difficulty_filter) if difficulty_filter else None
        
        templates = prompt_manager.list_templates(
            type_filter=prompt_type,
            tag_filter=tag_filter,
            difficulty_filter=difficulty
        )
        
        return [
            {
                "id": t.id,
                "name": t.name,
                "type": t.type.value,
                "description": t.description,
                "variables": t.variables,
                "tags": t.tags,
                "difficulty": t.difficulty.value,
                "created_at": t.created_at.isoformat() if t.created_at else None
            }
            for t in templates
        ]
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list templates: {str(e)}")


@ai_router.post("/prompts/templates", response_model=Dict[str, str])
async def create_prompt_template(request: PromptTemplateRequest):
    """创建提示词模板"""
    try:
        prompt_manager = get_prompt_manager()
        
        # 验证类型和难度
        try:
            prompt_type = PromptType(request.type)
            difficulty = DifficultyLevel(request.difficulty)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
        
        # 创建模板
        template = PromptTemplate(
            id=f"custom_{len(prompt_manager.templates)}_{request.name.lower().replace(' ', '_')}",
            name=request.name,
            type=prompt_type,
            content=request.content,
            variables=request.variables,
            description=request.description,
            tags=request.tags,
            difficulty=difficulty
        )
        
        # 保存模板
        success = prompt_manager.save_template(template)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to save template")
        
        return {
            "message": "Template created successfully",
            "template_id": template.id
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create template: {str(e)}")


@ai_router.get("/prompts/templates/{template_id}", response_model=Dict[str, Any])
async def get_prompt_template(template_id: str):
    """获取特定提示词模板"""
    try:
        prompt_manager = get_prompt_manager()
        template = prompt_manager.get_template(template_id)
        
        if not template:
            raise HTTPException(status_code=404, detail="Template not found")
        
        return {
            "id": template.id,
            "name": template.name,
            "type": template.type.value,
            "content": template.content,
            "variables": template.variables,
            "description": template.description,
            "tags": template.tags,
            "difficulty": template.difficulty.value,
            "examples": template.examples,
            "created_at": template.created_at.isoformat() if template.created_at else None,
            "updated_at": template.updated_at.isoformat() if template.updated_at else None
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get template: {str(e)}")


@ai_router.post("/context/sessions", response_model=Dict[str, str])
async def create_context_session(
    user_id: str,
    metadata: Optional[Dict[str, Any]] = None
):
    """创建上下文会话"""
    try:
        context_manager = get_context_manager()
        
        # 生成会话ID
        session_id = f"session_{user_id}_{int(datetime.now().timestamp())}"
        
        success = context_manager.create_session(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to create session")
        
        return {
            "message": "Session created successfully",
            "session_id": session_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {str(e)}")


@ai_router.put("/context/sessions/{session_id}")
async def update_context(session_id: str, request: ContextUpdateRequest):
    """更新会话上下文"""
    try:
        context_manager = get_context_manager()
        
        # 验证类型和优先级
        try:
            context_type = ContextType(request.context_type)
            priority = ContextPriority(request.priority)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=f"Invalid enum value: {str(e)}")
        
        context_id = f"{session_id}_{request.context_type}_{int(datetime.now().timestamp())}"
        
        success = context_manager.add_context(
            context_id=context_id,
            context_type=context_type,
            content=request.content,
            priority=priority,
            ttl=request.ttl
        )
        
        if not success:
            raise HTTPException(status_code=500, detail="Failed to update context")
        
        return {"message": "Context updated successfully"}
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to update context: {str(e)}")


@ai_router.get("/context/sessions/{session_id}/history")
async def get_conversation_history(session_id: str, limit: int = 20):
    """获取对话历史"""
    try:
        context_manager = get_context_manager()
        history = context_manager.get_conversation_history(session_id, limit)
        
        return {
            "session_id": session_id,
            "history": history,
            "count": len(history)
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get conversation history: {str(e)}")


@ai_router.get("/stats", response_model=Dict[str, Any])
async def get_ai_service_stats():
    """获取AI服务统计信息"""
    try:
        # 收集各服务的统计信息
        stats = {
            "timestamp": datetime.now().isoformat(),
            "services": {}
        }
        
        # Qwen服务统计
        try:
            qwen_service = get_enhanced_qwen_service()
            stats["services"]["qwen"] = qwen_service.get_service_health()
        except Exception as e:
            stats["services"]["qwen"] = {"error": str(e)}
        
        # 提示词管理器统计
        try:
            prompt_manager = get_prompt_manager()
            stats["services"]["prompts"] = {
                "total_templates": len(prompt_manager.templates),
                "usage_stats": prompt_manager.get_usage_stats()
            }
        except Exception as e:
            stats["services"]["prompts"] = {"error": str(e)}
        
        # 上下文管理器统计
        try:
            context_manager = get_context_manager()
            stats["services"]["context"] = context_manager.get_stats()
        except Exception as e:
            stats["services"]["context"] = {"error": str(e)}
        
        # 质量评估器统计
        try:
            quality_assessor = get_quality_assessor()
            stats["services"]["quality"] = quality_assessor.get_quality_trends()
        except Exception as e:
            stats["services"]["quality"] = {"error": str(e)}
        
        # 视觉分析器统计
        try:
            vision_analyzer = get_vision_analyzer()
            stats["services"]["vision"] = vision_analyzer.get_analysis_stats()
        except Exception as e:
            stats["services"]["vision"] = {"error": str(e)}
        
        return stats
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get service stats: {str(e)}")


@ai_router.post("/reset", response_model=Dict[str, str])
async def reset_ai_services():
    """重置AI服务状态（开发/测试用）"""
    try:
        # 重置各服务
        qwen_service = get_enhanced_qwen_service()
        qwen_service.reset_service_metrics()
        
        context_manager = get_context_manager()
        context_manager.reset()
        
        vision_analyzer = get_vision_analyzer()
        vision_analyzer.clear_history()
        
        return {"message": "AI services reset successfully"}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reset services: {str(e)}")


# Export router
__all__ = ['ai_router']