"""
Enhanced Qwen Service

Improved Qwen LLM service with better stability, error handling, and monitoring.
"""

import asyncio
import time
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import logging

from llm.qwen import QwenLLM, QwenResponse, CallStatus
from config import get_settings


@dataclass
class AIServiceMetrics:
    """AI服务指标"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    average_response_time: float = 0.0
    success_rate: float = 0.0
    last_request_time: Optional[float] = None


class EnhancedQwenService:
    """
    增强的Qwen AI服务
    
    提供稳定的AI调用接口，包含：
    - 智能重试机制
    - 请求监控和指标收集
    - 自适应超时控制
    - 错误分类和处理
    - 异步支持
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.config = config or {}
        self.settings = get_settings()
        self.logger = logging.getLogger(__name__)
        
        # 初始化Qwen客户端
        self.qwen = QwenLLM(
            model_name=self.settings.ai.qwen_model_name,
            vl_model_name=self.settings.ai.qwen_vl_model_name,
            api_key=self.settings.ai.dashscope_api_key,
            timeout=self.settings.ai.ai_request_timeout,
            max_retries=self.settings.ai.max_retries,
            retry_delay=self.settings.ai.retry_delay
        )
        
        # 服务指标
        self.metrics = AIServiceMetrics()
        
        # 自适应配置
        self.adaptive_timeout = self.settings.ai.ai_request_timeout
        self.circuit_breaker_threshold = 10  # 连续失败次数阈值
        self.consecutive_failures = 0
        self.circuit_open = False
        self.circuit_reset_time = 0.0
    
    async def generate_async(
        self,
        prompt: str,
        images: Optional[List[str]] = None,
        parse_json: bool = False,
        json_schema: Optional[Dict[str, Any]] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> QwenResponse:
        """
        异步生成内容
        
        Args:
            prompt: 提示词
            images: 图像列表（用于多模态）
            parse_json: 是否解析JSON
            json_schema: JSON模式验证
            context: 上下文信息
            **kwargs: 其他参数
            
        Returns:
            QwenResponse: 增强的响应对象
        """
        # 更新请求指标
        self.metrics.total_requests += 1
        self.metrics.last_request_time = time.time()
        
        # 检查熔断器
        if self._is_circuit_open():
            return QwenResponse(
                status=CallStatus.FAILED,
                error_message="Circuit breaker is open - service temporarily unavailable",
                response_time=0.0
            )
        
        try:
            # 运行在线程池中以避免阻塞异步循环
            loop = asyncio.get_event_loop()
            response = await loop.run_in_executor(
                None,
                self._sync_generate,
                prompt, images, parse_json, json_schema, context, kwargs
            )
            
            # 更新成功指标
            if response.status == CallStatus.SUCCESS:
                self.metrics.successful_requests += 1
                self.consecutive_failures = 0
                self._update_adaptive_timeout(response.response_time)
            else:
                self.metrics.failed_requests += 1
                self.consecutive_failures += 1
                self._check_circuit_breaker()
            
            # 更新指标
            self._update_metrics()
            
            return response
            
        except Exception as e:
            self.logger.error(f"Unexpected error in async generate: {e}")
            self.metrics.failed_requests += 1
            self.consecutive_failures += 1
            self._update_metrics()
            
            return QwenResponse(
                status=CallStatus.FAILED,
                error_message=f"Service error: {str(e)}",
                response_time=0.0
            )
    
    def _sync_generate(
        self,
        prompt: str,
        images: Optional[List[str]],
        parse_json: bool,
        json_schema: Optional[Dict[str, Any]],
        context: Optional[Dict[str, Any]],
        kwargs: Dict[str, Any]
    ) -> QwenResponse:
        """同步生成（在线程池中运行）"""
        # 应用上下文增强
        enhanced_prompt = self._enhance_prompt_with_context(prompt, context)
        
        # 调用底层Qwen服务
        response = self.qwen.generate(
            prompt=enhanced_prompt,
            images=images,
            parse_json=parse_json,
            json_schema=json_schema,
            **kwargs
        )
        
        # 增强响应信息
        if response.status == CallStatus.SUCCESS and context:
            response = self._enhance_response_with_context(response, context)
        
        return response
    
    def _enhance_prompt_with_context(self, prompt: str, context: Optional[Dict[str, Any]]) -> str:
        """使用上下文增强提示词"""
        if not context:
            return prompt
        
        enhanced_prompt = prompt
        
        # 添加任务类型上下文
        if 'task_type' in context:
            task_type = context['task_type']
            enhanced_prompt = f"[任务类型: {task_type}]\n{enhanced_prompt}"
        
        # 添加用户偏好
        if 'user_preferences' in context:
            prefs = context['user_preferences']
            pref_str = ", ".join([f"{k}: {v}" for k, v in prefs.items()])
            enhanced_prompt = f"[用户偏好: {pref_str}]\n{enhanced_prompt}"
        
        # 添加历史上下文
        if 'conversation_history' in context:
            history = context['conversation_history'][-3:]  # 只取最近3条
            history_str = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history])
            enhanced_prompt = f"[对话历史]\n{history_str}\n\n[当前请求]\n{enhanced_prompt}"
        
        return enhanced_prompt
    
    def _enhance_response_with_context(self, response: QwenResponse, context: Dict[str, Any]) -> QwenResponse:
        """使用上下文增强响应"""
        # 添加上下文标签到响应中
        if hasattr(response, 'metadata'):
            response.metadata = getattr(response, 'metadata', {})
        else:
            response.metadata = {}
        
        response.metadata.update({
            'task_type': context.get('task_type'),
            'user_id': context.get('user_id'),
            'session_id': context.get('session_id'),
            'enhanced': True
        })
        
        return response
    
    def _is_circuit_open(self) -> bool:
        """检查熔断器是否开启"""
        if not self.circuit_open:
            return False
        
        # 检查是否到了重置时间
        if time.time() > self.circuit_reset_time:
            self.circuit_open = False
            self.consecutive_failures = 0
            self.logger.info("Circuit breaker reset")
            return False
        
        return True
    
    def _check_circuit_breaker(self):
        """检查是否需要开启熔断器"""
        if self.consecutive_failures >= self.circuit_breaker_threshold:
            self.circuit_open = True
            self.circuit_reset_time = time.time() + 60  # 60秒后重置
            self.logger.warning(f"Circuit breaker opened after {self.consecutive_failures} consecutive failures")
    
    def _update_adaptive_timeout(self, response_time: float):
        """更新自适应超时时间"""
        # 简单的自适应逻辑：基于最近响应时间调整超时
        if response_time > self.adaptive_timeout * 0.8:
            # 如果响应时间接近超时，增加超时时间
            self.adaptive_timeout = min(self.adaptive_timeout * 1.2, 300)  # 最大5分钟
        elif response_time < self.adaptive_timeout * 0.3:
            # 如果响应很快，适当减少超时时间
            self.adaptive_timeout = max(self.adaptive_timeout * 0.9, self.settings.ai.ai_request_timeout)
    
    def _update_metrics(self):
        """更新服务指标"""
        if self.metrics.total_requests > 0:
            self.metrics.success_rate = self.metrics.successful_requests / self.metrics.total_requests
        
        # 更新Qwen统计
        qwen_stats = self.qwen.get_call_stats()
        if qwen_stats.get('total_calls', 0) > 0:
            self.metrics.average_response_time = qwen_stats.get('average_response_time', 0.0)
    
    def get_service_health(self) -> Dict[str, Any]:
        """获取服务健康状态"""
        return {
            'status': 'healthy' if not self.circuit_open else 'degraded',
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'successful_requests': self.metrics.successful_requests,
                'failed_requests': self.metrics.failed_requests,
                'success_rate': self.metrics.success_rate,
                'average_response_time': self.metrics.average_response_time,
                'consecutive_failures': self.consecutive_failures
            },
            'circuit_breaker': {
                'open': self.circuit_open,
                'reset_time': self.circuit_reset_time if self.circuit_open else None
            },
            'adaptive_timeout': self.adaptive_timeout,
            'qwen_stats': self.qwen.get_call_stats()
        }
    
    def reset_service_metrics(self):
        """重置服务指标"""
        self.metrics = AIServiceMetrics()
        self.qwen.reset_stats()
        self.consecutive_failures = 0
        self.circuit_open = False
        self.logger.info("Service metrics reset")


# 全局实例
_enhanced_qwen_service = None


def get_enhanced_qwen_service() -> EnhancedQwenService:
    """获取增强Qwen服务实例"""
    global _enhanced_qwen_service
    if _enhanced_qwen_service is None:
        _enhanced_qwen_service = EnhancedQwenService()
    return _enhanced_qwen_service