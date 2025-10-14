from dashscope import Generation
from typing import Dict, Any, List, Optional, Union
import json
import json5 as jsonlib  # 使用 json5 作为解析器
import asyncio
import time
import logging
from dataclasses import dataclass
from enum import Enum

class CallStatus(Enum):
    """API调用状态枚举"""
    SUCCESS = "success"
    FAILED = "failed"
    RATE_LIMITED = "rate_limited"
    TIMEOUT = "timeout"
    PARSING_ERROR = "parsing_error"
    VALIDATION_ERROR = "validation_error"


@dataclass
class QwenResponse:
    """Qwen API响应包装类"""
    status: CallStatus
    content: Optional[Union[str, Dict[str, Any]]] = None
    error_message: Optional[str] = None
    response_time: float = 0.0
    retry_count: int = 0
    model_used: Optional[str] = None


class QwenLLM:
    def __init__(self,
                 model_name: str = "qwen-max",
                 vl_model_name: str = "qwen-vl-max",
                 api_key: str = None,
                 timeout: int = 120,
                 max_retries: int = 3,
                 retry_delay: float = 1.0):
        """
        初始化Qwen LLM客户端

        Args:
            model_name: 文本模型名称
            vl_model_name: 视觉语言模型名称
            api_key: API密钥，如果为None则自动从环境变量读取
            timeout: 请求超时时间(秒)
            max_retries: 最大重试次数
            retry_delay: 重试间隔(秒)
        """
        # 自动从环境变量获取API密钥
        if api_key is None:
            import os
            api_key = os.getenv("DASHSCOPE_API_KEY") or os.getenv("AI__DASHSCOPE_API_KEY")

        if api_key:
            import dashscope
            dashscope.api_key = api_key
            self.logger = logging.getLogger(__name__)
            self.logger.info(f"✅ DashScope API密钥已设置: {api_key[:12]}...")
        else:
            self.logger = logging.getLogger(__name__)
            self.logger.warning("⚠️ 未找到DashScope API密钥，请设置DASHSCOPE_API_KEY环境变量")
            
        self.model_name = model_name
        self.vl_model_name = vl_model_name
        self.timeout = timeout
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.logger = logging.getLogger(__name__)
        
        # 调用统计
        self.call_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "average_response_time": 0.0
        }

    def generate(self, prompt: str, **kwargs):
        """
        原有的文本生成接口，返回字符串（兼容VGP节点）
        如果 parse_json=True，返回解析后的 dict，否则返回字符串
        """
        response = self.generate_full(prompt=prompt, **kwargs)

        if response.status == CallStatus.SUCCESS:
            # 如果启用了 JSON 解析且解析成功，返回字典
            if kwargs.get('parse_json', False) and isinstance(response.content, dict):
                return response.content
            # 否则返回字符串
            return str(response.content) if response.content else ""
        else:
            # 出错时返回空字符串或空字典
            self.logger.warning(f"QwenLLM生成失败: {response.error_message}")
            return {} if kwargs.get('parse_json', False) else ""

    def generate_full(
        self,
        prompt: str,
        images: List[str] = None,
        parse_json: bool = False,
        json_schema: Dict[str, Any] = None,
        max_retries: Optional[int] = None,
        **kwargs
    ) -> QwenResponse:
        """
        统一的文本/多模态生成接口。根据是否提供图片自动选择调用模式。

        Args:
            prompt: 提示词
            images: 可选，图像 URL 或 base64 列表。如果提供，则使用多模态模型
            parse_json: 是否尝试从输出中提取并解析 JSON
            json_schema: 可选，用于提示性校验 JSON 字段（不强制）
            max_retries: 最大重试次数，None则使用实例默认值
            **kwargs: 透传给底层模型的参数（如 temperature, top_p 等）
            
        Returns:
            QwenResponse: 包含调用结果和状态信息的响应对象
        """
        # 默认参数处理
        images = images or []
        if max_retries is None:
            max_retries = self.max_retries
            
        self.call_stats["total_calls"] += 1
        start_time = time.time()
        
        try:
            # 选择模型：有图用 VL，无图用文本模型
            if images:
                # 使用多模态模型 (VL)
                response = self._call_vl_model(
                    prompt=prompt,
                    images=images,
                    parse_json=parse_json,
                    json_schema=json_schema,
                    max_retries=max_retries,
                    **kwargs
                )
                response.model_used = self.vl_model_name
            else:
                # 使用纯文本模型
                response = self._call_text_model(
                    prompt=prompt,
                    parse_json=parse_json,
                    json_schema=json_schema,
                    max_retries=max_retries,
                    **kwargs
                )
                response.model_used = self.model_name
                
            # 更新统计信息
            response.response_time = time.time() - start_time
            if response.status == CallStatus.SUCCESS:
                self.call_stats["successful_calls"] += 1
            else:
                self.call_stats["failed_calls"] += 1
                
            # 更新平均响应时间
            self.call_stats["average_response_time"] = (
                (self.call_stats["average_response_time"] * (self.call_stats["total_calls"] - 1) + 
                 response.response_time) / self.call_stats["total_calls"]
            )
            
            return response
            
        except Exception as e:
            self.logger.error(f"Unexpected error in generate: {e}")
            return QwenResponse(
                status=CallStatus.FAILED,
                error_message=f"Unexpected error: {str(e)}",
                response_time=time.time() - start_time
            )

    def _call_text_model(
        self,
        prompt: str,
        parse_json: bool = False,
        json_schema: Dict[str, Any] = None,
        max_retries: int = 3,
        **kwargs
    ) -> QwenResponse:
        """调用纯文本模型"""
        last_error = None
        
        for retry_count in range(max_retries):
            try:
                # 重试延迟
                if retry_count > 0:
                    await_time = self.retry_delay * (2 ** (retry_count - 1))  # 指数退避
                    self.logger.info(f"Retrying after {await_time}s (attempt {retry_count + 1}/{max_retries})")
                    time.sleep(await_time)
                
                # 调用API
                call_start = time.time()
                response = Generation.call(
                    model=self.model_name,
                    prompt=prompt,
                    **kwargs
                )
                call_time = time.time() - call_start
                
                # 检查响应有效性
                if not response:
                    last_error = "Empty response from API"
                    continue
                    
                if not hasattr(response, 'output') or not response.output:
                    last_error = "No output in response"
                    continue
                    
                if not hasattr(response.output, 'text') or not response.output.text:
                    last_error = "No text in response output"
                    continue

                text = response.output.text.strip()
                self.logger.debug(f"QwenLLM text response: {text[:200]}...")

                if not parse_json:
                    return QwenResponse(
                        status=CallStatus.SUCCESS,
                        content=text,
                        response_time=call_time,
                        retry_count=retry_count
                    )

                # 尝试提取和解析 JSON
                json_str = self._extract_json(text)
                if not json_str:
                    last_error = f"No JSON found in response: {text[:100]}..."
                    self.logger.warning(f"No JSON extracted from: {text[:200]}...")
                    continue

                try:
                    result = jsonlib.loads(json_str)
                except Exception as parse_error:
                    last_error = f"JSON parsing failed: {str(parse_error)}"
                    self.logger.warning(f"JSON parse error: {parse_error}")
                    continue

                # 校验 JSON 结构
                if json_schema:
                    validation_result = self._validate_json_schema(result, json_schema)
                    if not validation_result.is_valid:
                        last_error = f"JSON validation failed: {validation_result.error_message}"
                        self.logger.warning(f"JSON validation failed: {validation_result.error_message}")
                        # 注意：这里不 continue，但记录验证错误
                        return QwenResponse(
                            status=CallStatus.VALIDATION_ERROR,
                            content=result,
                            error_message=validation_result.error_message,
                            response_time=call_time,
                            retry_count=retry_count
                        )

                return QwenResponse(
                    status=CallStatus.SUCCESS,
                    content=result,
                    response_time=call_time,
                    retry_count=retry_count
                )

            except Exception as e:
                last_error = str(e)
                self.logger.error(f"Text model call error (attempt {retry_count + 1}): {e}")
                
                # 检查是否是速率限制
                if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                    return QwenResponse(
                        status=CallStatus.RATE_LIMITED,
                        error_message=str(e),
                        retry_count=retry_count
                    )
                    
                # 检查是否是超时
                if "timeout" in str(e).lower():
                    return QwenResponse(
                        status=CallStatus.TIMEOUT,
                        error_message=str(e),
                        retry_count=retry_count
                    )
                
                continue

        return QwenResponse(
            status=CallStatus.FAILED,
            error_message=last_error or "All retries failed",
            retry_count=max_retries
        )


    def _call_vl_model(
        self,
        prompt: str,
        images: List[str],
        parse_json: bool = False,
        json_schema: Dict[str, Any] = None,
        max_retries: int = 3,
        **kwargs
    ) -> QwenResponse:
        """调用多模态模型"""
        last_error = None
        
        for retry_count in range(max_retries):
            try:
                # 重试延迟
                if retry_count > 0:
                    await_time = self.retry_delay * (2 ** (retry_count - 1))  # 指数退避
                    self.logger.info(f"VL model retrying after {await_time}s (attempt {retry_count + 1}/{max_retries})")
                    time.sleep(await_time)
                
                # 调用VL API
                call_start = time.time()
                response = Generation.call(
                    model=self.vl_model_name,
                    prompt=prompt,
                    images=images,
                    **kwargs
                )
                call_time = time.time() - call_start
                
                if not response:
                    last_error = "Empty response from VL API"
                    continue

                # VL模型响应格式可能不同，需要适配
                try:
                    if hasattr(response, 'output') and hasattr(response.output, 'text'):
                        text = response.output.text.strip()
                    else:
                        text = str(response).strip()
                except Exception as extract_error:
                    last_error = f"Failed to extract text from VL response: {extract_error}"
                    continue
                    
                self.logger.debug(f"QwenVL response: {text[:200]}...")

                if not parse_json:
                    return QwenResponse(
                        status=CallStatus.SUCCESS,
                        content=text,
                        response_time=call_time,
                        retry_count=retry_count
                    )

                # 尝试提取和解析 JSON
                json_str = self._extract_json(text)
                if not json_str:
                    last_error = f"No JSON found in VL response: {text[:100]}..."
                    self.logger.warning(f"No JSON extracted from VL: {text[:200]}...")
                    continue

                try:
                    result = jsonlib.loads(json_str)
                except Exception as parse_error:
                    last_error = f"VL JSON parsing failed: {str(parse_error)}"
                    self.logger.warning(f"VL JSON parse error: {parse_error}")
                    continue

                # 校验 JSON 结构
                if json_schema:
                    validation_result = self._validate_json_schema(result, json_schema)
                    if not validation_result.is_valid:
                        last_error = f"VL JSON validation failed: {validation_result.error_message}"
                        self.logger.warning(f"VL JSON validation failed: {validation_result.error_message}")
                        return QwenResponse(
                            status=CallStatus.VALIDATION_ERROR,
                            content=result,
                            error_message=validation_result.error_message,
                            response_time=call_time,
                            retry_count=retry_count
                        )

                return QwenResponse(
                    status=CallStatus.SUCCESS,
                    content=result,
                    response_time=call_time,
                    retry_count=retry_count
                )

            except Exception as e:
                last_error = str(e)
                self.logger.error(f"VL model call error (attempt {retry_count + 1}): {e}")
                
                # 检查特定错误类型
                if "rate limit" in str(e).lower() or "too many requests" in str(e).lower():
                    return QwenResponse(
                        status=CallStatus.RATE_LIMITED,
                        error_message=str(e),
                        retry_count=retry_count
                    )
                    
                if "timeout" in str(e).lower():
                    return QwenResponse(
                        status=CallStatus.TIMEOUT,
                        error_message=str(e),
                        retry_count=retry_count
                    )
                
                continue

        return QwenResponse(
            status=CallStatus.FAILED,
            error_message=last_error or "All VL retries failed",
            retry_count=max_retries
        )
    
    @dataclass
    class ValidationResult:
        """JSON校验结果"""
        is_valid: bool
        error_message: Optional[str] = None
        missing_fields: List[str] = None
        extra_fields: List[str] = None
    
    def _validate_json_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> 'QwenLLM.ValidationResult':
        """校验JSON数据是否符合期望的结构"""
        try:
            missing_fields = []
            extra_fields = []
            
            # 检查必需字段
            for key, expected_type in schema.items():
                if key not in data:
                    missing_fields.append(key)
                elif expected_type and not isinstance(data[key], expected_type):
                    return self.ValidationResult(
                        is_valid=False,
                        error_message=f"Field '{key}' expected {expected_type.__name__}, got {type(data[key]).__name__}",
                        missing_fields=missing_fields
                    )
            
            # 检查额外字段（仅记录，不影响有效性）
            for key in data.keys():
                if key not in schema:
                    extra_fields.append(key)
            
            if missing_fields:
                return self.ValidationResult(
                    is_valid=False,
                    error_message=f"Missing required fields: {missing_fields}",
                    missing_fields=missing_fields,
                    extra_fields=extra_fields
                )
            
            return self.ValidationResult(
                is_valid=True,
                missing_fields=[],
                extra_fields=extra_fields
            )
            
        except Exception as e:
            return self.ValidationResult(
                is_valid=False,
                error_message=f"Validation error: {str(e)}"
            )
    
    def get_call_stats(self) -> Dict[str, Any]:
        """获取调用统计信息"""
        return self.call_stats.copy()
    
    def reset_stats(self):
        """重置统计信息"""
        self.call_stats = {
            "total_calls": 0,
            "successful_calls": 0,
            "failed_calls": 0,
            "average_response_time": 0.0
        }
        
    def multi_generate(self, prompt: str, context: List[Dict[str, str]], max_tokens: int = 1000) -> str:
        from dashscope import Generation

        # 将 context 转换为 Qwen 所需格式：将 role 改为 user/assistant
        formatted_messages = []
        for msg in context:
            role = 'user' if msg['role'] == 'user' else 'assistant'
            formatted_messages.append({'role': role, 'content': msg['content']})
        formatted_messages.append({'role': 'user', 'content': prompt})

        response = Generation.call(
            model=self.model_name,
            input={'prompt': prompt, 'history': [(msg['role'], msg['content']) for msg in context]}
        )

        return response.output.text
    
    # def generate_vlu(
    #     self,
    #     prompt: str,
    #     images: List[str],
    #     parse_json: bool = False,
    #     json_schema: Dict[str, Any] = None,  # 可选：用于校验结构
    #     max_retries: int = 3,
    #     **kwargs
    # ) -> Union[Optional[str], Optional[Dict[str, Any]]]:
    #     """
    #     通用多模态调用接口

    #     :param prompt: 提示词（由调用方构造）
    #     :param images: 图像 URL 或 base64 列表
    #     :param parse_json: 是否尝试解析 JSON 输出
    #     :param json_schema: 可选，用于校验字段（仅提示，不强制）
    #     :param max_retries: 重试次数
    #     :param **kwargs: 透传给底层模型（如 temperature, top_p 等）
    #     :return: 字符串 或 解析后的 dict，失败返回 None
    #     """
    #     for _ in range(max_retries):
    #         try:
    #             # 调用多模态模型
    #             response = self.vl.generate(
    #                 prompt=prompt,
    #                 images=images,  # 假设模型接口支持 images 参数
    #                 **kwargs
    #             )
    #             if not response:
    #                 continue

    #             text = str(response.strip())
    #             if not parse_json:
    #                 return text

    #             # 尝试提取 JSON
    #             json_str = self._extract_json(text)
    #             if not json_str:
    #                 print(f"[QwenCaller] 未找到 JSON: {text[:200]}...")
    #                 continue

    #             result = json.loads(json_str)

    #             # 可选：简单校验关键字段（仅提示，不抛异常）
    #             if json_schema:
    #                 missing = [k for k in json_schema.keys() if k not in result]
    #                 if missing:
    #                     print(f"[QwenCaller] JSON 缺少字段 {missing}，期望: {list(json_schema.keys())}")

    #             return result

    #         except Exception as e:
    #             print(f"[QwenCaller VL Call Error] {e}")
    #             continue

    #     return None

    @staticmethod
    def _extract_json(text: str) -> Optional[str]:
        """
        从文本中提取第一个最外层的合法 JSON 对象或数组。
        使用 json5 支持宽松语法（单引号、注释、尾逗号、无引号键等）。
        能跳过非法或不完整的结构，继续搜索后续可能的 JSON。
        """
        if not text or not isinstance(text, str):
            return None

        stack = []
        start = -1
        i = 0
        n = len(text)

        while i < n:
            c = text[i]

            if c in '{[':
                if not stack:
                    start = i  # 记录最外层开始位置
                stack.append(c)

            elif c in '}]':
                if stack:
                    opening = stack.pop()
                    # 检查括号匹配
                    if (opening == '{' and c != '}') or (opening == '[' and c != ']'):
                        # 不匹配，清空状态，继续
                        stack.clear()
                        start = -1
                    else:
                        if not stack and start != -1:
                            # 最外层闭合，尝试解析
                            candidate = text[start:i+1]
                            try:
                                jsonlib.loads(candidate)  # 使用 json5 解析
                                return candidate
                            except Exception:
                                # 解析失败，重置 start，继续寻找下一个
                                start = -1
                # else: 多余的 ] 或 }，忽略
            i += 1

        return None
