# ai_content_pipeline/core/types.py
from typing import TypedDict, Literal, Optional, Dict, Any, List

# -----------------------------
# 通用类型定义
# -----------------------------

class BaseResult(TypedDict, total=False):
    """
    所有生成器返回结果的基类型
    """
    status: Literal["success", "failed", "processing"]
    error: str
    task_id: str
    metadata: Dict[str, Any]


class MediaResult(BaseResult, total=False):
    """
    媒体生成结果（视频/音频/图片）
    """
    url: str                  # 文件 URL
    local_path: str           # 本地路径（可选）
    duration: float           # 时长（秒，视频/音频）
    width: int                # 宽度
    height: int               # 高度
    format: str               # 格式：mp4, webm, mp3, png 等
    size_bytes: int           # 文件大小


class TextToSpeechResult(MediaResult, total=False):
    """
    TTS 生成结果
    """
    text_used: str
    voice_style: str
    audio_url: str  # 兼容字段


class TalkingHeadResult(MediaResult, total=False):
    """
    数字人生成结果
    """
    video_url: str
    audio_url: str
    text_used: str
    visemes: List[Dict[str, Any]]  # 口型数据


class AIVideoResult(MediaResult, total=False):
    """
    AI 视频生成结果
    """
    video_url: str
    prompt_used: str
    image_start_used: Optional[str]
    image_end_used: Optional[str]


class AIImageResult(MediaResult, total=False):
    """
    AI 图像生成结果
    """
    image_url: str
    prompt_used: str


# -----------------------------
# 任务与工作流类型
# -----------------------------

class TaskConfig(TypedDict, total=False):
    """
    任务配置（用于策略定义）
    """
    generator_key: str                 # 生成器标识（如 'aliyun_tts'）
    params: Dict[str, Any]             # 参数（支持模板 {{var}}）
    output_key: Optional[str]          # 输出绑定名（用于后续任务引用）
    timeout: Optional[float]           # 超时时间（秒）
    retry_policy: Optional['RetryPolicyConfig']


class WorkflowConfig(TypedDict, total=False):
    """
    工作流配置
    """
    name: str
    description: Optional[str]
    tasks: List[TaskConfig]


# -----------------------------
# 重试策略类型
# -----------------------------

class RetryPolicyConfig(TypedDict, total=False):
    """
    重试策略配置
    """
    max_retries: int                   # 最大重试次数
    base_delay: float                  # 基础延迟（秒）
    max_delay: float                   # 最大延迟（秒）
    backoff_factor: float              # 退避因子（默认 2.0）
    jitter: bool                       # 是否启用随机抖动
    retry_on: List[str]                # 可重试的错误类型


# -----------------------------
# 上下文与运行时类型
# -----------------------------

class GenerationContext(TypedDict, total=False):
    """
    生成上下文（由接口传入，供策略使用）
    """
    text: str
    prompt: str
    avatar_video: str
    voice: str
    style: str
    duration: float
    output_format: str
    # 可扩展字段...
    custom: Dict[str, Any]


class ExecutionResult(TypedDict, total=False):
    """
    单个任务执行结果
    """
    task_index: int
    generator_key: str
    input_params: Dict[str, Any]
    result: BaseResult
    elapsed_time: float  # 耗时（秒）
    attempt: int