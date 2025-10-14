"""
图像生成服务基础客户端 - 统一接口定义
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import aiohttp
import time
import base64


class ImageStyle(Enum):
    """图像风格"""
    REALISTIC = "realistic"
    ARTISTIC = "artistic"
    ANIME = "anime"
    CARTOON = "cartoon"
    PHOTOGRAPHIC = "photographic"
    DIGITAL_ART = "digital_art"
    OIL_PAINTING = "oil_painting"
    WATERCOLOR = "watercolor"
    SKETCH = "sketch"
    CYBERPUNK = "cyberpunk"
    FANTASY = "fantasy"
    MINIMALIST = "minimalist"


class ImageQuality(Enum):
    """图像质量"""
    DRAFT = "draft"
    STANDARD = "standard"
    HIGH = "high"
    ULTRA = "ultra"


class AspectRatio(Enum):
    """图像宽高比"""
    SQUARE = "1:1"
    PORTRAIT = "2:3"
    LANDSCAPE = "3:2"
    WIDE = "16:9"
    ULTRAWIDE = "21:9"
    VERTICAL = "9:16"
    CUSTOM = "custom"


@dataclass
class ImageGenerationRequest:
    """图像生成请求"""
    prompt: str
    negative_prompt: Optional[str] = None
    style: ImageStyle = ImageStyle.REALISTIC
    quality: ImageQuality = ImageQuality.STANDARD
    aspect_ratio: AspectRatio = AspectRatio.SQUARE
    width: Optional[int] = None
    height: Optional[int] = None
    num_images: int = 1
    seed: Optional[int] = None
    guidance_scale: float = 7.5  # CFG scale for diffusion models
    num_inference_steps: int = 50  # Steps for diffusion models
    enhance_prompt: bool = True  # 是否增强提示词
    style_reference: Optional[str] = None  # 参考图像URL
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GeneratedImage:
    """生成的图像"""
    image_url: str
    image_data: Optional[bytes] = None
    width: int = 0
    height: int = 0
    format: str = "png"
    size_bytes: int = 0
    prompt: str = ""
    revised_prompt: str = ""  # 增强后的提示词
    seed: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    generation_time_ms: int = 0
    cost: float = 0.0


@dataclass
class ImageGenerationResponse:
    """图像生成响应"""
    images: List[GeneratedImage]
    total_cost: float = 0.0
    processing_time_ms: int = 0
    model_name: str = ""
    provider: str = ""
    request_id: str = ""
    error_message: Optional[str] = None


@dataclass
class ModelCapabilities:
    """模型能力"""
    supported_styles: List[ImageStyle]
    supported_sizes: List[Tuple[int, int]]
    max_images_per_request: int
    supports_negative_prompt: bool
    supports_seed: bool
    supports_image_reference: bool
    supports_inpainting: bool
    supports_outpainting: bool
    max_prompt_length: int
    cost_per_image: float  # 基础成本


class BaseImageGenerationClient(ABC):
    """图像生成服务基础客户端"""

    def __init__(self, api_key: str, base_url: str, timeout: int = 120):
        self.api_key = api_key
        self.base_url = base_url
        self.timeout = timeout
        self.last_request_time = 0
        self.rate_limit_delay = 1.0
        self.session: Optional[aiohttp.ClientSession] = None

        # 默认能力
        self.capabilities = ModelCapabilities(
            supported_styles=list(ImageStyle),
            supported_sizes=[(512, 512), (1024, 1024)],
            max_images_per_request=4,
            supports_negative_prompt=True,
            supports_seed=True,
            supports_image_reference=False,
            supports_inpainting=False,
            supports_outpainting=False,
            max_prompt_length=1000,
            cost_per_image=0.01
        )

    async def __aenter__(self):
        self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()

    @abstractmethod
    async def generate_images(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """生成图像"""
        pass

    @abstractmethod
    async def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        pass

    async def enhance_prompt(self, prompt: str, style: ImageStyle) -> str:
        """增强提示词"""
        # 基础实现，子类可以覆盖
        style_keywords = {
            ImageStyle.REALISTIC: "highly detailed, photorealistic, professional photography",
            ImageStyle.ARTISTIC: "artistic, creative, expressive, masterpiece",
            ImageStyle.ANIME: "anime style, manga, japanese animation",
            ImageStyle.CARTOON: "cartoon style, animated, colorful, playful",
            ImageStyle.PHOTOGRAPHIC: "photograph, DSLR, bokeh, professional lighting",
            ImageStyle.DIGITAL_ART: "digital art, concept art, highly detailed",
            ImageStyle.OIL_PAINTING: "oil painting, traditional art, brush strokes",
            ImageStyle.WATERCOLOR: "watercolor painting, soft colors, fluid art",
            ImageStyle.SKETCH: "pencil sketch, line art, black and white drawing",
            ImageStyle.CYBERPUNK: "cyberpunk, neon lights, futuristic, high tech",
            ImageStyle.FANTASY: "fantasy art, magical, ethereal, enchanted",
            ImageStyle.MINIMALIST: "minimalist, simple, clean, modern design"
        }

        style_suffix = style_keywords.get(style, "")
        if style_suffix:
            enhanced = f"{prompt}, {style_suffix}"
        else:
            enhanced = prompt

        return enhanced[:self.capabilities.max_prompt_length]

    async def batch_generate(self, requests: List[ImageGenerationRequest]) -> List[ImageGenerationResponse]:
        """批量生成图像"""
        tasks = []
        for request in requests:
            task = self.generate_images(request)
            tasks.append(task)

        # 控制并发数量
        semaphore = asyncio.Semaphore(3)  # 最多3个并发请求

        async def limited_generate(request):
            async with semaphore:
                return await self.generate_images(request)

        return await asyncio.gather(*[limited_generate(req) for req in requests])

    async def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """发送HTTP请求"""
        if not self.session:
            self.session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=self.timeout))

        # 速率限制
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.rate_limit_delay:
            await asyncio.sleep(self.rate_limit_delay - time_since_last)

        url = f"{self.base_url.rstrip('/')}/{endpoint.lstrip('/')}"
        headers = self._get_auth_headers()

        if 'headers' in kwargs:
            headers.update(kwargs.pop('headers'))

        try:
            async with self.session.request(method, url, headers=headers, **kwargs) as response:
                self.last_request_time = time.time()

                if response.status >= 400:
                    error_text = await response.text()
                    raise Exception(f"API error {response.status}: {error_text}")

                # 根据content-type处理响应
                content_type = response.headers.get('content-type', '')

                if 'application/json' in content_type:
                    return await response.json()
                elif 'image/' in content_type:
                    return {"image_data": await response.read()}
                else:
                    return {"text": await response.text()}

        except Exception as e:
            print(f"Image generation request failed: {e}")
            raise

    @abstractmethod
    def _get_auth_headers(self) -> Dict[str, str]:
        """获取认证头"""
        pass

    def calculate_size(self, aspect_ratio: AspectRatio, quality: ImageQuality) -> Tuple[int, int]:
        """根据宽高比和质量计算图像尺寸"""
        # 基础尺寸映射
        quality_base = {
            ImageQuality.DRAFT: 512,
            ImageQuality.STANDARD: 1024,
            ImageQuality.HIGH: 1536,
            ImageQuality.ULTRA: 2048
        }

        base_size = quality_base.get(quality, 1024)

        # 宽高比映射
        ratio_map = {
            AspectRatio.SQUARE: (base_size, base_size),
            AspectRatio.PORTRAIT: (int(base_size * 2/3), base_size),
            AspectRatio.LANDSCAPE: (base_size, int(base_size * 2/3)),
            AspectRatio.WIDE: (base_size, int(base_size * 9/16)),
            AspectRatio.ULTRAWIDE: (base_size, int(base_size * 9/21)),
            AspectRatio.VERTICAL: (int(base_size * 9/16), base_size)
        }

        return ratio_map.get(aspect_ratio, (base_size, base_size))

    def estimate_cost(self, request: ImageGenerationRequest) -> float:
        """估算生成成本"""
        base_cost = self.capabilities.cost_per_image

        # 质量系数
        quality_multiplier = {
            ImageQuality.DRAFT: 0.5,
            ImageQuality.STANDARD: 1.0,
            ImageQuality.HIGH: 1.5,
            ImageQuality.ULTRA: 2.0
        }

        cost = base_cost * quality_multiplier.get(request.quality, 1.0)
        cost *= request.num_images

        return cost

    async def validate_request(self, request: ImageGenerationRequest) -> Dict[str, Any]:
        """验证请求参数"""
        result = {
            "valid": True,
            "warnings": [],
            "errors": []
        }

        # 检查提示词长度
        if len(request.prompt) > self.capabilities.max_prompt_length:
            result["errors"].append(f"Prompt exceeds maximum length of {self.capabilities.max_prompt_length}")
            result["valid"] = False

        # 检查图像数量
        if request.num_images > self.capabilities.max_images_per_request:
            result["errors"].append(f"Number of images exceeds maximum of {self.capabilities.max_images_per_request}")
            result["valid"] = False

        # 检查尺寸
        if request.width and request.height:
            size_tuple = (request.width, request.height)
            if size_tuple not in self.capabilities.supported_sizes:
                result["warnings"].append("Custom size may not be supported, will use closest supported size")

        # 检查风格支持
        if request.style not in self.capabilities.supported_styles:
            result["warnings"].append(f"Style {request.style.value} may not be fully supported")

        return result

    async def download_image(self, image_url: str) -> bytes:
        """下载图像"""
        if not self.session:
            self.session = aiohttp.ClientSession()

        try:
            async with self.session.get(image_url) as response:
                if response.status == 200:
                    return await response.read()
                else:
                    raise Exception(f"Failed to download image: HTTP {response.status}")
        except Exception as e:
            print(f"Image download failed: {e}")
            raise

    def image_to_base64(self, image_data: bytes) -> str:
        """将图像数据转换为base64"""
        return base64.b64encode(image_data).decode('utf-8')

    def base64_to_image(self, base64_str: str) -> bytes:
        """将base64转换为图像数据"""
        return base64.b64decode(base64_str)

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            model_info = await self.get_model_info()
            return bool(model_info)
        except Exception:
            return False

    async def get_capabilities(self) -> ModelCapabilities:
        """获取模型能力"""
        return self.capabilities