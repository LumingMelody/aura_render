"""
OpenAI DALL-E 图像生成客户端
"""
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import json
from .base_image_client import (
    BaseImageGenerationClient,
    ImageGenerationRequest,
    ImageGenerationResponse,
    GeneratedImage,
    ImageStyle,
    ImageQuality,
    AspectRatio,
    ModelCapabilities
)


class OpenAIDALLEClient(BaseImageGenerationClient):
    """OpenAI DALL-E 客户端"""

    def __init__(self, api_key: str, model: str = "dall-e-3"):
        super().__init__(api_key, "https://api.openai.com/v1", timeout=120)
        self.model = model  # dall-e-2 或 dall-e-3
        self.rate_limit_delay = 2.0  # DALL-E有较严格的速率限制

        # DALL-E 能力配置
        if model == "dall-e-3":
            self.capabilities = ModelCapabilities(
                supported_styles=[
                    ImageStyle.REALISTIC, ImageStyle.ARTISTIC, ImageStyle.DIGITAL_ART,
                    ImageStyle.PHOTOGRAPHIC, ImageStyle.CARTOON, ImageStyle.FANTASY
                ],
                supported_sizes=[(1024, 1024), (1792, 1024), (1024, 1792)],
                max_images_per_request=1,  # DALL-E 3每次只能生成1张
                supports_negative_prompt=False,
                supports_seed=False,
                supports_image_reference=False,
                supports_inpainting=False,
                supports_outpainting=False,
                max_prompt_length=4000,
                cost_per_image=0.040  # $0.040 for 1024×1024
            )
        else:  # dall-e-2
            self.capabilities = ModelCapabilities(
                supported_styles=[
                    ImageStyle.REALISTIC, ImageStyle.ARTISTIC, ImageStyle.DIGITAL_ART
                ],
                supported_sizes=[(256, 256), (512, 512), (1024, 1024)],
                max_images_per_request=10,
                supports_negative_prompt=False,
                supports_seed=False,
                supports_image_reference=False,
                supports_inpainting=True,  # DALL-E 2支持编辑
                supports_outpainting=False,
                max_prompt_length=1000,
                cost_per_image=0.020  # $0.020 for 1024×1024
            )

    def _get_auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

    async def generate_images(self, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """生成图像"""
        start_time = asyncio.get_event_loop().time()

        # 验证请求
        validation = await self.validate_request(request)
        if not validation["valid"]:
            return ImageGenerationResponse(
                images=[],
                error_message="; ".join(validation["errors"])
            )

        # 增强提示词
        enhanced_prompt = request.prompt
        if request.enhance_prompt:
            enhanced_prompt = await self.enhance_prompt(request.prompt, request.style)

        # 确定图像尺寸
        width, height = self._get_dalle_size(request)

        # 构建API请求
        payload = {
            "model": self.model,
            "prompt": enhanced_prompt,
            "size": f"{width}x{height}",
            "quality": self._map_quality(request.quality),
            "n": min(request.num_images, self.capabilities.max_images_per_request),
            "response_format": "url"  # 或 "b64_json"
        }

        # DALL-E 3 特有参数
        if self.model == "dall-e-3":
            payload["style"] = self._map_style(request.style)

        try:
            response = await self._make_request("POST", "/images/generations", json=payload)
            processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

            images = []
            for i, image_data in enumerate(response.get("data", [])):
                image_url = image_data.get("url", "")
                revised_prompt = image_data.get("revised_prompt", enhanced_prompt)

                # 下载图像数据（可选）
                image_bytes = None
                if image_url:
                    try:
                        image_bytes = await self.download_image(image_url)
                    except Exception as e:
                        print(f"Failed to download image: {e}")

                generated_image = GeneratedImage(
                    image_url=image_url,
                    image_data=image_bytes,
                    width=width,
                    height=height,
                    format="png",
                    size_bytes=len(image_bytes) if image_bytes else 0,
                    prompt=request.prompt,
                    revised_prompt=revised_prompt,
                    generation_time_ms=processing_time,
                    cost=self.estimate_cost(request) / request.num_images,
                    metadata={
                        "model": self.model,
                        "quality": request.quality.value,
                        "style": request.style.value
                    }
                )
                images.append(generated_image)

            return ImageGenerationResponse(
                images=images,
                total_cost=self.estimate_cost(request),
                processing_time_ms=processing_time,
                model_name=self.model,
                provider="openai",
                request_id=response.get("id", "")
            )

        except Exception as e:
            print(f"DALL-E generation failed: {e}")
            return ImageGenerationResponse(
                images=[],
                error_message=str(e)
            )

    async def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model": self.model,
            "provider": "OpenAI",
            "type": "Text-to-Image",
            "capabilities": {
                "max_images_per_request": self.capabilities.max_images_per_request,
                "supported_sizes": [f"{w}x{h}" for w, h in self.capabilities.supported_sizes],
                "supports_styles": self.model == "dall-e-3",
                "supports_quality_settings": self.model == "dall-e-3",
                "max_prompt_length": self.capabilities.max_prompt_length
            },
            "pricing": {
                "cost_per_image": self.capabilities.cost_per_image,
                "currency": "USD"
            }
        }

    def _get_dalle_size(self, request: ImageGenerationRequest) -> Tuple[int, int]:
        """获取DALL-E支持的尺寸"""
        if request.width and request.height:
            # 检查是否是支持的尺寸
            custom_size = (request.width, request.height)
            if custom_size in self.capabilities.supported_sizes:
                return custom_size

        # 根据宽高比选择最佳尺寸
        if self.model == "dall-e-3":
            size_map = {
                AspectRatio.SQUARE: (1024, 1024),
                AspectRatio.LANDSCAPE: (1792, 1024),
                AspectRatio.PORTRAIT: (1024, 1792),
                AspectRatio.WIDE: (1792, 1024),
                AspectRatio.VERTICAL: (1024, 1792)
            }
        else:  # dall-e-2
            size_map = {
                AspectRatio.SQUARE: (1024, 1024),
                AspectRatio.LANDSCAPE: (1024, 1024),  # DALL-E 2只支持正方形
                AspectRatio.PORTRAIT: (1024, 1024),
                AspectRatio.WIDE: (1024, 1024),
                AspectRatio.VERTICAL: (1024, 1024)
            }

        return size_map.get(request.aspect_ratio, (1024, 1024))

    def _map_quality(self, quality: ImageQuality) -> str:
        """映射质量设置"""
        if self.model == "dall-e-3":
            quality_map = {
                ImageQuality.DRAFT: "standard",
                ImageQuality.STANDARD: "standard",
                ImageQuality.HIGH: "hd",
                ImageQuality.ULTRA: "hd"
            }
            return quality_map.get(quality, "standard")
        else:
            return "standard"  # DALL-E 2只有标准质量

    def _map_style(self, style: ImageStyle) -> str:
        """映射风格设置（DALL-E 3）"""
        style_map = {
            ImageStyle.REALISTIC: "natural",
            ImageStyle.ARTISTIC: "vivid",
            ImageStyle.DIGITAL_ART: "vivid",
            ImageStyle.PHOTOGRAPHIC: "natural",
            ImageStyle.CARTOON: "vivid",
            ImageStyle.FANTASY: "vivid"
        }
        return style_map.get(style, "natural")

    async def enhance_prompt(self, prompt: str, style: ImageStyle) -> str:
        """增强DALL-E提示词"""
        # DALL-E 3会自动增强提示词，所以我们只做轻微调整
        style_prefixes = {
            ImageStyle.REALISTIC: "A realistic",
            ImageStyle.ARTISTIC: "An artistic",
            ImageStyle.DIGITAL_ART: "A digital art piece of",
            ImageStyle.PHOTOGRAPHIC: "A professional photograph of",
            ImageStyle.CARTOON: "A cartoon illustration of",
            ImageStyle.FANTASY: "A fantasy art depicting"
        }

        prefix = style_prefixes.get(style, "")
        if prefix and not prompt.lower().startswith(("a ", "an ", "the ")):
            enhanced = f"{prefix} {prompt.lower()}"
        else:
            enhanced = prompt

        # 添加质量关键词
        quality_suffixes = [
            "highly detailed",
            "professional quality",
            "masterpiece"
        ]

        # 随机选择一个质量后缀
        import random
        quality_suffix = random.choice(quality_suffixes)
        enhanced = f"{enhanced}, {quality_suffix}"

        return enhanced[:self.capabilities.max_prompt_length]

    async def edit_image(self, image_path: str, mask_path: str, prompt: str) -> ImageGenerationResponse:
        """编辑图像（仅DALL-E 2支持）"""
        if self.model != "dall-e-2":
            return ImageGenerationResponse(
                images=[],
                error_message="Image editing is only supported by DALL-E 2"
            )

        start_time = asyncio.get_event_loop().time()

        try:
            # 读取图像文件
            with open(image_path, "rb") as image_file:
                image_data = image_file.read()

            with open(mask_path, "rb") as mask_file:
                mask_data = mask_file.read()

            # 构建multipart form data
            data = aiohttp.FormData()
            data.add_field('image', image_data, filename='image.png', content_type='image/png')
            data.add_field('mask', mask_data, filename='mask.png', content_type='image/png')
            data.add_field('prompt', prompt)
            data.add_field('n', '1')
            data.add_field('size', '1024x1024')
            data.add_field('response_format', 'url')

            # 发送编辑请求
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = await self._make_request("POST", "/images/edits", data=data, headers=headers)

            processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

            images = []
            for image_data in response.get("data", []):
                image_url = image_data.get("url", "")

                generated_image = GeneratedImage(
                    image_url=image_url,
                    width=1024,
                    height=1024,
                    format="png",
                    prompt=prompt,
                    generation_time_ms=processing_time,
                    cost=0.020,  # DALL-E 2编辑成本
                    metadata={"model": self.model, "operation": "edit"}
                )
                images.append(generated_image)

            return ImageGenerationResponse(
                images=images,
                total_cost=0.020,
                processing_time_ms=processing_time,
                model_name=self.model,
                provider="openai"
            )

        except Exception as e:
            print(f"DALL-E image editing failed: {e}")
            return ImageGenerationResponse(
                images=[],
                error_message=str(e)
            )

    def estimate_cost(self, request: ImageGenerationRequest) -> float:
        """估算DALL-E生成成本"""
        base_cost = self.capabilities.cost_per_image

        # DALL-E 3的HD质量更贵
        if self.model == "dall-e-3" and request.quality in [ImageQuality.HIGH, ImageQuality.ULTRA]:
            base_cost = 0.080  # HD质量价格

        # 大尺寸图像更贵
        width, height = self._get_dalle_size(request)
        if width * height > 1024 * 1024:
            base_cost *= 1.5

        return base_cost * request.num_images

    async def get_usage_statistics(self) -> Dict[str, Any]:
        """获取使用统计（需要额外的API调用）"""
        # OpenAI不直接提供使用统计API，这里返回基础信息
        return {
            "model": self.model,
            "requests_made": 0,  # 需要自己记录
            "total_images_generated": 0,  # 需要自己记录
            "estimated_total_cost": 0.0  # 需要自己计算
        }