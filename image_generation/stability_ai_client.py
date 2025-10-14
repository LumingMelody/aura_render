"""
Stability AI (Stable Diffusion) 图像生成客户端
"""
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import json
import base64
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


class StabilityAIClient(BaseImageGenerationClient):
    """Stability AI Stable Diffusion 客户端"""

    def __init__(self, api_key: str, model: str = "stable-diffusion-xl-1024-v1-0"):
        super().__init__(api_key, "https://api.stability.ai/v1", timeout=180)
        self.model = model
        self.rate_limit_delay = 1.0

        # 支持的模型列表
        self.supported_models = {
            "stable-diffusion-xl-1024-v1-0": {
                "max_size": (1024, 1024),
                "supports_negative": True,
                "cost_per_image": 0.04
            },
            "stable-diffusion-v1-6": {
                "max_size": (512, 512),
                "supports_negative": True,
                "cost_per_image": 0.02
            },
            "stable-diffusion-512-v2-1": {
                "max_size": (512, 512),
                "supports_negative": True,
                "cost_per_image": 0.02
            }
        }

        model_config = self.supported_models.get(model, self.supported_models["stable-diffusion-xl-1024-v1-0"])

        # Stability AI 能力配置
        self.capabilities = ModelCapabilities(
            supported_styles=list(ImageStyle),  # 支持所有风格
            supported_sizes=[
                (512, 512), (768, 768), (1024, 1024),
                (512, 768), (768, 512), (1024, 768), (768, 1024)
            ],
            max_images_per_request=10,
            supports_negative_prompt=True,
            supports_seed=True,
            supports_image_reference=True,
            supports_inpainting=True,
            supports_outpainting=True,
            max_prompt_length=2000,
            cost_per_image=model_config["cost_per_image"]
        )

    def _get_auth_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json"
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
        width, height = self._get_stable_diffusion_size(request)

        # 构建API请求
        payload = {
            "text_prompts": [
                {
                    "text": enhanced_prompt,
                    "weight": 1.0
                }
            ],
            "cfg_scale": request.guidance_scale,
            "width": width,
            "height": height,
            "samples": request.num_images,
            "steps": request.num_inference_steps,
            "style_preset": self._map_style_preset(request.style)
        }

        # 添加负面提示词
        if request.negative_prompt:
            payload["text_prompts"].append({
                "text": request.negative_prompt,
                "weight": -1.0
            })

        # 添加种子
        if request.seed:
            payload["seed"] = request.seed

        # 添加采样器设置
        payload["sampler"] = self._get_sampler(request.quality)

        try:
            endpoint = f"/generation/{self.model}/text-to-image"
            response = await self._make_request("POST", endpoint, json=payload)
            processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

            images = []
            for i, artifact in enumerate(response.get("artifacts", [])):
                if artifact.get("finishReason") == "SUCCESS":
                    # 解码base64图像
                    image_data = base64.b64decode(artifact.get("base64", ""))

                    generated_image = GeneratedImage(
                        image_url="",  # Stability AI返回base64数据
                        image_data=image_data,
                        width=width,
                        height=height,
                        format="png",
                        size_bytes=len(image_data),
                        prompt=request.prompt,
                        revised_prompt=enhanced_prompt,
                        seed=artifact.get("seed"),
                        generation_time_ms=processing_time,
                        cost=self.estimate_cost(request) / request.num_images,
                        metadata={
                            "model": self.model,
                            "cfg_scale": request.guidance_scale,
                            "steps": request.num_inference_steps,
                            "style_preset": payload["style_preset"]
                        }
                    )
                    images.append(generated_image)

            return ImageGenerationResponse(
                images=images,
                total_cost=self.estimate_cost(request),
                processing_time_ms=processing_time,
                model_name=self.model,
                provider="stability_ai",
                request_id=response.get("id", "")
            )

        except Exception as e:
            print(f"Stability AI generation failed: {e}")
            return ImageGenerationResponse(
                images=[],
                error_message=str(e)
            )

    async def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        try:
            response = await self._make_request("GET", "/engines/list")
            engines = response.get("engines", [])

            current_engine = None
            for engine in engines:
                if engine.get("id") == self.model:
                    current_engine = engine
                    break

            return {
                "model": self.model,
                "provider": "Stability AI",
                "type": "Text-to-Image Diffusion",
                "engine_info": current_engine,
                "capabilities": {
                    "max_images_per_request": self.capabilities.max_images_per_request,
                    "supported_sizes": [f"{w}x{h}" for w, h in self.capabilities.supported_sizes],
                    "supports_negative_prompts": True,
                    "supports_seeds": True,
                    "supports_img2img": True,
                    "max_prompt_length": self.capabilities.max_prompt_length
                },
                "pricing": {
                    "cost_per_image": self.capabilities.cost_per_image,
                    "currency": "USD"
                }
            }

        except Exception as e:
            return {
                "model": self.model,
                "provider": "Stability AI",
                "error": str(e)
            }

    def _get_stable_diffusion_size(self, request: ImageGenerationRequest) -> Tuple[int, int]:
        """获取Stable Diffusion支持的尺寸"""
        if request.width and request.height:
            # 检查是否是支持的尺寸
            custom_size = (request.width, request.height)
            if custom_size in self.capabilities.supported_sizes:
                return custom_size

            # 调整到最接近的支持尺寸
            return self._adjust_to_supported_size(request.width, request.height)

        # 根据质量和宽高比选择尺寸
        base_size = self._get_base_size_for_quality(request.quality)

        size_map = {
            AspectRatio.SQUARE: (base_size, base_size),
            AspectRatio.LANDSCAPE: (base_size, int(base_size * 0.75)),
            AspectRatio.PORTRAIT: (int(base_size * 0.75), base_size),
            AspectRatio.WIDE: (base_size, int(base_size * 0.5625)),  # 16:9
            AspectRatio.VERTICAL: (int(base_size * 0.5625), base_size)
        }

        return size_map.get(request.aspect_ratio, (base_size, base_size))

    def _get_base_size_for_quality(self, quality: ImageQuality) -> int:
        """根据质量获取基础尺寸"""
        if "xl" in self.model.lower():
            quality_map = {
                ImageQuality.DRAFT: 512,
                ImageQuality.STANDARD: 1024,
                ImageQuality.HIGH: 1024,
                ImageQuality.ULTRA: 1024
            }
        else:
            quality_map = {
                ImageQuality.DRAFT: 512,
                ImageQuality.STANDARD: 512,
                ImageQuality.HIGH: 768,
                ImageQuality.ULTRA: 768
            }

        return quality_map.get(quality, 1024)

    def _adjust_to_supported_size(self, width: int, height: int) -> Tuple[int, int]:
        """调整到支持的尺寸"""
        # 找到最接近的支持尺寸
        min_diff = float('inf')
        best_size = (512, 512)

        for w, h in self.capabilities.supported_sizes:
            diff = abs(w - width) + abs(h - height)
            if diff < min_diff:
                min_diff = diff
                best_size = (w, h)

        return best_size

    def _map_style_preset(self, style: ImageStyle) -> Optional[str]:
        """映射风格预设"""
        style_map = {
            ImageStyle.REALISTIC: "photographic",
            ImageStyle.ARTISTIC: "enhance",
            ImageStyle.ANIME: "anime",
            ImageStyle.CARTOON: "comic-book",
            ImageStyle.DIGITAL_ART: "digital-art",
            ImageStyle.FANTASY: "fantasy-art",
            ImageStyle.CYBERPUNK: "neon-punk",
            ImageStyle.OIL_PAINTING: "analog-film",
            ImageStyle.SKETCH: "line-art"
        }
        return style_map.get(style)

    def _get_sampler(self, quality: ImageQuality) -> str:
        """根据质量选择采样器"""
        sampler_map = {
            ImageQuality.DRAFT: "K_EULER_ANCESTRAL",
            ImageQuality.STANDARD: "K_DPMPP_2M",
            ImageQuality.HIGH: "K_DPMPP_2S_ANCESTRAL",
            ImageQuality.ULTRA: "K_DPMPP_SDE"
        }
        return sampler_map.get(quality, "K_DPMPP_2M")

    async def enhance_prompt(self, prompt: str, style: ImageStyle) -> str:
        """增强Stable Diffusion提示词"""
        # Stable Diffusion风格关键词
        style_keywords = {
            ImageStyle.REALISTIC: "photorealistic, highly detailed, 8k, professional photography",
            ImageStyle.ARTISTIC: "artistic masterpiece, creative composition, award winning",
            ImageStyle.ANIME: "anime style, cel shading, vibrant colors, studio ghibli style",
            ImageStyle.CARTOON: "cartoon illustration, colorful, playful, animated style",
            ImageStyle.DIGITAL_ART: "digital art, concept art, trending on artstation, detailed",
            ImageStyle.PHOTOGRAPHIC: "professional photography, DSLR, bokeh, studio lighting",
            ImageStyle.OIL_PAINTING: "oil painting, traditional art, canvas texture, brush strokes",
            ImageStyle.WATERCOLOR: "watercolor painting, soft colors, flowing paint, artistic",
            ImageStyle.SKETCH: "pencil sketch, line art, detailed drawing, black and white",
            ImageStyle.CYBERPUNK: "cyberpunk style, neon lights, futuristic, high tech, dark",
            ImageStyle.FANTASY: "fantasy art, magical, ethereal, mystical, enchanted",
            ImageStyle.MINIMALIST: "minimalist design, clean, simple, modern, elegant"
        }

        style_suffix = style_keywords.get(style, "")

        # 质量增强关键词
        quality_keywords = [
            "highly detailed",
            "high quality",
            "masterpiece",
            "best quality",
            "ultra detailed"
        ]

        enhanced = prompt
        if style_suffix:
            enhanced = f"{enhanced}, {style_suffix}"

        # 添加质量关键词
        enhanced = f"{enhanced}, {', '.join(quality_keywords[:2])}"

        return enhanced[:self.capabilities.max_prompt_length]

    async def image_to_image(self, init_image_path: str, request: ImageGenerationRequest) -> ImageGenerationResponse:
        """图生图"""
        start_time = asyncio.get_event_loop().time()

        try:
            # 读取初始图像
            with open(init_image_path, "rb") as f:
                init_image_data = f.read()

            init_image_b64 = base64.b64encode(init_image_data).decode('utf-8')

            # 增强提示词
            enhanced_prompt = request.prompt
            if request.enhance_prompt:
                enhanced_prompt = await self.enhance_prompt(request.prompt, request.style)

            # 构建请求
            payload = {
                "text_prompts": [
                    {
                        "text": enhanced_prompt,
                        "weight": 1.0
                    }
                ],
                "init_image": init_image_b64,
                "init_image_mode": "IMAGE_STRENGTH",
                "image_strength": 0.35,  # 0.0-1.0, 较低值保留更多原图
                "cfg_scale": request.guidance_scale,
                "samples": request.num_images,
                "steps": request.num_inference_steps
            }

            if request.negative_prompt:
                payload["text_prompts"].append({
                    "text": request.negative_prompt,
                    "weight": -1.0
                })

            endpoint = f"/generation/{self.model}/image-to-image"
            response = await self._make_request("POST", endpoint, json=payload)
            processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

            images = []
            for artifact in response.get("artifacts", []):
                if artifact.get("finishReason") == "SUCCESS":
                    image_data = base64.b64decode(artifact.get("base64", ""))

                    generated_image = GeneratedImage(
                        image_url="",
                        image_data=image_data,
                        width=request.width or 1024,
                        height=request.height or 1024,
                        format="png",
                        size_bytes=len(image_data),
                        prompt=request.prompt,
                        revised_prompt=enhanced_prompt,
                        seed=artifact.get("seed"),
                        generation_time_ms=processing_time,
                        cost=self.estimate_cost(request) / request.num_images,
                        metadata={
                            "model": self.model,
                            "operation": "image-to-image",
                            "image_strength": 0.35
                        }
                    )
                    images.append(generated_image)

            return ImageGenerationResponse(
                images=images,
                total_cost=self.estimate_cost(request),
                processing_time_ms=processing_time,
                model_name=self.model,
                provider="stability_ai"
            )

        except Exception as e:
            print(f"Image-to-image generation failed: {e}")
            return ImageGenerationResponse(
                images=[],
                error_message=str(e)
            )

    def estimate_cost(self, request: ImageGenerationRequest) -> float:
        """估算Stability AI生成成本"""
        base_cost = self.capabilities.cost_per_image

        # 尺寸系数
        width, height = self._get_stable_diffusion_size(request)
        pixel_count = width * height

        if pixel_count > 1024 * 1024:
            base_cost *= 1.5
        elif pixel_count > 512 * 512:
            base_cost *= 1.2

        # 步数系数
        if request.num_inference_steps > 50:
            base_cost *= 1.2

        return base_cost * request.num_images

    async def get_available_engines(self) -> List[Dict[str, Any]]:
        """获取可用引擎列表"""
        try:
            response = await self._make_request("GET", "/engines/list")
            return response.get("engines", [])
        except Exception as e:
            print(f"Failed to get engines: {e}")
            return []