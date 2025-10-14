"""
Midjourney 图像生成客户端（通过第三方API或Discord Bot）
注意：这是一个示例实现，实际使用需要集成Midjourney的官方API或第三方服务
"""
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import json
import uuid
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


class MidjourneyClient(BaseImageGenerationClient):
    """Midjourney 客户端（通过第三方API）"""

    def __init__(self, api_key: str, proxy_url: str = "https://api.midjourney.com/v1"):
        """
        初始化Midjourney客户端

        Args:
            api_key: API密钥（从第三方服务获取）
            proxy_url: 代理服务URL（如midjourneyapi.io等）
        """
        super().__init__(api_key, proxy_url, timeout=300)  # MJ生成时间较长
        self.rate_limit_delay = 5.0  # MJ有严格的速率限制

        # Midjourney 能力配置
        self.capabilities = ModelCapabilities(
            supported_styles=list(ImageStyle),  # MJ支持所有风格
            supported_sizes=[
                (1024, 1024), (1024, 1456), (1456, 1024),  # MJ v6标准尺寸
                (1344, 768), (768, 1344)  # 宽屏和竖屏
            ],
            max_images_per_request=4,  # MJ一次生成4张
            supports_negative_prompt=True,
            supports_seed=True,
            supports_image_reference=True,
            supports_inpainting=False,
            supports_outpainting=True,  # MJ的zoom out功能
            max_prompt_length=4000,
            cost_per_image=0.10  # 估算成本，实际根据服务商定价
        )

        # Midjourney 版本
        self.version = "6"  # 默认使用v6

        # 参数映射
        self.style_mapping = {
            ImageStyle.REALISTIC: "--style raw",
            ImageStyle.ARTISTIC: "--style expressive",
            ImageStyle.ANIME: "--niji 6",
            ImageStyle.CARTOON: "--style cute",
            ImageStyle.PHOTOGRAPHIC: "--style raw --s 250",
            ImageStyle.DIGITAL_ART: "--style expressive --s 400",
            ImageStyle.OIL_PAINTING: "oil painting style",
            ImageStyle.WATERCOLOR: "watercolor painting",
            ImageStyle.SKETCH: "pencil sketch --style raw",
            ImageStyle.CYBERPUNK: "cyberpunk --style expressive --s 600",
            ImageStyle.FANTASY: "fantasy art --style expressive",
            ImageStyle.MINIMALIST: "minimalist --style raw --s 50"
        }

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

        # 构建Midjourney提示词
        mj_prompt = await self._build_midjourney_prompt(request)

        # 生成任务ID
        task_id = str(uuid.uuid4())

        try:
            # 提交生成任务
            submit_payload = {
                "type": "imagine",
                "prompt": mj_prompt,
                "webhook_url": None,  # 可选的webhook回调
                "webhook_secret": None
            }

            submit_response = await self._make_request("POST", "/submit/imagine", json=submit_payload)

            if submit_response.get("success"):
                job_id = submit_response.get("job_id")

                # 轮询等待完成
                images = await self._wait_for_completion(job_id, start_time)
                processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

                return ImageGenerationResponse(
                    images=images,
                    total_cost=self.estimate_cost(request),
                    processing_time_ms=processing_time,
                    model_name=f"midjourney-v{self.version}",
                    provider="midjourney",
                    request_id=job_id
                )
            else:
                error_msg = submit_response.get("error", "Unknown submission error")
                return ImageGenerationResponse(
                    images=[],
                    error_message=error_msg
                )

        except Exception as e:
            print(f"Midjourney generation failed: {e}")
            return ImageGenerationResponse(
                images=[],
                error_message=str(e)
            )

    async def _build_midjourney_prompt(self, request: ImageGenerationRequest) -> str:
        """构建Midjourney提示词"""
        prompt_parts = []

        # 基础提示词
        base_prompt = request.prompt
        if request.enhance_prompt:
            base_prompt = await self.enhance_prompt(request.prompt, request.style)

        prompt_parts.append(base_prompt)

        # 添加风格参数
        style_param = self.style_mapping.get(request.style, "")
        if style_param:
            prompt_parts.append(style_param)

        # 添加质量参数
        quality_param = self._get_quality_param(request.quality)
        if quality_param:
            prompt_parts.append(quality_param)

        # 添加宽高比
        aspect_ratio = self._get_aspect_ratio_param(request.aspect_ratio, request.width, request.height)
        if aspect_ratio:
            prompt_parts.append(aspect_ratio)

        # 添加版本参数
        prompt_parts.append(f"--v {self.version}")

        # 添加种子
        if request.seed:
            prompt_parts.append(f"--seed {request.seed}")

        # 添加负面提示词（通过--no参数）
        if request.negative_prompt:
            no_params = self._convert_negative_prompt(request.negative_prompt)
            if no_params:
                prompt_parts.append(no_params)

        return " ".join(prompt_parts)

    def _get_quality_param(self, quality: ImageQuality) -> str:
        """获取质量参数"""
        quality_map = {
            ImageQuality.DRAFT: "--q 0.25",
            ImageQuality.STANDARD: "--q 1",
            ImageQuality.HIGH: "--q 2",
            ImageQuality.ULTRA: "--q 2 --s 750"  # 高质量+高风格化
        }
        return quality_map.get(quality, "--q 1")

    def _get_aspect_ratio_param(self, aspect_ratio: AspectRatio, width: Optional[int], height: Optional[int]) -> str:
        """获取宽高比参数"""
        if width and height:
            # 计算最简比例
            from math import gcd
            ratio_gcd = gcd(width, height)
            w_ratio = width // ratio_gcd
            h_ratio = height // ratio_gcd
            return f"--ar {w_ratio}:{h_ratio}"

        ratio_map = {
            AspectRatio.SQUARE: "--ar 1:1",
            AspectRatio.PORTRAIT: "--ar 2:3",
            AspectRatio.LANDSCAPE: "--ar 3:2",
            AspectRatio.WIDE: "--ar 16:9",
            AspectRatio.ULTRAWIDE: "--ar 21:9",
            AspectRatio.VERTICAL: "--ar 9:16"
        }
        return ratio_map.get(aspect_ratio, "")

    def _convert_negative_prompt(self, negative_prompt: str) -> str:
        """将负面提示词转换为--no参数"""
        # 简单处理：将负面提示词转换为--no参数
        negative_terms = [term.strip() for term in negative_prompt.split(',')]
        return f"--no {', '.join(negative_terms)}"

    async def _wait_for_completion(self, job_id: str, start_time: float) -> List[GeneratedImage]:
        """等待生成完成"""
        max_wait_time = 300  # 最大等待5分钟
        check_interval = 10  # 每10秒检查一次

        while True:
            current_time = asyncio.get_event_loop().time()
            if current_time - start_time > max_wait_time:
                raise Exception("Generation timeout")

            try:
                # 检查任务状态
                status_response = await self._make_request("GET", f"/status/{job_id}")

                status = status_response.get("status")
                if status == "completed":
                    # 解析结果
                    return await self._parse_completion_result(status_response)
                elif status == "failed":
                    error_msg = status_response.get("error", "Generation failed")
                    raise Exception(error_msg)
                elif status in ["queued", "processing"]:
                    # 继续等待
                    await asyncio.sleep(check_interval)
                    continue
                else:
                    # 未知状态
                    await asyncio.sleep(check_interval)
                    continue

            except Exception as e:
                if "timeout" in str(e).lower():
                    raise
                # 其他错误，继续重试
                await asyncio.sleep(check_interval)
                continue

    async def _parse_completion_result(self, result_data: Dict) -> List[GeneratedImage]:
        """解析完成结果"""
        images = []
        image_urls = result_data.get("image_urls", [])

        for i, image_url in enumerate(image_urls):
            try:
                # 下载图像数据
                image_data = await self.download_image(image_url)

                generated_image = GeneratedImage(
                    image_url=image_url,
                    image_data=image_data,
                    width=1024,  # MJ默认尺寸
                    height=1024,
                    format="png",
                    size_bytes=len(image_data) if image_data else 0,
                    prompt=result_data.get("prompt", ""),
                    generation_time_ms=result_data.get("processing_time_ms", 0),
                    cost=self.capabilities.cost_per_image,
                    metadata={
                        "model": f"midjourney-v{self.version}",
                        "grid_position": i + 1,  # MJ生成4张网格图
                        "job_id": result_data.get("job_id", "")
                    }
                )
                images.append(generated_image)

            except Exception as e:
                print(f"Failed to process image {i}: {e}")
                continue

        return images

    async def upscale_image(self, job_id: str, index: int) -> ImageGenerationResponse:
        """放大指定位置的图像"""
        start_time = asyncio.get_event_loop().time()

        try:
            upscale_payload = {
                "type": "upscale",
                "job_id": job_id,
                "index": index  # 1-4
            }

            submit_response = await self._make_request("POST", "/submit/upscale", json=upscale_payload)

            if submit_response.get("success"):
                upscale_job_id = submit_response.get("job_id")
                images = await self._wait_for_completion(upscale_job_id, start_time)
                processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

                return ImageGenerationResponse(
                    images=images,
                    total_cost=self.capabilities.cost_per_image * 0.5,  # 放大成本较低
                    processing_time_ms=processing_time,
                    model_name=f"midjourney-v{self.version}",
                    provider="midjourney",
                    request_id=upscale_job_id
                )
            else:
                error_msg = submit_response.get("error", "Upscale submission failed")
                return ImageGenerationResponse(
                    images=[],
                    error_message=error_msg
                )

        except Exception as e:
            print(f"Midjourney upscale failed: {e}")
            return ImageGenerationResponse(
                images=[],
                error_message=str(e)
            )

    async def vary_image(self, job_id: str, index: int, variation_type: str = "strong") -> ImageGenerationResponse:
        """生成变体图像"""
        start_time = asyncio.get_event_loop().time()

        try:
            vary_payload = {
                "type": "vary",
                "job_id": job_id,
                "index": index,
                "variation_type": variation_type  # "strong", "subtle"
            }

            submit_response = await self._make_request("POST", "/submit/vary", json=vary_payload)

            if submit_response.get("success"):
                vary_job_id = submit_response.get("job_id")
                images = await self._wait_for_completion(vary_job_id, start_time)
                processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)

                return ImageGenerationResponse(
                    images=images,
                    total_cost=self.capabilities.cost_per_image,
                    processing_time_ms=processing_time,
                    model_name=f"midjourney-v{self.version}",
                    provider="midjourney",
                    request_id=vary_job_id
                )
            else:
                error_msg = submit_response.get("error", "Variation submission failed")
                return ImageGenerationResponse(
                    images=[],
                    error_message=error_msg
                )

        except Exception as e:
            print(f"Midjourney variation failed: {e}")
            return ImageGenerationResponse(
                images=[],
                error_message=str(e)
            )

    async def get_model_info(self) -> Dict[str, Any]:
        """获取模型信息"""
        return {
            "model": f"midjourney-v{self.version}",
            "provider": "Midjourney",
            "type": "Text-to-Image Diffusion",
            "capabilities": {
                "max_images_per_request": 4,  # 生成4张网格图
                "supported_aspect_ratios": ["1:1", "2:3", "3:2", "16:9", "9:16"],
                "supports_upscaling": True,
                "supports_variations": True,
                "supports_style_parameters": True,
                "supports_seeds": True,
                "max_prompt_length": self.capabilities.max_prompt_length
            },
            "pricing": {
                "cost_per_generation": self.capabilities.cost_per_image * 4,  # 4张图的成本
                "cost_per_upscale": self.capabilities.cost_per_image * 0.5,
                "currency": "USD"
            },
            "generation_time": {
                "average_seconds": 60,
                "max_seconds": 300
            }
        }

    async def enhance_prompt(self, prompt: str, style: ImageStyle) -> str:
        """增强Midjourney提示词"""
        # Midjourney风格关键词
        style_keywords = {
            ImageStyle.REALISTIC: "photorealistic, hyperrealistic, ultra detailed, 8k resolution",
            ImageStyle.ARTISTIC: "artistic masterpiece, fine art, museum quality, award winning",
            ImageStyle.ANIME: "anime style, manga art, japanese animation, cel shading",
            ImageStyle.CARTOON: "cartoon illustration, animated style, vibrant colors",
            ImageStyle.DIGITAL_ART: "digital art, concept art, trending on artstation",
            ImageStyle.PHOTOGRAPHIC: "professional photography, DSLR quality, studio lighting",
            ImageStyle.OIL_PAINTING: "oil painting, traditional art, canvas texture, impasto",
            ImageStyle.WATERCOLOR: "watercolor painting, flowing colors, paper texture",
            ImageStyle.SKETCH: "detailed sketch, line art, pencil drawing",
            ImageStyle.CYBERPUNK: "cyberpunk aesthetic, neon lights, futuristic, dark atmosphere",
            ImageStyle.FANTASY: "fantasy art, magical, mystical, enchanted world",
            ImageStyle.MINIMALIST: "minimalist design, clean composition, simple elegance"
        }

        style_suffix = style_keywords.get(style, "")

        if style_suffix:
            enhanced = f"{prompt}, {style_suffix}"
        else:
            enhanced = prompt

        # 添加通用质量增强词
        quality_boost = "highly detailed, best quality, masterpiece"
        enhanced = f"{enhanced}, {quality_boost}"

        return enhanced[:self.capabilities.max_prompt_length]

    def estimate_cost(self, request: ImageGenerationRequest) -> float:
        """估算Midjourney生成成本"""
        # MJ按次计费，每次生成4张图
        base_cost = self.capabilities.cost_per_image * 4

        # 高质量设置可能增加成本
        if request.quality in [ImageQuality.HIGH, ImageQuality.ULTRA]:
            base_cost *= 1.2

        return base_cost

    async def get_job_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取任务历史"""
        try:
            response = await self._make_request("GET", f"/history?limit={limit}")
            return response.get("jobs", [])
        except Exception as e:
            print(f"Failed to get job history: {e}")
            return []