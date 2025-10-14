"""
图像生成服务管理器 - 统一管理多个图像生成服务提供商
"""
from typing import Dict, List, Any, Optional, Union, Tuple
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
import os
import tempfile
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
from .openai_dalle_client import OpenAIDALLEClient
from .stability_ai_client import StabilityAIClient
from .midjourney_client import MidjourneyClient


class ImageProvider(Enum):
    DALLE = "dalle"
    STABILITY_AI = "stability_ai"
    MIDJOURNEY = "midjourney"


@dataclass
class ImageProviderConfig:
    """图像生成服务配置"""
    provider: ImageProvider
    api_key: str
    model: Optional[str] = None
    proxy_url: Optional[str] = None
    enabled: bool = True
    priority: int = 1  # 1=最高优先级
    cost_weight: float = 1.0  # 成本权重
    quality_weight: float = 1.0  # 质量权重
    speed_weight: float = 1.0  # 速度权重


@dataclass
class GenerationCriteria:
    """图像生成选择标准"""
    preferred_style: Optional[ImageStyle] = None
    required_quality: ImageQuality = ImageQuality.STANDARD
    max_cost_per_image: Optional[float] = None
    max_generation_time: Optional[int] = None  # 秒
    require_batch_support: bool = False
    require_negative_prompt: bool = False
    require_seed_support: bool = False


class ImageGenerationManager:
    """图像生成服务管理器"""

    def __init__(self):
        self.clients: Dict[ImageProvider, BaseImageGenerationClient] = {}
        self.configs: Dict[ImageProvider, ImageProviderConfig] = {}
        self.temp_dir = tempfile.mkdtemp(prefix="image_generation_")

        # 性能统计
        self.generation_stats = {
            provider: {
                "total_requests": 0,
                "successful_requests": 0,
                "total_cost": 0.0,
                "total_time_ms": 0,
                "average_quality_score": 0.0
            } for provider in ImageProvider
        }

    async def register_provider(self, config: ImageProviderConfig):
        """注册图像生成服务提供商"""
        if not config.enabled:
            return

        try:
            if config.provider == ImageProvider.DALLE:
                if not config.api_key:
                    raise ValueError("DALL-E requires API key")
                client = OpenAIDALLEClient(config.api_key, config.model or "dall-e-3")

            elif config.provider == ImageProvider.STABILITY_AI:
                if not config.api_key:
                    raise ValueError("Stability AI requires API key")
                client = StabilityAIClient(config.api_key, config.model or "stable-diffusion-xl-1024-v1-0")

            elif config.provider == ImageProvider.MIDJOURNEY:
                if not config.api_key:
                    raise ValueError("Midjourney requires API key")
                client = MidjourneyClient(config.api_key, config.proxy_url or "https://api.midjourney.com/v1")

            else:
                raise ValueError(f"Unsupported image provider: {config.provider}")

            self.clients[config.provider] = client
            self.configs[config.provider] = config

            print(f"✅ Registered {config.provider.value} image generation service")

        except Exception as e:
            print(f"❌ Failed to register {config.provider.value}: {e}")

    async def generate_images(self, request: ImageGenerationRequest,
                            provider: Optional[ImageProvider] = None,
                            criteria: Optional[GenerationCriteria] = None,
                            fallback: bool = True) -> ImageGenerationResponse:
        """生成图像"""
        if provider:
            # 使用指定提供商
            if provider in self.clients:
                try:
                    async with self.clients[provider]:
                        response = await self.clients[provider].generate_images(request)
                        await self._update_stats(provider, response)
                        return response
                except Exception as e:
                    if not fallback:
                        raise
                    print(f"❌ {provider.value} failed, trying fallback: {e}")

        # 自动选择提供商或使用回退策略
        providers = await self._rank_providers(request, criteria)

        for provider in providers:
            if provider not in self.clients:
                continue

            try:
                async with self.clients[provider]:
                    response = await self.clients[provider].generate_images(request)
                    await self._update_stats(provider, response)
                    print(f"✅ Images generated using {provider.value}")
                    return response

            except Exception as e:
                print(f"❌ {provider.value} generation failed: {e}")
                continue

        return ImageGenerationResponse(
            images=[],
            error_message="All image generation providers failed"
        )

    async def batch_generate(self, requests: List[ImageGenerationRequest],
                           provider: Optional[ImageProvider] = None) -> List[ImageGenerationResponse]:
        """批量生成图像"""
        if provider and provider in self.clients:
            # 使用指定提供商
            async with self.clients[provider]:
                return await self.clients[provider].batch_generate(requests)

        # 并发处理多个请求
        tasks = []
        for request in requests:
            task = self.generate_images(request)
            tasks.append(task)

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def get_available_capabilities(self) -> Dict[ImageProvider, ModelCapabilities]:
        """获取所有提供商的能力"""
        capabilities = {}

        for provider, client in self.clients.items():
            try:
                async with client:
                    capabilities[provider] = await client.get_capabilities()
            except Exception as e:
                print(f"Failed to get capabilities from {provider.value}: {e}")

        return capabilities

    async def estimate_costs(self, request: ImageGenerationRequest) -> Dict[ImageProvider, float]:
        """估算各提供商的生成成本"""
        costs = {}

        for provider, client in self.clients.items():
            try:
                cost = client.estimate_cost(request)
                costs[provider] = cost
            except Exception as e:
                print(f"Failed to estimate cost for {provider.value}: {e}")
                costs[provider] = 0.0

        return costs

    async def compare_providers(self, request: ImageGenerationRequest) -> Dict[ImageProvider, Dict[str, Any]]:
        """比较不同提供商的性能和成本"""
        comparison = {}

        for provider, client in self.clients.items():
            try:
                config = self.configs[provider]
                cost = client.estimate_cost(request)
                capabilities = await client.get_capabilities()
                model_info = await client.get_model_info()

                comparison[provider] = {
                    "estimated_cost": cost,
                    "max_images_per_request": capabilities.max_images_per_request,
                    "supports_style": request.style in capabilities.supported_styles,
                    "supports_negative_prompt": capabilities.supports_negative_prompt,
                    "supports_seed": capabilities.supports_seed,
                    "estimated_time_seconds": self._estimate_generation_time(provider),
                    "provider_priority": config.priority,
                    "model_info": model_info,
                    "recent_success_rate": self._get_success_rate(provider)
                }

            except Exception as e:
                comparison[provider] = {
                    "error": str(e),
                    "available": False
                }

        return comparison

    async def select_best_provider(self, request: ImageGenerationRequest,
                                 criteria: Optional[GenerationCriteria] = None) -> Optional[ImageProvider]:
        """选择最佳提供商"""
        providers = await self._rank_providers(request, criteria)
        return providers[0] if providers else None

    async def health_check(self) -> Dict[ImageProvider, bool]:
        """健康检查"""
        health_status = {}

        for provider, client in self.clients.items():
            try:
                async with client:
                    health_status[provider] = await client.health_check()
            except Exception:
                health_status[provider] = False

        return health_status

    async def _rank_providers(self, request: ImageGenerationRequest,
                            criteria: Optional[GenerationCriteria] = None) -> List[ImageProvider]:
        """根据请求和标准对提供商进行排序"""
        available_providers = []

        for provider, config in self.configs.items():
            if provider not in self.clients or not config.enabled:
                continue

            # 计算提供商分数
            score = await self._calculate_provider_score(provider, request, criteria)
            available_providers.append((provider, score))

        # 按分数排序
        available_providers.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in available_providers]

    async def _calculate_provider_score(self, provider: ImageProvider,
                                      request: ImageGenerationRequest,
                                      criteria: Optional[GenerationCriteria] = None) -> float:
        """计算提供商分数"""
        config = self.configs[provider]
        client = self.clients[provider]
        score = 0.0

        try:
            # 基础优先级分数
            score += (5 - config.priority) * 0.2

            # 成本分数
            cost = client.estimate_cost(request)
            if criteria and criteria.max_cost_per_image:
                cost_per_image = cost / request.num_images
                if cost_per_image <= criteria.max_cost_per_image:
                    score += 0.3
                else:
                    score -= 0.2
            else:
                # 成本越低分数越高
                if cost <= 0.05:
                    score += 0.3
                elif cost <= 0.10:
                    score += 0.2
                else:
                    score += 0.1

            # 能力匹配分数
            capabilities = await client.get_capabilities()

            # 风格支持
            if request.style in capabilities.supported_styles:
                score += 0.2

            # 负面提示词支持
            if request.negative_prompt and capabilities.supports_negative_prompt:
                score += 0.1

            # 种子支持
            if request.seed and capabilities.supports_seed:
                score += 0.1

            # 批量支持
            if request.num_images > 1 and request.num_images <= capabilities.max_images_per_request:
                score += 0.1

            # 历史成功率
            success_rate = self._get_success_rate(provider)
            score += success_rate * 0.2

            # 特定提供商优势
            if provider == ImageProvider.DALLE:
                if request.style in [ImageStyle.REALISTIC, ImageStyle.PHOTOGRAPHIC]:
                    score += 0.1
            elif provider == ImageProvider.STABILITY_AI:
                if request.style in [ImageStyle.ARTISTIC, ImageStyle.DIGITAL_ART]:
                    score += 0.1
            elif provider == ImageProvider.MIDJOURNEY:
                if request.style in [ImageStyle.ARTISTIC, ImageStyle.FANTASY, ImageStyle.ANIME]:
                    score += 0.1

        except Exception as e:
            print(f"Error calculating score for {provider.value}: {e}")
            score = 0.0

        return score

    def _estimate_generation_time(self, provider: ImageProvider) -> int:
        """估算生成时间（秒）"""
        time_estimates = {
            ImageProvider.DALLE: 30,
            ImageProvider.STABILITY_AI: 60,
            ImageProvider.MIDJOURNEY: 120
        }
        return time_estimates.get(provider, 60)

    def _get_success_rate(self, provider: ImageProvider) -> float:
        """获取提供商成功率"""
        stats = self.generation_stats[provider]
        if stats["total_requests"] == 0:
            return 1.0  # 没有历史数据时假设100%成功率

        return stats["successful_requests"] / stats["total_requests"]

    async def _update_stats(self, provider: ImageProvider, response: ImageGenerationResponse):
        """更新统计信息"""
        stats = self.generation_stats[provider]
        stats["total_requests"] += 1

        if response.images:
            stats["successful_requests"] += 1
            stats["total_cost"] += response.total_cost
            stats["total_time_ms"] += response.processing_time_ms

            # 计算平均质量分数（基于图像大小和处理时间）
            if response.images:
                avg_size = sum(img.width * img.height for img in response.images) / len(response.images)
                quality_score = min(1.0, avg_size / (1024 * 1024))  # 基于分辨率的质量分数
                stats["average_quality_score"] = (
                    (stats["average_quality_score"] * (stats["successful_requests"] - 1) + quality_score) /
                    stats["successful_requests"]
                )

    async def save_images(self, response: ImageGenerationResponse,
                        save_directory: Optional[str] = None,
                        filename_prefix: str = "generated") -> List[str]:
        """保存生成的图像"""
        if not response.images:
            return []

        save_dir = save_directory or self.temp_dir
        os.makedirs(save_dir, exist_ok=True)

        saved_paths = []

        for i, image in enumerate(response.images):
            if image.image_data:
                # 从数据保存
                filename = f"{filename_prefix}_{i+1:03d}.{image.format}"
                filepath = os.path.join(save_dir, filename)

                with open(filepath, "wb") as f:
                    f.write(image.image_data)

                saved_paths.append(filepath)

            elif image.image_url:
                # 从URL下载
                try:
                    # 选择任一客户端下载
                    client = next(iter(self.clients.values()))
                    image_data = await client.download_image(image.image_url)

                    filename = f"{filename_prefix}_{i+1:03d}.{image.format}"
                    filepath = os.path.join(save_dir, filename)

                    with open(filepath, "wb") as f:
                        f.write(image_data)

                    saved_paths.append(filepath)

                except Exception as e:
                    print(f"Failed to download image {i}: {e}")

        return saved_paths

    async def get_generation_history(self, provider: Optional[ImageProvider] = None,
                                   limit: int = 10) -> List[Dict[str, Any]]:
        """获取生成历史（如果提供商支持）"""
        history = []

        providers_to_check = [provider] if provider else list(self.clients.keys())

        for p in providers_to_check:
            if p not in self.clients:
                continue

            try:
                client = self.clients[p]
                if hasattr(client, 'get_job_history'):
                    async with client:
                        provider_history = await client.get_job_history(limit)
                        for job in provider_history:
                            job["provider"] = p.value
                            history.append(job)
            except Exception as e:
                print(f"Failed to get history from {p.value}: {e}")

        return sorted(history, key=lambda x: x.get("created_at", ""), reverse=True)[:limit]

    async def get_service_statistics(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        stats = {
            "total_providers": len(self.clients),
            "active_providers": len([c for c in self.configs.values() if c.enabled]),
            "provider_health": await self.health_check(),
            "generation_stats": self.generation_stats
        }

        # 计算总体统计
        total_requests = sum(p["total_requests"] for p in self.generation_stats.values())
        total_successful = sum(p["successful_requests"] for p in self.generation_stats.values())
        total_cost = sum(p["total_cost"] for p in self.generation_stats.values())

        stats["overall"] = {
            "total_requests": total_requests,
            "success_rate": total_successful / total_requests if total_requests > 0 else 0,
            "total_cost": total_cost,
            "average_cost_per_request": total_cost / total_successful if total_successful > 0 else 0
        }

        return stats

    async def cleanup(self):
        """清理临时文件和关闭连接"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"Cleanup failed: {e}")

        for client in self.clients.values():
            if hasattr(client, 'session') and client.session:
                await client.session.close()


# 预设配置
def create_default_image_manager(
    dalle_api_key: Optional[str] = None,
    dalle_model: str = "dall-e-3",
    stability_api_key: Optional[str] = None,
    stability_model: str = "stable-diffusion-xl-1024-v1-0",
    midjourney_api_key: Optional[str] = None,
    midjourney_proxy_url: Optional[str] = None
) -> ImageGenerationManager:
    """创建默认图像生成管理器"""

    manager = ImageGenerationManager()

    # 注册DALL-E
    if dalle_api_key:
        dalle_config = ImageProviderConfig(
            provider=ImageProvider.DALLE,
            api_key=dalle_api_key,
            model=dalle_model,
            enabled=True,
            priority=1,  # 高优先级
            cost_weight=0.8,
            quality_weight=1.0,
            speed_weight=0.9
        )
        asyncio.create_task(manager.register_provider(dalle_config))

    # 注册Stability AI
    if stability_api_key:
        stability_config = ImageProviderConfig(
            provider=ImageProvider.STABILITY_AI,
            api_key=stability_api_key,
            model=stability_model,
            enabled=True,
            priority=2,
            cost_weight=1.0,
            quality_weight=0.9,
            speed_weight=0.8
        )
        asyncio.create_task(manager.register_provider(stability_config))

    # 注册Midjourney
    if midjourney_api_key:
        midjourney_config = ImageProviderConfig(
            provider=ImageProvider.MIDJOURNEY,
            api_key=midjourney_api_key,
            proxy_url=midjourney_proxy_url,
            enabled=True,
            priority=3,
            cost_weight=0.6,
            quality_weight=1.0,
            speed_weight=0.5
        )
        asyncio.create_task(manager.register_provider(midjourney_config))

    return manager