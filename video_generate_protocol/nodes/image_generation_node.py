"""
图像生成节点 - 集成多种AI图像生成服务的统一节点
"""
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

from image_generation import (
    ImageGenerationManager,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageStyle,
    ImageQuality,
    AspectRatio,
    ImageProvider,
    GenerationCriteria,
    create_default_image_manager
)


@dataclass
class ImageGenerationTask:
    """图像生成任务"""
    prompt: str
    negative_prompt: Optional[str] = None
    style: str = "realistic"
    quality: str = "standard"
    aspect_ratio: str = "square"
    width: Optional[int] = None
    height: Optional[int] = None
    num_images: int = 1
    seed: Optional[int] = None
    reference_image: Optional[str] = None  # 参考图像路径
    metadata: Dict[str, Any] = None


@dataclass
class ImageGenerationNodeRequest:
    """图像生成节点请求"""
    tasks: List[ImageGenerationTask]
    provider_preference: Optional[str] = None  # 首选提供商
    generation_config: Dict[str, Any] = None  # 生成配置
    output_config: Dict[str, Any] = None  # 输出配置
    batch_mode: bool = True  # 是否批量处理


@dataclass
class GeneratedImageResult:
    """生成的图像结果"""
    task_id: int
    image_path: str
    thumbnail_path: Optional[str] = None
    prompt: str = ""
    revised_prompt: str = ""
    width: int = 0
    height: int = 0
    format: str = "png"
    size_bytes: int = 0
    generation_time_ms: int = 0
    cost: float = 0.0
    provider: str = ""
    model: str = ""
    quality_score: float = 0.0
    metadata: Dict[str, Any] = None


@dataclass
class ImageGenerationNodeResponse:
    """图像生成节点响应"""
    generated_images: List[GeneratedImageResult]
    total_cost: float = 0.0
    total_time_ms: int = 0
    success_count: int = 0
    failure_count: int = 0
    provider_stats: Dict[str, Any] = None
    error_messages: List[str] = None


class ImageGenerationNode:
    """图像生成节点"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.image_manager: Optional[ImageGenerationManager] = None
        self.temp_dir = tempfile.mkdtemp(prefix="image_generation_node_")

        # 默认配置
        self.default_generation_config = {
            "enhance_prompts": True,
            "fallback_enabled": True,
            "max_retries": 2,
            "guidance_scale": 7.5,
            "num_inference_steps": 50
        }

        self.default_output_config = {
            "save_originals": True,
            "generate_thumbnails": True,
            "thumbnail_size": (256, 256),
            "output_format": "png",
            "include_metadata": True
        }

    async def initialize(self):
        """初始化图像生成管理器"""
        if self.image_manager is None:
            # 从配置或环境变量获取API密钥
            dalle_key = self.config.get("dalle_api_key") or os.getenv("OPENAI_API_KEY")
            dalle_model = self.config.get("dalle_model", "dall-e-3")

            stability_key = self.config.get("stability_api_key") or os.getenv("STABILITY_API_KEY")
            stability_model = self.config.get("stability_model", "stable-diffusion-xl-1024-v1-0")

            midjourney_key = self.config.get("midjourney_api_key") or os.getenv("MIDJOURNEY_API_KEY")
            midjourney_proxy = self.config.get("midjourney_proxy_url")

            self.image_manager = create_default_image_manager(
                dalle_api_key=dalle_key,
                dalle_model=dalle_model,
                stability_api_key=stability_key,
                stability_model=stability_model,
                midjourney_api_key=midjourney_key,
                midjourney_proxy_url=midjourney_proxy
            )

            # 等待服务注册完成
            await asyncio.sleep(0.1)

    async def process(self, request: ImageGenerationNodeRequest) -> ImageGenerationNodeResponse:
        """处理图像生成请求"""
        await self.initialize()

        # 合并配置
        generation_config = {**self.default_generation_config, **(request.generation_config or {})}
        output_config = {**self.default_output_config, **(request.output_config or {})}

        start_time = asyncio.get_event_loop().time()
        generated_images = []
        total_cost = 0.0
        success_count = 0
        failure_count = 0
        error_messages = []

        # 处理任务
        if request.batch_mode and len(request.tasks) > 1:
            # 批量处理
            results = await self._process_batch(
                request.tasks, request.provider_preference,
                generation_config, output_config
            )
        else:
            # 逐个处理
            results = []
            for i, task in enumerate(request.tasks):
                result = await self._process_single_task(
                    i, task, request.provider_preference,
                    generation_config, output_config
                )
                results.append(result)

        # 处理结果
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failure_count += 1
                error_messages.append(f"Task {i}: {str(result)}")
                continue

            if result.generated_images:
                generated_images.extend(result.generated_images)
                total_cost += result.total_cost
                success_count += result.success_count
                failure_count += result.failure_count
                if result.error_messages:
                    error_messages.extend(result.error_messages)

        total_time_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)

        # 获取提供商统计
        provider_stats = await self._get_provider_stats()

        return ImageGenerationNodeResponse(
            generated_images=generated_images,
            total_cost=total_cost,
            total_time_ms=total_time_ms,
            success_count=success_count,
            failure_count=failure_count,
            provider_stats=provider_stats,
            error_messages=error_messages if error_messages else None
        )

    async def _process_batch(self, tasks: List[ImageGenerationTask],
                           provider_preference: Optional[str],
                           generation_config: Dict[str, Any],
                           output_config: Dict[str, Any]) -> List:
        """批量处理任务"""
        # 将任务转换为生成请求
        requests = []
        for task in tasks:
            gen_request = await self._task_to_request(task, generation_config)
            requests.append(gen_request)

        # 确定提供商
        provider = None
        if provider_preference:
            try:
                provider = ImageProvider(provider_preference)
            except ValueError:
                print(f"Invalid provider preference: {provider_preference}")

        # 批量生成
        try:
            responses = await self.image_manager.batch_generate(requests, provider)

            # 处理响应
            results = []
            for i, (task, response) in enumerate(zip(tasks, responses)):
                if isinstance(response, Exception):
                    results.append(response)
                    continue

                result = await self._process_response(i, task, response, output_config)
                results.append(result)

            return results

        except Exception as e:
            print(f"Batch processing failed: {e}")
            # 回退到逐个处理
            results = []
            for i, task in enumerate(tasks):
                result = await self._process_single_task(
                    i, task, provider_preference, generation_config, output_config
                )
                results.append(result)
            return results

    async def _process_single_task(self, task_id: int, task: ImageGenerationTask,
                                 provider_preference: Optional[str],
                                 generation_config: Dict[str, Any],
                                 output_config: Dict[str, Any]) -> ImageGenerationNodeResponse:
        """处理单个任务"""
        try:
            # 转换为生成请求
            gen_request = await self._task_to_request(task, generation_config)

            # 确定提供商
            provider = None
            if provider_preference:
                try:
                    provider = ImageProvider(provider_preference)
                except ValueError:
                    pass

            # 生成图像
            response = await self.image_manager.generate_images(
                gen_request, provider,
                fallback=generation_config.get("fallback_enabled", True)
            )

            # 处理响应
            return await self._process_response(task_id, task, response, output_config)

        except Exception as e:
            print(f"Task {task_id} failed: {e}")
            return ImageGenerationNodeResponse(
                generated_images=[],
                total_cost=0.0,
                total_time_ms=0,
                success_count=0,
                failure_count=1,
                error_messages=[str(e)]
            )

    async def _task_to_request(self, task: ImageGenerationTask,
                             generation_config: Dict[str, Any]) -> ImageGenerationRequest:
        """将任务转换为生成请求"""
        # 映射枚举值
        style = ImageStyle(task.style) if task.style else ImageStyle.REALISTIC
        quality = ImageQuality(task.quality) if task.quality else ImageQuality.STANDARD
        aspect_ratio = AspectRatio(task.aspect_ratio) if task.aspect_ratio else AspectRatio.SQUARE

        # 构建请求
        request = ImageGenerationRequest(
            prompt=task.prompt,
            negative_prompt=task.negative_prompt,
            style=style,
            quality=quality,
            aspect_ratio=aspect_ratio,
            width=task.width,
            height=task.height,
            num_images=task.num_images,
            seed=task.seed,
            enhance_prompt=generation_config.get("enhance_prompts", True),
            guidance_scale=generation_config.get("guidance_scale", 7.5),
            num_inference_steps=generation_config.get("num_inference_steps", 50),
            metadata=task.metadata or {}
        )

        return request

    async def _process_response(self, task_id: int, task: ImageGenerationTask,
                              response: ImageGenerationResponse,
                              output_config: Dict[str, Any]) -> ImageGenerationNodeResponse:
        """处理生成响应"""
        generated_images = []

        if response.error_message:
            return ImageGenerationNodeResponse(
                generated_images=[],
                total_cost=response.total_cost,
                total_time_ms=response.processing_time_ms,
                success_count=0,
                failure_count=1,
                error_messages=[response.error_message]
            )

        # 保存图像
        if response.images:
            saved_paths = await self.image_manager.save_images(
                response,
                save_directory=self.temp_dir,
                filename_prefix=f"task_{task_id:03d}"
            )

            for i, (image, saved_path) in enumerate(zip(response.images, saved_paths)):
                # 生成缩略图
                thumbnail_path = None
                if output_config.get("generate_thumbnails", True):
                    thumbnail_path = await self._generate_thumbnail(
                        saved_path, output_config.get("thumbnail_size", (256, 256))
                    )

                # 计算质量分数
                quality_score = self._calculate_quality_score(image)

                result = GeneratedImageResult(
                    task_id=task_id,
                    image_path=saved_path,
                    thumbnail_path=thumbnail_path,
                    prompt=task.prompt,
                    revised_prompt=image.revised_prompt,
                    width=image.width,
                    height=image.height,
                    format=image.format,
                    size_bytes=image.size_bytes,
                    generation_time_ms=image.generation_time_ms,
                    cost=image.cost,
                    provider=response.provider,
                    model=response.model_name,
                    quality_score=quality_score,
                    metadata=image.metadata
                )
                generated_images.append(result)

        return ImageGenerationNodeResponse(
            generated_images=generated_images,
            total_cost=response.total_cost,
            total_time_ms=response.processing_time_ms,
            success_count=len(generated_images),
            failure_count=1 if not generated_images else 0
        )

    async def _generate_thumbnail(self, image_path: str, size: Tuple[int, int]) -> str:
        """生成缩略图"""
        try:
            from PIL import Image

            thumbnail_path = image_path.replace('.png', '_thumb.jpg').replace('.jpg', '_thumb.jpg')

            with Image.open(image_path) as img:
                img.thumbnail(size, Image.Resampling.LANCZOS)
                img.convert('RGB').save(thumbnail_path, 'JPEG', quality=85)

            return thumbnail_path

        except Exception as e:
            print(f"Failed to generate thumbnail: {e}")
            return None

    def _calculate_quality_score(self, image) -> float:
        """计算图像质量分数"""
        score = 0.5  # 基础分数

        # 分辨率分数
        pixel_count = image.width * image.height
        if pixel_count >= 1024 * 1024:
            score += 0.3
        elif pixel_count >= 512 * 512:
            score += 0.2
        else:
            score += 0.1

        # 文件大小分数（合理的文件大小表示质量）
        if image.size_bytes > 1024 * 1024:  # > 1MB
            score += 0.2
        elif image.size_bytes > 512 * 1024:  # > 512KB
            score += 0.1

        return min(score, 1.0)

    async def _get_provider_stats(self) -> Dict[str, Any]:
        """获取提供商统计信息"""
        try:
            return await self.image_manager.get_service_statistics()
        except Exception as e:
            print(f"Failed to get provider stats: {e}")
            return {}

    async def generate_single_image(self, prompt: str,
                                  style: str = "realistic",
                                  quality: str = "standard",
                                  provider: Optional[str] = None) -> Optional[GeneratedImageResult]:
        """生成单个图像（便捷方法）"""
        task = ImageGenerationTask(
            prompt=prompt,
            style=style,
            quality=quality,
            num_images=1
        )

        request = ImageGenerationNodeRequest(
            tasks=[task],
            provider_preference=provider,
            batch_mode=False
        )

        response = await self.process(request)
        return response.generated_images[0] if response.generated_images else None

    async def estimate_costs(self, tasks: List[ImageGenerationTask]) -> Dict[str, float]:
        """估算生成成本"""
        await self.initialize()

        total_costs = {}

        for task in tasks:
            gen_request = await self._task_to_request(task, self.default_generation_config)
            costs = await self.image_manager.estimate_costs(gen_request)

            for provider, cost in costs.items():
                provider_name = provider.value if hasattr(provider, 'value') else str(provider)
                if provider_name not in total_costs:
                    total_costs[provider_name] = 0.0
                total_costs[provider_name] += cost

        return total_costs

    async def get_available_providers(self) -> List[str]:
        """获取可用的提供商"""
        await self.initialize()
        health_status = await self.image_manager.health_check()
        return [provider.value for provider, healthy in health_status.items() if healthy]

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        await self.initialize()

        health_status = await self.image_manager.health_check()
        service_stats = await self.image_manager.get_service_statistics()

        return {
            "provider_health": {k.value: v for k, v in health_status.items()},
            "service_statistics": service_stats,
            "temp_directory": self.temp_dir,
            "temp_directory_exists": os.path.exists(self.temp_dir)
        }

    async def cleanup(self):
        """清理临时文件"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"Cleanup failed: {e}")

        if self.image_manager:
            await self.image_manager.cleanup()

    def __del__(self):
        """析构函数"""
        try:
            asyncio.create_task(self.cleanup())
        except:
            pass