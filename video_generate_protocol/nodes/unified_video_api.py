"""
统一视频生成API - /generate 接口
支持多种输入方式：纯文本、图片、图片+描述
"""
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from pathlib import Path
import asyncio

from video_generate_protocol.nodes.enhanced_video_orchestrator import EnhancedVideoOrchestrator
from video_generate_protocol.nodes.image_to_video_orchestrator import ImageToVideoOrchestrator
from video_generate_protocol.nodes.image_description_to_video import ImageDescriptionToVideoOrchestrator


@dataclass
class UnifiedVideoGenerateRequest:
    """
    统一的视频生成请求体

    支持三种模式：
    1. 纯文本模式：只有 text_description
    2. 纯图片模式：只有 image_path（或 images）
    3. 图片+描述模式：image_path + text_description
    """

    # 输入内容（至少需要一个）
    text_description: Optional[str] = None  # 文本描述
    image_path: Optional[str] = None  # 单张图片路径
    images: Optional[List[str]] = None  # 多张图片路径

    # 视频参数
    duration_seconds: int = 30  # 视频时长（秒）

    # VGP系统必需的ID参数
    theme_id: Optional[str] = None  # 主题ID
    keywords_id: Optional[str] = None  # 关键词ID
    target_duration_id: Optional[str] = None  # 目标时长ID
    user_description_id: Optional[str] = None  # 用户描述ID

    # 可选：分镜描述（配合images使用）
    storyboard_descriptions: Optional[List[str]] = None

    # 产品信息（可选）
    product_info: Optional[Dict[str, Any]] = None

    # 生成配置
    generation_config: Dict[str, Any] = field(default_factory=lambda: {
        "style": "realistic",
        "quality": "high",
        "motion_intensity": "medium",
        "transition_type": "smooth"
    })

    # 高级选项
    auto_detect_mode: bool = True  # 自动检测最佳生成模式
    use_vl_validation: bool = True  # 使用VL验证
    enable_frame_reuse: bool = True  # 启用帧复用优化

    # 输出配置
    output_path: Optional[str] = None
    save_intermediate: bool = False


@dataclass
class UnifiedVideoGenerateResponse:
    """统一的视频生成响应"""
    success: bool
    video_path: Optional[str] = None

    # 基本信息
    duration_seconds: int = 0
    generation_mode: str = ""  # 实际使用的生成模式

    # 生成统计
    segments_count: int = 0
    keyframes_count: int = 0

    # 耗时和成本
    total_time_ms: int = 0
    total_cost: float = 0.0

    # 错误信息
    error_code: Optional[str] = None
    error_message: Optional[str] = None

    # 详细信息
    metadata: Dict[str, Any] = field(default_factory=dict)


class UnifiedVideoGenerator:
    """
    统一视频生成器
    单一 /generate 接口，智能选择生成模式
    """

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 初始化各种生成器
        self.text_orchestrator = EnhancedVideoOrchestrator(config)
        self.image_orchestrator = ImageToVideoOrchestrator(config)
        self.image_desc_orchestrator = ImageDescriptionToVideoOrchestrator(config)

        self.work_dir = Path(config.get("work_dir", "/tmp/unified_video"))
        self.work_dir.mkdir(parents=True, exist_ok=True)

    async def generate(self, request: UnifiedVideoGenerateRequest) -> UnifiedVideoGenerateResponse:
        """
        统一的生成接口 - /generate
        根据输入自动选择最佳生成模式
        """

        print("\n" + "="*60)
        print("🎬 统一视频生成接口 /generate")
        print("="*60)

        try:
            # 检测生成模式
            generation_mode = self._detect_generation_mode(request)
            print(f"\n📋 检测到生成模式: {generation_mode}")

            # 根据模式调用对应的生成器
            if generation_mode == "text_only":
                return await self._generate_from_text(request)

            elif generation_mode == "image_only":
                return await self._generate_from_image(request)

            elif generation_mode == "image_with_description":
                return await self._generate_from_image_and_description(request)

            elif generation_mode == "multi_images":
                return await self._generate_from_storyboard(request)

            else:
                raise ValueError(f"不支持的生成模式: {generation_mode}")

        except Exception as e:
            print(f"❌ 生成失败: {e}")
            return UnifiedVideoGenerateResponse(
                success=False,
                error_code="GENERATION_FAILED",
                error_message=str(e)
            )

    def _detect_generation_mode(self, request: UnifiedVideoGenerateRequest) -> str:
        """
        自动检测最佳生成模式
        """

        has_text = bool(request.text_description)
        has_single_image = bool(request.image_path)
        has_multi_images = bool(request.images and len(request.images) > 1)

        if has_multi_images:
            return "multi_images"  # 多张图片（分镜）
        elif has_single_image and has_text:
            return "image_with_description"  # 图片+描述
        elif has_single_image:
            return "image_only"  # 仅图片
        elif has_text:
            return "text_only"  # 仅文本
        else:
            raise ValueError("请提供至少一种输入：text_description 或 image_path 或 images")

    async def _generate_from_text(self, request: UnifiedVideoGenerateRequest) -> UnifiedVideoGenerateResponse:
        """
        纯文本生成模式
        """

        print("\n[模式] 📝 纯文本生成")
        print(f"  描述: {request.text_description[:100]}...")
        print(f"  时长: {request.duration_seconds}秒")

        # 调用文本视频生成器
        from video_generate_protocol.nodes.video_storyboard_orchestrator import VideoStoryboardRequest

        storyboard_request = VideoStoryboardRequest(
            text_description=request.text_description,
            duration_seconds=request.duration_seconds,
            product_info=request.product_info,
            style_config=request.generation_config,
            output_path=request.output_path,
            theme_id=request.theme_id,
            keywords_id=request.keywords_id,
            target_duration_id=request.target_duration_id,
            user_description_id=request.user_description_id
        )

        result = await self.text_orchestrator.process_video_request(storyboard_request)

        if result["success"]:
            return UnifiedVideoGenerateResponse(
                success=True,
                video_path=result["video_path"],
                duration_seconds=request.duration_seconds,
                generation_mode="text_only",
                segments_count=result.get("segments_count", 0),
                keyframes_count=result.get("keyframes_count", 0),
                metadata={
                    "input_type": "text",
                    "storyboard_plan": result.get("storyboard_plan")
                }
            )
        else:
            return UnifiedVideoGenerateResponse(
                success=False,
                generation_mode="text_only",
                error_message=result.get("error", "文本生成失败")
            )

    async def _generate_from_image(self, request: UnifiedVideoGenerateRequest) -> UnifiedVideoGenerateResponse:
        """
        纯图片生成模式
        """

        print("\n[模式] 🖼️ 纯图片生成")
        print(f"  图片: {request.image_path}")
        print(f"  时长: {request.duration_seconds}秒")
        print(f"  运动强度: {request.generation_config.get('motion_intensity', 'medium')}")

        # 调用图片视频生成器
        from video_generate_protocol.nodes.image_to_video_orchestrator import ImageToVideoRequest

        image_request = ImageToVideoRequest(
            image_path=request.image_path,
            duration_seconds=request.duration_seconds,
            motion_intensity=request.generation_config.get("motion_intensity", "medium"),
            style=request.generation_config.get("style", "realistic"),
            output_path=request.output_path
        )

        result = await self.image_orchestrator.process_image_to_video(image_request)

        if result.success:
            return UnifiedVideoGenerateResponse(
                success=True,
                video_path=result.video_path,
                duration_seconds=result.duration_seconds,
                generation_mode="image_only",
                segments_count=result.segments_count,
                metadata={
                    "input_type": "image",
                    "motion_intensity": request.generation_config.get("motion_intensity")
                }
            )
        else:
            return UnifiedVideoGenerateResponse(
                success=False,
                generation_mode="image_only",
                error_message=result.error_message
            )

    async def _generate_from_image_and_description(self,
                                                  request: UnifiedVideoGenerateRequest) -> UnifiedVideoGenerateResponse:
        """
        图片+描述生成模式
        """

        print("\n[模式] 🖼️+📝 图片+描述生成")
        print(f"  图片: {request.image_path}")
        print(f"  描述: {request.text_description[:100]}...")
        print(f"  时长: {request.duration_seconds}秒")

        # 调用图片+描述生成器
        from video_generate_protocol.nodes.image_description_to_video import ImageDescriptionToVideoRequest

        img_desc_request = ImageDescriptionToVideoRequest(
            image_path=request.image_path,
            description=request.text_description,
            total_duration_seconds=request.duration_seconds,
            product_info=request.product_info,
            style_config=request.generation_config,
            use_vl_validation=request.use_vl_validation,
            output_path=request.output_path,
            theme_id=request.theme_id,
            keywords_id=request.keywords_id,
            target_duration_id=request.target_duration_id,
            user_description_id=request.user_description_id
        )

        result = await self.image_desc_orchestrator.process_request(img_desc_request)

        if result["success"]:
            return UnifiedVideoGenerateResponse(
                success=True,
                video_path=result["video_path"],
                duration_seconds=request.duration_seconds,
                generation_mode="image_with_description",
                segments_count=result.get("segments_count", 0),
                metadata={
                    "input_type": "image_with_description",
                    "segment_plans": result.get("segment_plans")
                }
            )
        else:
            return UnifiedVideoGenerateResponse(
                success=False,
                generation_mode="image_with_description",
                error_message=result.get("error", "图片+描述生成失败")
            )

    async def _generate_from_storyboard(self, request: UnifiedVideoGenerateRequest) -> UnifiedVideoGenerateResponse:
        """
        多图分镜生成模式
        """

        print("\n[模式] 🎬 多图分镜生成")
        print(f"  图片数: {len(request.images)}")
        print(f"  时长: {request.duration_seconds}秒")

        # 构建分镜项
        storyboard_items = []
        for i, image_path in enumerate(request.images):
            description = ""
            if request.storyboard_descriptions and i < len(request.storyboard_descriptions):
                description = request.storyboard_descriptions[i]
            else:
                description = f"场景 {i+1}"

            storyboard_items.append({
                "image": image_path,
                "description": description
            })

        # 调用分镜生成器
        from video_generate_protocol.nodes.image_description_to_video import ImageDescriptionToVideoRequest

        storyboard_request = ImageDescriptionToVideoRequest(
            storyboard_items=storyboard_items,
            total_duration_seconds=request.duration_seconds,
            product_info=request.product_info,
            style_config=request.generation_config,
            output_path=request.output_path,
            theme_id=request.theme_id,
            keywords_id=request.keywords_id,
            target_duration_id=request.target_duration_id,
            user_description_id=request.user_description_id
        )

        result = await self.image_desc_orchestrator.process_request(storyboard_request)

        if result["success"]:
            return UnifiedVideoGenerateResponse(
                success=True,
                video_path=result["video_path"],
                duration_seconds=request.duration_seconds,
                generation_mode="multi_images",
                segments_count=result.get("segments_count", 0),
                metadata={
                    "input_type": "storyboard",
                    "images_count": len(request.images)
                }
            )
        else:
            return UnifiedVideoGenerateResponse(
                success=False,
                generation_mode="multi_images",
                error_message=result.get("error", "分镜生成失败")
            )


# 统一API接口
async def generate_video(
    # 输入（至少需要一个）
    text_description: Optional[str] = None,
    image_path: Optional[str] = None,
    images: Optional[List[str]] = None,

    # 必填
    duration_seconds: int = 30,

    # VGP系统必需的ID参数
    theme_id: Optional[str] = None,
    keywords_id: Optional[str] = None,
    target_duration_id: Optional[str] = None,
    user_description_id: Optional[str] = None,

    # 可选
    product_info: Optional[Dict] = None,
    style: str = "realistic",
    motion_intensity: str = "medium",
    output_path: Optional[str] = None,

    # 配置
    config: Optional[Dict] = None
) -> UnifiedVideoGenerateResponse:
    """
    统一的视频生成API - /generate 接口

    支持多种输入方式：
    1. 纯文本：只传 text_description
    2. 纯图片：只传 image_path
    3. 图片+描述：传 image_path + text_description
    4. 多图分镜：传 images 列表

    返回:
        UnifiedVideoGenerateResponse
    """

    if not config:
        config = {
            "qwen_api_key": "your_qwen_api_key",
            "openai_api_key": "your_openai_api_key",
            "work_dir": "/tmp/unified_video"
        }

    generator = UnifiedVideoGenerator(config)

    request = UnifiedVideoGenerateRequest(
        text_description=text_description,
        image_path=image_path,
        images=images,
        duration_seconds=duration_seconds,
        theme_id=theme_id,
        keywords_id=keywords_id,
        target_duration_id=target_duration_id,
        user_description_id=user_description_id,
        product_info=product_info,
        generation_config={
            "style": style,
            "motion_intensity": motion_intensity
        },
        output_path=output_path
    )

    return await generator.generate(request)


# 使用示例
async def demo_unified_api():
    """演示统一API的各种用法"""

    config = {
        "qwen_api_key": "your_actual_key",
        "openai_api_key": "your_actual_key",
        "work_dir": "/tmp/demo"
    }

    # 示例1：只有文本
    print("\n📝 示例1: 纯文本生成")
    result1 = await generate_video(
        text_description="展示智能手表的完整功能，从外观到使用场景",
        duration_seconds=20,
        config=config
    )
    print(f"结果: {result1.generation_mode} - {'成功' if result1.success else '失败'}")

    # 示例2：只有图片
    print("\n🖼️ 示例2: 纯图片生成")
    result2 = await generate_video(
        image_path="/path/to/product.jpg",
        duration_seconds=10,
        motion_intensity="low",
        config=config
    )
    print(f"结果: {result2.generation_mode} - {'成功' if result2.success else '失败'}")

    # 示例3：图片+描述
    print("\n🖼️+📝 示例3: 图片+描述生成")
    result3 = await generate_video(
        image_path="/path/to/product.jpg",
        text_description="产品从静态展示到动态使用的转变",
        duration_seconds=15,
        config=config
    )
    print(f"结果: {result3.generation_mode} - {'成功' if result3.success else '失败'}")

    # 示例4：多图分镜
    print("\n🎬 示例4: 多图分镜生成")
    result4 = await generate_video(
        images=[
            "/path/to/scene1.jpg",
            "/path/to/scene2.jpg",
            "/path/to/scene3.jpg"
        ],
        duration_seconds=15,
        config=config
    )
    print(f"结果: {result4.generation_mode} - {'成功' if result4.success else '失败'}")


if __name__ == "__main__":
    asyncio.run(demo_unified_api())