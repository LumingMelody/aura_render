"""
图片到视频编排器 - 支持从单张图片生成视频
"""
from typing import Dict, List, Any, Optional
import asyncio
from dataclasses import dataclass
from pathlib import Path
import base64

from enhanced_video_orchestrator import EnhancedVideoOrchestrator
from qwen_integration import QwenVideoGenerator


@dataclass
class ImageToVideoRequest:
    """图片到视频请求"""

    # 必填字段
    image_path: str  # 输入图片路径
    duration_seconds: int  # 视频时长

    # 可选字段
    description: Optional[str] = None  # 视频描述（可选，用于引导生成）
    motion_intensity: str = "medium"  # 运动强度: low, medium, high
    style: str = "realistic"  # 视频风格

    # 生成参数
    generation_mode: str = "single_image_extend"  # 生成模式
    fps: int = 30  # 帧率
    resolution: str = "1920x1080"  # 分辨率

    # 输出配置
    output_path: Optional[str] = None
    save_intermediate: bool = False  # 是否保存中间帧


@dataclass
class ImageToVideoResponse:
    """图片到视频响应"""
    success: bool
    video_path: Optional[str] = None
    duration_seconds: int = 0
    segments_count: int = 0
    generation_mode: str = ""
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


class ImageToVideoOrchestrator:
    """图片到视频编排器"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 千问视频生成器
        qwen_key = config.get("qwen_api_key")
        if not qwen_key:
            raise ValueError("qwen_api_key is required")
        self.qwen_generator = QwenVideoGenerator(qwen_key)

        # 增强视频编排器（用于复杂场景）
        self.enhanced_orchestrator = EnhancedVideoOrchestrator(config)

        self.work_dir = Path(config.get("work_dir", "/tmp/image_to_video"))
        self.work_dir.mkdir(parents=True, exist_ok=True)

    async def process_image_to_video(self, request: ImageToVideoRequest) -> ImageToVideoResponse:
        """
        处理图片到视频的请求
        """

        print(f"\n🖼️ 开始图片到视频生成")
        print(f"📁 输入图片: {request.image_path}")
        print(f"⏱️ 目标时长: {request.duration_seconds}秒")
        print("="*60)

        try:
            # 验证输入图片
            if not Path(request.image_path).exists():
                raise FileNotFoundError(f"输入图片不存在: {request.image_path}")

            # 根据时长选择生成策略
            if request.duration_seconds <= 5:
                # 5秒以内，直接使用千问单次生成
                result = await self._generate_short_video(request)
            else:
                # 超过5秒，需要分段生成
                result = await self._generate_long_video(request)

            return result

        except Exception as e:
            print(f"❌ 图片到视频生成失败: {e}")
            return ImageToVideoResponse(
                success=False,
                error_message=str(e)
            )

    async def _generate_short_video(self, request: ImageToVideoRequest) -> ImageToVideoResponse:
        """生成短视频（≤5秒）"""

        print("\n[模式] 🎬 短视频直接生成")

        if request.generation_mode == "single_image_extend":
            # 单图扩展模式：使用同一张图作为首尾帧
            video_result = await self.qwen_generator.generate_video_from_frames(
                start_image_path=request.image_path,
                end_image_path=request.image_path,  # 使用同一张图
                duration_seconds=request.duration_seconds
            )

        elif request.generation_mode == "image_to_sequence":
            # 图片序列模式：生成中间帧然后生成视频
            end_frame_path = await self._generate_end_frame(request)

            video_result = await self.qwen_generator.generate_video_from_frames(
                start_image_path=request.image_path,
                end_image_path=end_frame_path,
                duration_seconds=request.duration_seconds
            )
        else:
            raise ValueError(f"不支持的生成模式: {request.generation_mode}")

        if video_result["success"]:
            # 等待生成完成
            completion_result = await self.qwen_generator.wait_for_completion(
                video_result["task_id"]
            )

            if completion_result["success"]:
                # 下载视频
                output_path = request.output_path or str(self.work_dir / "output.mp4")
                await self._download_video(completion_result["video_url"], output_path)

                return ImageToVideoResponse(
                    success=True,
                    video_path=output_path,
                    duration_seconds=request.duration_seconds,
                    segments_count=1,
                    generation_mode=request.generation_mode,
                    metadata={
                        "input_image": request.image_path,
                        "motion_intensity": request.motion_intensity,
                        "task_id": video_result["task_id"]
                    }
                )
            else:
                raise Exception(f"视频生成失败: {completion_result['error']}")
        else:
            raise Exception(f"API调用失败: {video_result['error']}")

    async def _generate_long_video(self, request: ImageToVideoRequest) -> ImageToVideoResponse:
        """生成长视频（>5秒）"""

        print(f"\n[模式] 🎬 长视频分段生成 ({request.duration_seconds}秒)")

        # 计算需要的段数
        segments_count = (request.duration_seconds + 4) // 5
        print(f"  📊 将分成 {segments_count} 个5秒片段")

        # 生成关键帧序列
        keyframes = await self._generate_keyframes_from_image(request, segments_count)

        # 生成视频片段
        video_clips = []
        for i in range(segments_count):
            start_frame = keyframes[i]
            end_frame = keyframes[i + 1] if i + 1 < len(keyframes) else keyframes[i]

            print(f"  🎥 生成片段 {i+1}/{segments_count}")

            clip_result = await self.qwen_generator.generate_video_from_frames(
                start_image_path=start_frame["path"],
                end_image_path=end_frame["path"],
                duration_seconds=5.0
            )

            if clip_result["success"]:
                completion = await self.qwen_generator.wait_for_completion(clip_result["task_id"])
                if completion["success"]:
                    clip_path = str(self.work_dir / f"clip_{i:03d}.mp4")
                    await self._download_video(completion["video_url"], clip_path)
                    video_clips.append(clip_path)
                    print(f"    ✅ 片段 {i+1} 生成完成")
                else:
                    print(f"    ❌ 片段 {i+1} 生成失败")
            else:
                print(f"    ❌ 片段 {i+1} API调用失败")

        if not video_clips:
            raise Exception("没有成功生成任何视频片段")

        # 合并视频片段
        output_path = request.output_path or str(self.work_dir / "final_video.mp4")
        final_video = await self._merge_video_clips(video_clips, output_path)

        return ImageToVideoResponse(
            success=True,
            video_path=final_video,
            duration_seconds=request.duration_seconds,
            segments_count=len(video_clips),
            generation_mode="multi_segment",
            metadata={
                "input_image": request.image_path,
                "keyframes_count": len(keyframes),
                "clips_generated": len(video_clips)
            }
        )

    async def _generate_keyframes_from_image(self, request: ImageToVideoRequest, segments_count: int) -> List[Dict]:
        """从输入图片生成关键帧序列"""

        print("  🖼️ 生成关键帧序列...")

        keyframes = []

        # 第一帧使用原始图片
        keyframes.append({
            "frame_id": "frame_000",
            "path": request.image_path,
            "is_original": True
        })

        # 根据描述生成中间帧和尾帧
        if request.description:
            # 有描述，使用AI生成演变帧
            for i in range(1, segments_count + 1):
                # 生成描述该时间点的提示词
                time_progress = i / segments_count
                frame_prompt = self._generate_frame_prompt(request, time_progress)

                # 使用图生图生成帧
                frame_path = await self._generate_frame_from_prompt(
                    reference_image=request.image_path,
                    prompt=frame_prompt,
                    frame_id=f"frame_{i:03d}"
                )

                keyframes.append({
                    "frame_id": f"frame_{i:03d}",
                    "path": frame_path,
                    "is_original": False,
                    "prompt": frame_prompt
                })
        else:
            # 无描述，生成轻微变化的帧
            for i in range(1, segments_count + 1):
                # 生成轻微运动的帧
                frame_path = await self._generate_motion_frame(
                    reference_image=request.image_path,
                    motion_intensity=request.motion_intensity,
                    frame_id=f"frame_{i:03d}"
                )

                keyframes.append({
                    "frame_id": f"frame_{i:03d}",
                    "path": frame_path,
                    "is_original": False
                })

        print(f"  ✅ 生成了 {len(keyframes)} 个关键帧")
        return keyframes

    def _generate_frame_prompt(self, request: ImageToVideoRequest, time_progress: float) -> str:
        """根据时间进度生成帧提示词"""

        base_description = request.description or "natural movement and progression"

        # 根据时间进度添加变化描述
        if time_progress < 0.3:
            stage = "beginning"
            description = f"early stage, {base_description}"
        elif time_progress < 0.7:
            stage = "middle"
            description = f"developing, {base_description}"
        else:
            stage = "end"
            description = f"final stage, {base_description}"

        # 添加运动强度
        motion_desc = {
            "low": "subtle movement",
            "medium": "moderate motion",
            "high": "dynamic action"
        }.get(request.motion_intensity, "moderate motion")

        return f"{description}, {motion_desc}, {request.style} style"

    async def _generate_frame_from_prompt(self, reference_image: str, prompt: str, frame_id: str) -> str:
        """使用提示词从参考图生成新帧"""

        # 这里应该调用实际的图生图API
        # 暂时返回原图路径（实际应该生成新的变化帧）

        print(f"    生成帧 {frame_id}: {prompt[:50]}...")

        # 模拟生成过程
        output_path = str(self.work_dir / f"{frame_id}.png")

        # 实际应该调用 img2img API
        # await self.image_generation_api.img2img(
        #     reference_image=reference_image,
        #     prompt=prompt,
        #     output_path=output_path
        # )

        # 暂时复制原图（演示用）
        import shutil
        shutil.copy2(reference_image, output_path)

        return output_path

    async def _generate_motion_frame(self, reference_image: str, motion_intensity: str, frame_id: str) -> str:
        """生成运动帧"""

        motion_prompts = {
            "low": "very subtle movement, minimal change",
            "medium": "gentle motion, natural progression",
            "high": "dynamic movement, noticeable change"
        }

        prompt = motion_prompts.get(motion_intensity, motion_prompts["medium"])

        return await self._generate_frame_from_prompt(reference_image, prompt, frame_id)

    async def _generate_end_frame(self, request: ImageToVideoRequest) -> str:
        """生成尾帧"""

        if request.description:
            end_prompt = f"final result of {request.description}, {request.style} style"
        else:
            end_prompt = f"natural progression, {request.motion_intensity} motion, {request.style} style"

        return await self._generate_frame_from_prompt(
            request.image_path,
            end_prompt,
            "end_frame"
        )

    async def _download_video(self, url: str, output_path: str):
        """下载视频文件"""

        import aiohttp

        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status == 200:
                    content = await response.read()
                    with open(output_path, "wb") as f:
                        f.write(content)
                else:
                    raise Exception(f"下载视频失败: {response.status}")

    async def _merge_video_clips(self, clip_paths: List[str], output_path: str) -> str:
        """合并视频片段"""

        print(f"  🔄 合并 {len(clip_paths)} 个视频片段...")

        # 创建ffmpeg输入列表
        list_file = self.work_dir / "clips.txt"
        with open(list_file, "w") as f:
            for clip_path in clip_paths:
                f.write(f"file '{clip_path}'\n")

        # 使用ffmpeg合并
        import subprocess

        cmd = [
            "ffmpeg",
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
            "-c", "copy",
            "-y",
            output_path
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        if result.returncode != 0:
            raise Exception(f"视频合并失败: {result.stderr}")

        # 清理临时文件
        list_file.unlink()

        print(f"  ✅ 视频合并完成: {output_path}")
        return output_path


# API接口函数
async def generate_video_from_image(
    image_path: str,
    duration_seconds: int,
    description: Optional[str] = None,
    motion_intensity: str = "medium",
    config: Optional[Dict] = None
) -> ImageToVideoResponse:
    """
    便捷API：从图片生成视频

    参数:
        image_path: 输入图片路径
        duration_seconds: 视频时长（秒）
        description: 视频描述（可选）
        motion_intensity: 运动强度 (low/medium/high)
        config: 配置信息

    返回:
        ImageToVideoResponse
    """

    if not config:
        config = {
            "qwen_api_key": "your_qwen_api_key",
            "work_dir": "/tmp/image_to_video"
        }

    orchestrator = ImageToVideoOrchestrator(config)

    request = ImageToVideoRequest(
        image_path=image_path,
        duration_seconds=duration_seconds,
        description=description,
        motion_intensity=motion_intensity
    )

    return await orchestrator.process_image_to_video(request)


# 使用示例
async def demo_image_to_video():
    """演示图片到视频功能"""

    # 配置
    config = {
        "qwen_api_key": "your_actual_qwen_key",
        "work_dir": "/tmp/image_to_video_demo"
    }

    # 示例1: 短视频（5秒以内）
    print("\n🎬 示例1: 短视频生成")
    short_result = await generate_video_from_image(
        image_path="/path/to/your/image.jpg",
        duration_seconds=3,
        description="产品缓慢旋转展示",
        motion_intensity="low",
        config=config
    )

    if short_result.success:
        print(f"✅ 短视频生成成功: {short_result.video_path}")
    else:
        print(f"❌ 短视频生成失败: {short_result.error_message}")

    # 示例2: 长视频（超过5秒）
    print("\n🎬 示例2: 长视频生成")
    long_result = await generate_video_from_image(
        image_path="/path/to/your/image.jpg",
        duration_seconds=15,
        description="从静态展示到动态使用场景的转变",
        motion_intensity="medium",
        config=config
    )

    if long_result.success:
        print(f"✅ 长视频生成成功: {long_result.video_path}")
        print(f"📊 片段数: {long_result.segments_count}")
    else:
        print(f"❌ 长视频生成失败: {long_result.error_message}")


if __name__ == "__main__":
    asyncio.run(demo_image_to_video())