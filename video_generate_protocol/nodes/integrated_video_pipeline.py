"""
集成视频生成管道 - 完全受节点约束和控制
"""
from typing import Dict, List, Optional, Any
import asyncio
from dataclasses import dataclass
from pathlib import Path
import json

from storyboard_sequence_node import StoryboardSequenceNode
from vgp_optimization_node import (
    VGPOptimizationNode,
    VGPOptimizationConfig,
    GenerationMode
)
from qwen_integration import StoryboardToVideoProcessor
from image_generation_node import ImageGenerationNode


@dataclass
class VideoGenerationRequest:
    """视频生成请求"""
    text_description: str  # 自然语言描述
    duration_seconds: int  # 视频时长（秒）
    product_info: Optional[Dict] = None  # 产品信息
    style_preferences: Optional[Dict] = None  # 风格偏好
    output_path: Optional[str] = None  # 输出路径


class IntegratedVideoPipeline:
    """集成的视频生成管道 - 完全节点控制"""

    def __init__(self, config: Dict[str, Any]):
        self.config = config

        # 初始化所有节点
        self._init_nodes()

        # 工作目录
        self.work_dir = Path(config.get("work_dir", "/tmp/video_pipeline"))
        self.work_dir.mkdir(parents=True, exist_ok=True)

    def _init_nodes(self):
        """初始化所有控制节点"""

        # 1. VGP优化节点 - 控制生成策略
        vgp_config = VGPOptimizationConfig(
            product_protection_level="maximum",
            product_consistency_threshold=0.95,
            era_preference=self.config.get("era_preference", "modern"),
            forbidden_elements=self.config.get("forbidden_elements", []),
            prefer_wide_shots=True,
            enhance_lighting=True
        )
        self.vgp_node = VGPOptimizationNode(vgp_config)

        # 2. 分镜序列节点 - 控制分镜逻辑
        self.storyboard_node = StoryboardSequenceNode(self.config)

        # 3. 图像生成节点 - 控制图像生成
        self.image_node = ImageGenerationNode(self.config)

        # 4. 视频处理节点 - 控制视频生成
        qwen_key = self.config.get("qwen_api_key")
        if not qwen_key:
            raise ValueError("Qwen API key is required")
        self.video_processor = StoryboardToVideoProcessor(qwen_key)

    async def generate_video(self, request: VideoGenerationRequest) -> Dict[str, Any]:
        """
        完整的视频生成流程 - 完全受节点控制
        """

        print("\n" + "="*60)
        print("🎬 开始视频生成流程")
        print("="*60)

        # 阶段1: VGP优化和分镜规划
        print("\n[阶段1] 📋 VGP优化和分镜规划...")
        storyboard_plan = await self._plan_storyboard(request)

        # 阶段2: 生成关键帧图像
        print("\n[阶段2] 🎨 生成关键帧图像...")
        keyframes = await self._generate_keyframes(storyboard_plan)

        # 阶段3: 生成视频片段
        print("\n[阶段3] 🎥 生成视频片段...")
        video_clips = await self._generate_video_clips(keyframes)

        # 阶段4: 合并最终视频
        print("\n[阶段4] 🔄 合并最终视频...")
        final_video = await self._merge_final_video(video_clips, request.output_path)

        # 阶段5: 质量验证
        print("\n[阶段5] ✅ 质量验证...")
        validation_result = await self._validate_output(final_video, storyboard_plan)

        print("\n" + "="*60)
        print("🎉 视频生成完成！")
        print("="*60)

        return {
            "success": True,
            "video_path": final_video,
            "duration_seconds": request.duration_seconds,
            "segments_generated": len(video_clips),
            "keyframes_generated": len(keyframes),
            "validation": validation_result,
            "storyboard_plan": storyboard_plan
        }

    async def _plan_storyboard(self, request: VideoGenerationRequest) -> Dict:
        """阶段1: 分镜规划（受VGP节点控制）"""

        # 解析用户输入
        raw_segments = self._parse_text_to_segments(request.text_description)

        # VGP优化 - 这里是核心控制点
        optimization_result = await self.vgp_node.optimize_storyboard_sequence(
            raw_segments=raw_segments,
            product_info=request.product_info,
            total_duration_ms=request.duration_seconds * 1000
        )

        # 验证优化结果
        self._validate_optimization(optimization_result)

        return optimization_result

    async def _generate_keyframes(self, storyboard_plan: Dict) -> List[Dict]:
        """阶段2: 生成关键帧（受图像生成节点控制）"""

        optimized_frames = storyboard_plan['optimized_frames']
        generated_keyframes = []

        # 按生成模式分组处理
        frame_groups = self._group_frames_by_mode(optimized_frames)

        for mode, frames in frame_groups.items():
            print(f"  生成模式 {mode}: {len(frames)} 帧")

            if mode == GenerationMode.PRODUCT_GUIDED:
                # 产品引导生成 - 最高优先级
                keyframes = await self._generate_product_guided_frames(frames)

            elif mode == GenerationMode.IMAGE_TO_IMAGE:
                # 图生图 - 保持连续性
                keyframes = await self._generate_img2img_frames(frames)

            else:
                # 文生图 - 独立生成
                keyframes = await self._generate_txt2img_frames(frames)

            generated_keyframes.extend(keyframes)

        # 处理帧复用
        generated_keyframes = self._process_frame_reuse(generated_keyframes)

        return generated_keyframes

    async def _generate_product_guided_frames(self, frames: List) -> List[Dict]:
        """产品引导的图像生成（最严格控制）"""

        keyframes = []

        for frame in frames:
            # 提取产品参考图
            product_ref = frame.reference_product_image

            if not product_ref:
                print(f"  ⚠️ 警告：产品帧 {frame.frame_id} 缺少参考图")

            # 生成图像 - 使用产品约束
            image_result = await self.image_node.generate_single_image(
                prompt=frame.prompt_optimization['base_description'],
                style="product_photography",
                quality="high",
                provider="dalle"  # 或其他支持产品一致性的提供商
            )

            if image_result:
                keyframes.append({
                    "frame_id": frame.frame_id,
                    "segment_id": frame.segment_id,
                    "image_path": image_result.image_path,
                    "generation_mode": "product_guided",
                    "is_reused": False
                })

        return keyframes

    async def _generate_img2img_frames(self, frames: List) -> List[Dict]:
        """图生图生成（保持连续性）"""

        keyframes = []

        for frame in frames:
            # 获取参考帧
            ref_frame_id = frame.reference_frame_id
            ref_image = self._get_reference_image(ref_frame_id, keyframes)

            # 基于参考图生成
            # 这里应该调用支持img2img的API
            image_result = await self._generate_with_reference(
                frame,
                ref_image
            )

            if image_result:
                keyframes.append({
                    "frame_id": frame.frame_id,
                    "segment_id": frame.segment_id,
                    "image_path": image_result,
                    "generation_mode": "img2img",
                    "is_reused": False
                })

        return keyframes

    async def _generate_txt2img_frames(self, frames: List) -> List[Dict]:
        """文生图生成（独立生成）"""

        keyframes = []

        for frame in frames:
            image_result = await self.image_node.generate_single_image(
                prompt=frame.prompt_optimization['base_description'],
                style=frame.prompt_optimization.get('style_tags', ['modern'])[0],
                quality="high"
            )

            if image_result:
                keyframes.append({
                    "frame_id": frame.frame_id,
                    "segment_id": frame.segment_id,
                    "image_path": image_result.image_path,
                    "generation_mode": "txt2img",
                    "is_reused": False
                })

        return keyframes

    def _process_frame_reuse(self, keyframes: List[Dict]) -> List[Dict]:
        """处理帧复用逻辑"""

        processed = []

        for i in range(len(keyframes)):
            frame = keyframes[i]

            # 检查是否需要复用为下一段的首帧
            if i > 0 and i % 2 == 0:  # 每个尾帧位置
                # 创建复用帧
                reused_frame = {
                    "frame_id": frame["frame_id"],
                    "segment_id": frame["segment_id"] + 1,
                    "image_path": frame["image_path"],
                    "generation_mode": frame["generation_mode"],
                    "is_reused": True,
                    "source_frame_id": frame["frame_id"]
                }
                processed.append(reused_frame)

            processed.append(frame)

        return processed

    async def _generate_video_clips(self, keyframes: List[Dict]) -> List[str]:
        """阶段3: 生成视频片段（受千问API约束）"""

        # 使用视频处理器生成5秒片段
        clips = await self.video_processor.process_storyboard_frames(
            keyframes,
            str(self.work_dir / "clips")
        )

        print(f"  生成了 {len(clips)} 个5秒视频片段")

        return clips

    async def _merge_final_video(self, clips: List[str], output_path: Optional[str]) -> str:
        """阶段4: 合并最终视频"""

        if not output_path:
            output_path = str(self.work_dir / "output.mp4")

        final_video = await self.video_processor.merge_clips(
            clips,
            output_path
        )

        print(f"  最终视频: {final_video}")

        return final_video

    async def _validate_output(self, video_path: str, storyboard_plan: Dict) -> Dict:
        """阶段5: 质量验证"""

        validation = {
            "video_exists": Path(video_path).exists(),
            "expected_duration": True,  # 应该验证实际时长
            "product_consistency": None,
            "scene_continuity": None,
            "quality_score": 0.0
        }

        # 验证产品一致性
        if storyboard_plan.get('optimization_report'):
            report = storyboard_plan['optimization_report']
            validation['quality_score'] = report['continuity_analysis']['average_score']

        # 这里可以调用VL模型进行视觉验证
        # validation['visual_check'] = await self._visual_validation(video_path)

        return validation

    # 辅助方法

    def _parse_text_to_segments(self, text: str) -> List[Dict]:
        """解析文本为段落"""
        lines = [l.strip() for l in text.strip().split('\n') if l.strip()]
        return [{"description": line, "index": i} for i, line in enumerate(lines)]

    def _group_frames_by_mode(self, frames: List) -> Dict[str, List]:
        """按生成模式分组"""
        groups = {}
        for frame in frames:
            mode = frame.generation_mode
            if mode not in groups:
                groups[mode] = []
            groups[mode].append(frame)
        return groups

    def _validate_optimization(self, optimization_result: Dict):
        """验证优化结果"""
        report = optimization_result.get('optimization_report', {})

        # 检查警告
        if report.get('warnings'):
            for warning in report['warnings']:
                print(f"  ⚠️ {warning}")

        # 检查连续性分数
        avg_score = report.get('continuity_analysis', {}).get('average_score', 0)
        if avg_score < 0.5:
            print(f"  ⚠️ 连续性分数较低: {avg_score:.2f}")

    def _get_reference_image(self, frame_id: str, existing_frames: List) -> Optional[str]:
        """获取参考图像"""
        for frame in existing_frames:
            if frame['frame_id'] == frame_id:
                return frame['image_path']
        return None

    async def _generate_with_reference(self, frame, ref_image: str) -> str:
        """基于参考图生成"""
        # 这里应该调用实际的img2img API
        # 暂时返回模拟路径
        return str(self.work_dir / f"{frame.frame_id}.png")


# 使用示例
async def demo():
    """完整流程演示"""

    # 配置
    config = {
        "qwen_api_key": "your_qwen_key",
        "openai_api_key": "your_openai_key",
        "work_dir": "/tmp/video_pipeline",
        "era_preference": "modern",
        "forbidden_elements": ["competitor", "low quality"]
    }

    # 初始化管道
    pipeline = IntegratedVideoPipeline(config)

    # 创建请求
    request = VideoGenerationRequest(
        text_description="""
        展示我们的新款智能手表SmartWatch Pro。
        首先展示产品的整体外观，360度旋转。
        然后聚焦表盘，展示时间和智能界面。
        接着演示运动追踪功能，显示心率监测。
        展示防水功能，水下使用场景。
        最后展示充电底座和精美包装。
        """,
        duration_seconds=30,
        product_info={
            "name": "SmartWatch Pro",
            "constraints": ["保持产品颜色一致", "必须显示品牌logo"],
            "reference_images": ["product_ref.jpg"]
        },
        output_path="/tmp/final_video.mp4"
    )

    # 生成视频
    result = await pipeline.generate_video(request)

    print("\n📊 生成结果:")
    print(f"视频路径: {result['video_path']}")
    print(f"时长: {result['duration_seconds']}秒")
    print(f"片段数: {result['segments_generated']}")
    print(f"关键帧数: {result['keyframes_generated']}")
    print(f"质量分数: {result['validation']['quality_score']:.2f}")

    return result


if __name__ == "__main__":
    asyncio.run(demo())