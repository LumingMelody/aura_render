# ai_content_pipeline/strategies/storyboard_to_video_strategy.py
from typing import List, Dict, Any, Optional
from ai_content_pipeline.core.types import WorkflowConfig, TaskConfig, GenerationContext
from ai_content_pipeline.strategies.base_strategy import Strategy
from ai_content_pipeline.orchestrator.workflow import Workflow
from ai_content_pipeline.utils.helpers import parse_duration, is_consecutive

import asyncio


class StoryboardToVideoStrategy(Strategy):
    """
    将分镜脚本（blocks）编排为 AI 视频生成任务流
    支持连续性分析、时间切片、自动插入过渡帧
    """

    @property
    def name(self) -> str:
        return "storyboard_to_video"

    @property
    def description(self) -> str:
        return "将分镜脚本编排为 AI 视频，支持连续性分析与自动过渡帧插入"

    def build_workflow(self, context: GenerationContext) -> Workflow:
        blocks: List[Dict] = context["blocks"]
        durations: List[float] = context["total_duration"]  # 每个 block 的时长
        global_style: str = context.get("global_style", "")
        output_format: str = context.get("output_format", "mp4")

        workflow = Workflow(name="storyboard_video")

        # 1. 分析分镜结构
        segments = self._analyze_segments(blocks, durations)

        # 2. 生成所有图像（并行或串行）
        image_tasks = []
        for i, seg in enumerate(segments):
            task = TaskConfig(
                generator_key="stable_image",  # 可配置
                params={
                    "prompt": self._build_image_prompt(seg["block"], global_style),
                    "output_format": "png",
                    "size": "1024x576"
                },
                output_key=f"image_{i}"
            )
            image_tasks.append(task)

        # 添加所有生图任务（可并行）
        for task in image_tasks:
            workflow.add_task(Task(**task))

        # 3. 生成视频片段（按顺序，依赖图像）
        video_tasks = []
        for i in range(len(segments)):
            block = segments[i]["block"]
            duration = segments[i]["duration"]
            image_key = f"{{image_{i}}}"

            # 如果时长 > 5s，插入过渡帧
            sub_tasks = self._build_video_subtasks(
                block=block,
                duration=duration,
                image_start=image_key,
                global_style=global_style,
                output_format=output_format,
                task_index=i
            )

            for sub_task in sub_tasks:
                workflow.add_task(Task(**sub_task))

        return workflow

    def _analyze_segments(
        self,
        blocks: List[Dict],
        durations: List[float]
    ) -> List[Dict[str, Any]]:
        """
        分析分镜是否连续，合并连续分镜
        """
        if len(blocks) != len(durations):
            raise ValueError("blocks 和 durations 长度不匹配")

        segments = []
        i = 0
        while i < len(blocks):
            segment = {
                "block": blocks[i],
                "duration": durations[i],
                "is_continuous": False,
                "start_index": i,
                "end_index": i
            }

            # 检查是否与下一分镜连续
            if i < len(blocks) - 1 and is_consecutive(blocks[i], blocks[i + 1]):
                # 合并连续分镜（当前只合并两个，可扩展）
                combined_block = self._merge_blocks(blocks[i], blocks[i + 1])
                segment["block"] = combined_block
                segment["duration"] = durations[i] + durations[i + 1]
                segment["is_continuous"] = True
                segment["end_index"] = i + 1
                i += 2
            else:
                i += 1

            segments.append(segment)

        return segments

    def _merge_blocks(self, block1: Dict, block2: Dict) -> Dict:
        """
        合并两个连续分镜（例如：镜头推近 + 对话）
        """
        return {
            "type": "combined",
            "prompt": f"{block1.get('prompt', '')} → {block2.get('prompt', '')}",
            "scene": block1.get("scene"),
            "action": f"{block1.get('action', '')}; then {block2.get('action', '')}",
            "metadata": {
                "original_blocks": [block1, block2]
            }
        }

    def _build_image_prompt(self, block: Dict, global_style: str) -> str:
        """
        构建生图提示词
        """
        prompt = block.get("prompt") or block.get("description", "")
        style = block.get("style", global_style)
        return f"{prompt}, {style}, high quality, 4K, detailed"

    def _build_video_subtasks(
        self,
        block: Dict,
        duration: float,
        image_start: str,
        global_style: str,
        output_format: str,
        task_index: int
    ) -> List[TaskConfig]:
        """
        构建视频生成任务（支持 >5s 时插入过渡帧）
        """
        tasks = []
        remaining = duration
        segment_id = 0

        while remaining > 0:
            seg_duration = min(5.0, remaining)  # 最大 5s 一段
            is_last = seg_duration >= remaining

            # 构建视频提示词
            video_prompt = self._build_video_prompt(block, seg_duration, not is_last)

            # 视频生成任务
            video_task: TaskConfig = {
                "generator_key": "pika_video",
                "params": {
                    "prompt": video_prompt,
                    "image_start": image_start,
                    "image_end": "{{next_image}}" if not is_last else None,
                    "duration": seg_duration,
                    "output_format": output_format,
                    "motion_intensity": block.get("motion", "medium"),
                    "global_style": global_style
                },
                "output_key": f"video_segment_{task_index}_{segment_id}"
            }

            tasks.append(video_task)

            # 如果不是最后一段，插入“补充描述分镜”作为下一段的起始图
            if not is_last:
                # 生成过渡帧图像
                transition_task: TaskConfig = {
                    "generator_key": "stable_image",
                    "params": {
                        "prompt": self._build_transition_prompt(block),
                        "size": "1024x576"
                    },
                    "output_key": f"image_transition_{task_index}_{segment_id}"
                }
                tasks.append(transition_task)
                # 下一段的 image_start 就是这个 transition image
                image_start = f"{{image_transition_{task_index}_{segment_id}}}"

            remaining -= seg_duration
            segment_id += 1

        return tasks

    def _build_video_prompt(self, block: Dict, duration: float, has_transition: bool) -> str:
        """
        构建视频生成提示词
        """
        base_prompt = block.get("prompt") or block.get("description", "")
        if has_transition:
            return f"{base_prompt}, 持续 {duration:.1f}s，逐渐过渡到下一个场景"
        else:
            return f"{base_prompt}, 持续 {duration:.1f}s，自然结束"

    def _build_transition_prompt(self, block: Dict) -> str:
        """
        构建过渡帧提示词（补充描述分镜）
        """
        action = block.get("action", "")
        scene = block.get("scene", "")
        return f"过渡帧：{scene} 中 {action} 的中间状态，模糊过渡效果，soft blur, motion trail, 4K"