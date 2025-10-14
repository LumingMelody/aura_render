# ai_content_pipeline/strategies/talking_avatar_strategy.py
from typing import List, Dict, Any, Optional
from ai_content_pipeline.core.types import TaskConfig, GenerationContext
from ai_content_pipeline.strategies.base_strategy import Strategy
from ai_content_pipeline.orchestrator.workflow import Workflow
from ai_content_pipeline.orchestrator.task import Task
from ai_content_pipeline.utils.llm_client import LLMClient  # 假设有统一 LLM 接口

import asyncio
import aiohttp


class TalkingAvatarStoryStrategy(Strategy):
    """
    使用数字人讲解分镜内容的策略
    步骤：
    1. 匹配最合适的数字人形象（基于风格 + 内容）
    2. 为每个 block 生成讲解文案
    3. 调用数字人生成视频
    """

    AVATAR_API_URL = "https://api.avatar-service.com/v1/avatars"  # 示例 API

    @property
    def name(self) -> str:
        return "talking_avatar"

    @property
    def description(self) -> str:
        return "使用数字人讲解分镜内容，支持自动选角、文案生成"

    async def build_workflow(self, context: GenerationContext) -> Workflow:
        blocks: List[Dict] = context["blocks"]
        durations: List[float] = context["total_duration"]
        global_style: str = context.get("global_style", "professional")
        output_format: str = context.get("output_format", "mp4")

        workflow = Workflow(name="talking_avatar_video")

        # Step 1: 获取最匹配的数字人
        avatar_info = await self._select_avatar(blocks, global_style)
        if not avatar_info:
            raise RuntimeError("No suitable avatar found")

        # Step 2: 为每个 block 生成讲解文本
        narrations = await self._generate_narrations(blocks, global_style)

        # Step 3: 为每个 block 创建数字人讲解任务
        for i, (block, duration, narration) in enumerate(zip(blocks, durations, narrations)):
            task_config: TaskConfig = {
                "generator_key": "talking_avatar",  # 假设已注册该生成器
                "params": {
                    "text": narration,
                    "avatar_id": avatar_info["id"],
                    "avatar_url": avatar_info["video_url"],
                    "voice_style": avatar_info.get("voice_style", "neutral"),
                    "background": block.get("background", "studio"),
                    "gesture_style": block.get("emotion", "normal"),
                    "duration": duration,  # 用于控制语速
                    "output_format": output_format,
                    "style_prompt": global_style
                },
                "output_key": f"avatar_clip_{i}",
                "retry_policy": {
                    "max_retries": 3,
                    "base_delay": 2.0,
                    "jitter": True
                }
            }
            workflow.add_task(Task(**task_config))

        return workflow

    async def _select_avatar(self, blocks: List[Dict], style: str) -> Optional[Dict[str, Any]]:
        """
        调用外部 API 查询最匹配的数字人
        """
        # 提取内容关键词
        keywords = self._extract_keywords(blocks)

        async with aiohttp.ClientSession() as session:
            params = {
                "style": style,
                "keywords": ",".join(keywords),
                "limit": 50
            }
            try:
                async with session.get(self.AVATAR_API_URL, params=params) as resp:
                    if resp.status != 200:
                        return None
                    avatars = await resp.json()
            except Exception as e:
                print(f"Avatar API error: {e}")
                return None

        # 匹配逻辑：优先 style 完全匹配，其次 tag 匹配数最多
        best_match = None
        max_score = -1

        for avatar in avatars:
            score = 0
            tags = [t.lower() for t in avatar.get("tags", [])]

            if style.lower() in [t.lower() for t in avatar.get("styles", [])]:
                score += 10
            score += sum(1 for kw in keywords if kw.lower() in tags)

            if score > max_score:
                max_score = score
                best_match = avatar

        return best_match

    def _extract_keywords(self, blocks: List[Dict]) -> List[str]:
        """
        从 blocks 中提取关键词（可用于匹配数字人）
        """
        keywords = []
        for block in blocks:
            prompt = block.get("prompt", "")
            description = block.get("description", "")
            text = (prompt + " " + description).strip()

            # 这里可以调用 LLM 提取关键词，或简单关键词提取
            # 示例：简单关键词（实际可用 NLP 增强）
            words = text.split()
            # 提取名词性词汇（简化版）
            candidates = [w for w in words if len(w) > 2 and w.isalpha()]
            keywords.extend(candidates[:3])  # 每个 block 取前 3 个

        return list(set(keywords))  # 去重

    async def _generate_narrations(self, blocks: List[Dict], style: str) -> List[str]:
        """
        使用 LLM 为每个 block 生成讲解文案
        """
        llm = LLMClient()
        narrations = []

        system_prompt = f"""
        你是一位专业的视频解说员，风格为：{style}。
        请根据以下分镜描述，生成一段自然、口语化、时长约 3-5 秒的讲解词。
        要求：
        - 语言流畅，适合配音
        - 不要使用复杂术语
        - 控制在 20-40 字以内
        """

        for block in blocks:
            user_prompt = f"""
            分镜描述：{block.get('prompt', '')}
            补充信息：{block.get('description', '')}
            情绪：{block.get('emotion', 'neutral')}
            """

            try:
                narration = await llm.generate(
                    system=system_prompt,
                    prompt=user_prompt,
                    max_tokens=64,
                    temperature=0.7
                )
                narration = narration.strip().strip('"').strip("'").strip()
            except Exception:
                narration = block.get("prompt", "这段内容正在讲解中")[:50]

            narrations.append(narration)

        return narrations