from typing import Dict, List, Any, Optional, Tuple
import uuid
import requests
import asyncio
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

from llm.qwen import QwenLLM
from ..intelligent_material_supply_system import (
    IntelligentMaterialSupplySystem,
    MaterialSupplyRequest,
    CostBudget
)


# 假设的服务地址
AI_IMAGE_GENERATOR_URL = "https://api.ai-image-gen.com/v1/generate"
AI_TALKING_HEAD_URL = "https://api.talkinghead.ai/v1/generate"
ASSET_LIBRARY_URL = "https://api.assetlib.com/v1/search"  # 素材库搜索
VL_VALIDATION_URL = "https://api.qwen-vl.com/v1/validate"  # Qwen-VL 接口（模拟）

class Intelligent_video_matcher():
    required_inputs = [
        {
            "name": "scheduled_shots",
            "label": "已调度分镜列表",
            "type": list,
            "required": True,
            "desc": "包含已匹配素材或需生成标记的分镜",
            "field_type": "json"
        }
    ]

    system_parameters = {
        "ai_image_url": AI_IMAGE_GENERATOR_URL,
        "ai_talking_url": AI_TALKING_HEAD_URL,
        "asset_library_url": ASSET_LIBRARY_URL,
        "vl_validation_url": VL_VALIDATION_URL,
        "use_unified_style_reference": True,
        "use_unified_talking_source": True,
        "max_retries": 3,
        "timeout": 30
    }

    def __init__(self):
        # 全局一致性锚点
        self.style_reference_image: Optional[str] = None  # 统一风格图
        self.talking_source_video: Optional[str] = None   # 统一口播人物
        self.global_style_prompt: Optional[str] = None    # 全局风格描述（用于 AI 生成）

    async def match_intelligent_video(self, context: Dict[str, Any]) -> Dict[str, Any]:
        scheduled_shots = context.get("scheduled_shots", [])
        if not scheduled_shots:
            raise ValueError("缺少输入：scheduled_shots 不能为空")

        # 使用新的智能素材供给系统
        budget = CostBudget(
            total_budget=context.get("budget", 10.0),
            vl_call_limit=context.get("vl_limit", 50)
        )

        supply_system = IntelligentMaterialSupplySystem(budget)

        # 构建供给请求
        supply_request = MaterialSupplyRequest(
            shots=scheduled_shots,
            project_config=context.get("project_config", {}),
            user_preferences=context.get("user_preferences", {}),
            budget_constraints=context.get("budget_constraints", {})
        )

        # 执行智能素材供给
        supply_result = await supply_system.supply_materials(supply_request)

        # 转换为原有格式
        final_sequence = []
        current_time = 0.0

        for i, shot in enumerate(scheduled_shots):
            # 查找对应的素材
            matched_material = None
            for material in supply_result.final_materials:
                if self._material_matches_shot(material, shot, i):
                    matched_material = material
                    break

            if matched_material:
                # 使用找到的素材
                final_block = self._convert_material_to_block(matched_material, shot)
            else:
                # 使用占位符
                final_block = self._create_placeholder_asset(shot)

            final_sequence.append(final_block)
            current_time += final_block.get("duration", 3.0)

        # 返回结果
        return {
            "final_sequence": final_sequence,
            "total_duration": current_time,
            "style_reference_used": supply_result.style_anchor.style_type.value if supply_result.style_anchor else None,
            "talking_source_used": self.talking_source_video,
            "global_style_prompt": supply_result.style_anchor.style_type.value if supply_result.style_anchor else None,
            "generated_count": len([m for m in supply_result.final_materials if m.get("supply_method") == "ai_generation"]),
            "reused_asset_count": len([m for m in supply_result.final_materials if m.get("supply_method") != "ai_generation"]),
            "user_forced_asset_count": len([s for s in scheduled_shots if s.get("asset_status") == "matched" and s.get("scheduled_asset", {}).get("source") == "user_upload"]),
            "decision_strategy": supply_result.decision_strategy.value,
            "consistency_ratio": supply_result.quality_metrics.get("style_consistency_score", 0.0),
            "processing_time": supply_result.processing_time,
            "cost_report": supply_result.cost_report,
            "timestamp": datetime.now().isoformat(),
            "node_id": getattr(self, 'node_id', 'intelligent_video_matcher')
        }

    def _material_matches_shot(self, material: Dict, shot: Dict, shot_index: int) -> bool:
        """检查素材是否匹配分镜"""
        # 检查材料类型
        if material.get("type") == "generated_material":
            generation_info = material.get("generation_info", {})
            segments = generation_info.get("segments", [])

            # 检查是否有对应的片段
            for segment in segments:
                if shot_index in segment.get("original_indices", []):
                    return True

        # 检查现有素材的匹配
        material_description = material.get("verification_info", {}).get("source_description", "")
        shot_description = shot.get("description", "")

        # 简单的描述匹配
        if material_description and shot_description:
            common_words = set(material_description.lower().split()) & set(shot_description.lower().split())
            return len(common_words) >= 2

        return False

    def _convert_material_to_block(self, material: Dict, shot: Dict) -> Dict:
        """将素材转换为分镜块格式"""
        if material.get("type") == "generated_material":
            generation_info = material.get("generation_info", {})

            return {
                "type": generation_info.get("type", "ai_generated"),
                "source": "ai_generation",
                "url": generation_info.get("url", ""),
                "duration": material.get("duration", shot.get("duration", 3.0)),
                "description": shot.get("description", ""),
                "style": generation_info.get("style", {}),
                "generated_type": generation_info.get("type"),
                "generation_info": generation_info,
                "confidence": material.get("confidence", 0.95)
            }
        else:
            return {
                "type": "library_asset",
                "source": "library",
                "url": material.get("url", ""),
                "duration": material.get("duration", shot.get("duration", 3.0)),
                "description": shot.get("description", ""),
                "verification_info": material.get("verification_info", {}),
                "confidence": material.get("confidence", 0.8)
            }

    async def _analyze_user_assets_style(self, user_blocks: List[Dict]) -> Optional[str]:
        """
        分析用户上传素材的真实视觉风格，作为全局风格锚点
        """
        styles = []
        for block in user_blocks:
            asset = block["scheduled_asset"]
            thumbnail = asset.get("thumbnail")
            if not thumbnail:
                continue

            # 使用 Qwen-VL 分析真实风格
            visual_style = await self._analyze_visual_style_with_vl(thumbnail, block.get("description", ""))
            if visual_style:
                styles.append(visual_style)

        if styles:
            from collections import Counter
            return Counter(styles).most_common(1)[0][0]
        return None
    
    async def _analyze_visual_style_with_vl(self, image_url: str, description: str) -> Optional[str]:
        """
        使用 Qwen-VL 异步分析图像风格（非阻塞）
        """
        prompt = f"""
        请分析该图像的视觉风格和美学特征，结合描述判断其所属风格类别。

        【分镜描述】：{description}

        请从以下类别中选择最匹配的一项，并只回答该词：
        - cinematic（电影感，暗调，镜头光晕）
        - realistic（写实，自然光，真实场景）
        - anime（二次元，卡通，夸张表情）
        - documentary（纪实，手持感，低饱和）
        - advertisement（广告风，高光，产品聚焦）
        - cyberpunk（赛博朋克，霓虹灯，科技感）
        - watercolor（水彩，手绘，柔和边缘）

        请只回答一个词。
        """

        qwen = getattr(self, 'qwen', None)
        if not qwen:
            qwen = QwenLLM()

        try:
            # 使用线程池执行同步 generate 方法
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=4)
            response = await loop.run_in_executor(executor, self._call_qwen_generate, qwen, prompt, image_url)

            if response:
                answer = str(response).strip().lower()
                valid_styles = {
                    'cinematic', 'realistic', 'anime', 'documentary',
                    'advertisement', 'cyberpunk', 'watercolor'
                }
                for style in valid_styles:
                    if style in answer:
                        return style
        except Exception as e:
            print(f"[VL Style Analysis] 失败: {e}")

        return None

    # 提取为独立同步函数供 run_in_executor 调用
    def _call_qwen_generate(self, qwen: QwenLLM, prompt: str, image_url: str):
        return qwen.generate(
            prompt=prompt,
            images=[image_url],
            max_retries=3
        )


    async def _infer_style_from_description(self, description: str) -> Optional[str]:
        """
        使用 Qwen 文本大模型分析分镜描述，推断其视觉风格。
        输入：分镜文字描述
        输出：风格标签，如 'cinematic', 'anime' 等
        """
        prompt = f"""
        请根据以下视频分镜描述，分析其预期的视觉风格和美学倾向。

        【分镜描述】：
        {description}

        请从以下类别中选择最匹配的一项，并只回答该词：
        - cinematic（电影感，暗调，镜头光晕）
        - realistic（写实，自然光，真实场景）
        - anime（二次元，卡通，夸张表情）
        - documentary（纪实，手持感，低饱和）
        - advertisement（广告风，高光，产品聚焦）
        - cyberpunk（赛博朋克，霓虹灯，科技感）
        - watercolor（水彩，手绘，柔和边缘）

        要求：
        1. 只输出一个词，不要解释。
        2. 必须是上述类别之一。
        """

        qwen = getattr(self, 'qwen', None)
        if not qwen:
            qwen = QwenLLM()

        try:
            # 同步调用 generate（无 images → 自动使用文本模型）
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)
            response = await loop.run_in_executor(
                executor,
                lambda: qwen.generate(prompt=prompt, max_retries=3)
            )

            if response:
                answer = str(response).strip().lower()
                valid_styles = {
                    'cinematic', 'realistic', 'anime', 'documentary',
                    'advertisement', 'cyberpunk', 'watercolor'
                }
                # 清洗并匹配
                for style in valid_styles:
                    if style in answer:
                        return style
        except Exception as e:
            print(f"[Text Style Inference] 失败: {e}")
        
        return None

    def _group_consecutive_blocks(self, blocks: List[Tuple[int, Dict]]) -> List[List[Tuple[int, Dict]]]:
        """
        将连续的缺失块分组
        输入: [(3, b3), (5, b5), (6, b6), (7, b7), (10, b10), (11, b11)]
        输出: [[(3,b3)], [(5,b5),(6,b6),(7,b7)], [(10,b10),(11,b11)]]
        """
        if not blocks:
            return []

        sorted_blocks = sorted(blocks, key=lambda x: x[0])
        groups = []
        current_group = [sorted_blocks[0]]

        for i in range(1, len(sorted_blocks)):
            prev_idx = current_group[-1][0]
            curr_idx = sorted_blocks[i][0]
            if curr_idx == prev_idx + 1:  # 连续
                current_group.append(sorted_blocks[i])
            else:
                groups.append(current_group)
                current_group = [sorted_blocks[i]]
        groups.append(current_group)

        return groups
    async def _extract_global_style(self, shots: List[Dict]):
        """
        提取全局风格的正确逻辑：
        1. 优先：从已匹配的素材（matched）中分析真实风格
        2. 其次：使用分镜中定义的 style 字段（如 'cinematic', 'anime'）
        3. 可选：调用 Qwen-VL 对关键帧做视觉风格理解
        """
        # Step 1: 收集所有已匹配素材的风格
        matched_styles = []
        for block in shots:
            if block.get("asset_status") == "matched" and "scheduled_asset" in block:
                asset = block["scheduled_asset"]
                
                # 方法1：使用元数据中的 style（如果有）
                asset_style = asset.get("style")
                if asset_style:
                    matched_styles.append(asset_style.lower())
                    continue

                # 方法2（更新）：使用 Qwen 分析 description 推断风格
                description = block.get("description", "").strip()
                if description:
                    inferred_style = await self._infer_style_from_description(description)
                    if inferred_style:
                        matched_styles.append(inferred_style)
                        continue

                # 方法3（高级）：调用 Qwen-VL 分析关键帧视觉风格（推荐）
                thumbnail = asset.get("thumbnail")
                if thumbnail:
                    visual_style = await self._analyze_visual_style_with_vl(thumbnail, description)
                    if visual_style:
                        matched_styles.append(visual_style)
                        continue
        if matched_styles:
            # 取出现最多的风格（众数），增强鲁棒性
            from collections import Counter
            most_common_style = Counter(matched_styles).most_common(1)[0][0]
            self.global_style_prompt = most_common_style
            return

        # Step 2: fallback —— 使用所有分镜中定义的 style 字段
        defined_styles = [s.get("style") for s in shots if s.get("style")]
        if defined_styles:
            from collections import Counter
            most_common_style = Counter(defined_styles).most_common(1)[0][0]
            self.global_style_prompt = most_common_style.lower()
            return

        # Step 3: 完全无风格信息 → 使用 Qwen 大模型智能分析整体主题与风格
        topics = [s.get("description", "").strip() for s in shots if s.get("description")]
        if not topics:
            self.global_style_prompt = "realistic"
        else:
            # 汇总前几条关键描述（避免过长）
            MAX_DESC = 10  # 取最多10条分镜描述
            sample_descriptions = "\n".join(f"- {desc}" for desc in topics[:MAX_DESC])

            prompt = f"""
            你是一个视频美学分析专家。请根据以下分镜描述内容，判断整段视频最可能采用的**视觉风格**。

            【分镜描述示例】：
            {sample_descriptions}

            请从以下类别中选择一个最匹配的风格，并**只回答一个词**：
            - cinematic（电影感，暗调，镜头光晕）
            - realistic（写实，自然光，真实场景）
            - anime（二次元，卡通，夸张表情）
            - documentary（纪实，手持感，低饱和）
            - advertisement（广告风，高光，产品聚焦）
            - cyberpunk（赛博朋克，霓虹灯，科技感）
            - watercolor（水彩，手绘，柔和边缘）

            要求：
            1. 必须只输出上述词之一，不要解释。
            2. 综合整体氛围、用词、场景设定判断。
            3. 若偏科技/未来感 → cyberpunk
            偏自然/真实 → documentary 或 realistic
            偏幻想/卡通 → anime
            商业展示 → advertisement
            电影化叙事 → cinematic
            """

            try:
                qwen = getattr(self, 'qwen', None)
                if not qwen:
                    qwen = QwenLLM()

                # 同步调用 generate（纯文本）
                loop = asyncio.get_event_loop()
                executor = ThreadPoolExecutor(max_workers=1)
                response = await loop.run_in_executor(
                    executor,
                    lambda: qwen.generate(prompt=prompt, max_retries=3)
                )

                if response:
                    answer = str(response).strip().lower()
                    valid_styles = {
                        'cinematic', 'realistic', 'anime', 'documentary',
                        'advertisement', 'cyberpunk', 'watercolor'
                    }
                    for style in valid_styles:
                        if style in answer:
                            self.global_style_prompt = style
                            break
                    else:
                        # 模型输出无效时的兜底
                        self.global_style_prompt = "realistic"
                else:
                    self.global_style_prompt = "realistic"

            except Exception as e:
                print(f"[Global Style Inference] 失败: {e}")
                self.global_style_prompt = "realistic"

    
    
    def _detect_talking_head(self, block: Dict) -> bool:
        """
        判断是否为“数字人口播”类型
        """
        desc = block.get("description", "").lower()
        keywords = ["讲解", "介绍", "说明", "演讲", "主持人", "主播", "老师", "专家", "说", "谈到"]
        return any(kw in desc for kw in keywords) or block.get("content_type") == "talking_head"
    
    # async def _generate_ai_content(
    #     self,
    #     block: Dict,
    #     index: int,
    #     global_style: str,
    #     is_talking_head: bool
    # ) -> Dict:
    #     if is_talking_head:
    #         return await self._generate_talking_head_video(block, index, global_style)
    #     else:
    #         return await self._generate_pure_ai_video(block, index, global_style)
        

    async def _generate_ai_content_chunk(
        self,
        chunk_index: int,
        blocks: List[Dict],
        indices: List[int],
        total_duration: float,
        combined_description: str,
        global_style: str,
        is_talking_head: bool
    ) -> Dict:
        if is_talking_head:
            return await self._generate_talking_head_chunk(
                chunk_index=chunk_index,
                blocks=blocks,
                total_duration=total_duration,
                global_style=global_style
            )
        else:
            return await self._generate_pure_ai_chunk(
                chunk_index=chunk_index,
                blocks=blocks,
                total_duration=total_duration,
                combined_description=combined_description,
                global_style=global_style
            )
        

    async def _generate_pure_ai_chunk(
        self,
        chunk_index: int,
        blocks: List[Dict],
        total_duration: float,
        combined_description: str,
        global_style: str
    ) -> Dict:
        # 合并描述
        full_prompt = (
            f"[风格要求: {global_style}] "
            f"请生成一个连续的视频片段，包含以下场景：{combined_description}"
        )

        # 切分为 5s 分镜（基于总时长）
        segment_duration = 5.0
        num_segments = int(total_duration / segment_duration) + (1 if total_duration % segment_duration > 0 else 0)

        generated_clips = []

        for seg_idx in range(num_segments):
            seg_start = seg_idx * segment_duration
            seg_end = min(seg_start + segment_duration, total_duration)
            seg_len = seg_end - seg_start

            segment_prompt = f"{full_prompt} (第{seg_idx+1}段，{seg_len:.1f}秒)"

            # 生成分镜图
            storyboard_image = await self._call_image_generation_api(segment_prompt)

            # 图生视频
            video_clip = await self._call_image_to_video_api(
                image=storyboard_image,
                prompt=segment_prompt,
                duration=seg_len
            )

            generated_clips.append({
                "clip_id": f"chunk_{chunk_index}_seg_{seg_idx}",
                "url": video_clip["url"],
                "duration": seg_len,
                "prompt": segment_prompt
            })

        # 拼接
        final_video = await self._concatenate_videos(generated_clips)

        return {
            "type": "ai_video_chunk",
            "source": "ai_generation",
            "url": final_video["url"],
            "duration": total_duration,
            "clips": generated_clips,
            "style": global_style,
            "generated_type": "pure_ai_chunk",
            "original_indices": [i for i in range(indices[0], indices[-1]+1)],
            "chunk_id": f"ai_chunk_{indices[0]}_{indices[-1]}"
        }
        
    async def _generate_talking_head_chunk(
        self,
        chunk_index: int,
        blocks: List[Dict],
        total_duration: float,
        global_style: str
    ) -> Dict:
        # 合并脚本
        scripts = []
        for block in blocks:
            script = block.get("script") or block.get("description")
            scripts.append(script)
        full_script = "。".join(scripts)

        # 选择数字人
        digital_human = await self._select_best_digital_human(full_script, global_style)

        # TTS
        voice_audio = await self._call_tts_api(
            text=full_script,
            voice=digital_human["voice_id"],
            speed=1.0
        )

        # 驱动数字人（使用同一视频源）
        talking_video = await self._call_talking_avatar_api(
            avatar_id=digital_human["avatar_id"],
            audio_url=voice_audio["url"],
            background=digital_human.get("background", "studio"),
            style=global_style
        )

        return {
            "type": "talking_head_chunk",
            "source": "digital_human",
            "url": talking_video["url"],
            "duration": total_duration,
            "digital_human_used": digital_human["name"],
            "script": full_script,
            "generated_type": "talking_head_chunk",
            "style": global_style,
            "original_indices": [i for i in range(indices[0], indices[-1]+1)],
            "chunk_id": f"talking_chunk_{indices[0]}_{indices[-1]}"
        }
    # async def _generate_pure_ai_video(
    #     self,
    #     block: Dict,
    #     index: int,
    #     global_style: str
    # ) -> Dict:
    #     duration = block.get("duration", 3.0)
    #     description = block["description"]

    #     # 强制注入风格
    #     prompt = f"[风格要求: {global_style}] {description}"

    #     # 切分为 5s 一段的分镜
    #     segment_duration = 5.0
    #     num_segments = int(duration / segment_duration) + (1 if duration % segment_duration > 0 else 0)

    #     generated_clips = []
    #     total_duration = 0.0

    #     for seg_idx in range(num_segments):
    #         seg_start = seg_idx * segment_duration
    #         seg_end = min(seg_start + segment_duration, duration)
    #         seg_len = seg_end - seg_start

    #         # 生成分镜描述
    #         segment_prompt = f"{prompt} (第{seg_idx+1}段，{seg_len:.1f}秒)"

    #         # Step 1: 生成分镜图（Image-to-Video 基础）
    #         storyboard_image = await self._call_image_generation_api(segment_prompt)

    #         # Step 2: 用分镜图生成视频
    #         video_clip = await self._call_image_to_video_api(
    #             image=storyboard_image,
    #             prompt=segment_prompt,
    #             duration=seg_len
    #         )

    #         generated_clips.append({
    #             "clip_id": f"ai_clip_{index}_{seg_idx}",
    #             "url": video_clip["url"],
    #             "duration": seg_len,
    #             "type": "generated_video",
    #             "source": "ai",
    #             "style": global_style,
    #             "prompt": segment_prompt
    #         })

    #         total_duration += seg_len

    #     # Step 3: 拼接视频（可调用 FFmpeg 或云服务）
    #     final_video = await self._concatenate_videos(generated_clips)

    #     return {
    #         "type": "ai_video",
    #         "source": "ai_generation",
    #         "url": final_video["url"],
    #         "duration": total_duration,
    #         "clips": generated_clips,  # 保留分镜信息
    #         "style": global_style,
    #         "generated_type": "pure_ai",
    #         "original_block_index": index
    #     }
    
    # async def _generate_talking_head_video(
    #     self,
    #     block: Dict,
    #     index: int,
    #     global_style: str
    # ) -> Dict:
    #     duration = block.get("duration", 3.0)
    #     script = block.get("script") or block.get("description")

    #     # Step 1: 选择最合适的数字人
    #     digital_human = await self._select_best_digital_human(script, global_style)

    #     # Step 2: 文本转语音（TTS）
    #     voice_audio = await self._call_tts_api(
    #         text=script,
    #         voice=digital_human["voice_id"],
    #         speed=1.0
    #     )

    #     # Step 3: 驱动数字人生成视频（使用同一视频源）
    #     talking_video = await self._call_talking_avatar_api(
    #         avatar_id=digital_human["avatar_id"],
    #         audio_url=voice_audio["url"],
    #         background=digital_human.get("background", "studio"),
    #         style=global_style
    #     )

    #     return {
    #         "type": "talking_head",
    #         "source": "digital_human",
    #         "url": talking_video["url"],
    #         "duration": duration,
    #         "digital_human_used": digital_human["name"],
    #         "voice_id": digital_human["voice_id"],
    #         "avatar_id": digital_human["avatar_id"],
    #         "script": script,
    #         "generated_type": "talking_head",
    #         "style": global_style,
    #         "original_block_index": index
    #     }
    
    async def _select_best_digital_human(self, script: str, required_style: str) -> Dict:
        # 可根据风格、性别、年龄、语种匹配
        candidates = [
            {"name": "Alex_Pro", "avatar_id": "alex_v1", "voice_id": "en-US-1", "style": "professional"},
            {"name": "Ling_Chinese", "avatar_id": "ling_v2", "voice_id": "zh-CN-2", "style": "warm"},
            {"name": "Eva_Cartoon", "avatar_id": "eva_cartoon", "voice_id": "en-US-3", "style": "cartoon"},
        ]
        # 匹配风格
        for cand in candidates:
            if required_style in cand["style"] or cand["style"] in required_style:
                return cand
        return candidates[0]  # 默认
    
    def _is_style_compatible(self, style_a: str, style_b: str) -> bool:
        if not style_a or not style_b:
            return False
        style_a, style_b = style_a.lower(), style_b.lower()
        # 可扩展为风格映射表
        compatibility_map = {
            "cinematic": ["cinematic", "realistic", "documentary"],
            "anime": ["anime", "cartoon"],
            "realistic": ["realistic", "cinematic", "documentary"],
            "documentary": ["documentary", "realistic"],
            "cyberpunk": ["cyberpunk", "cinematic"],
        }
        return style_b in compatibility_map.get(style_a, [style_a])
    
    async def _search_asset_library_with_style(self, block: Dict, required_style: str) -> Optional[Dict]:
        payload = {
            "query": block["description"],
            "filters": {"style": required_style},
            "return_count": 1
        }
        try:
            resp = requests.post(
                self.system_parameters["asset_library_url"],
                json=payload,
                timeout=10
            )
            if resp.status_code == 200:
                results = resp.json().get("results", [])
                if results:
                    asset = results[0]
                    block["asset_status"] = "matched"
                    block["scheduled_asset"] = asset
                    block["source_url"] = asset["url"]
                    return block
        except Exception as e:
            print(f"[Asset Search] 失败: {e}")
        return None
    
    def _finalize_user_asset_block(self, block: Dict) -> Dict:
        asset = block["scheduled_asset"]
        block["media_type"] = "user_asset"
        block["verified"] = True
        block["style_aligned"] = True
        block["source_url"] = asset["url"]
        return block
    
    def _recalculate_timeline(self, sequence: List[Dict]) -> float:
        time = 0.0
        for i, block in enumerate(sequence):
            time += block.get("duration", 3.0)
            if i < len(sequence) - 1:
                time += 0.5
        return time
    
    
    
    def _create_placeholder_asset(self, block: Dict) -> Dict:
        """可返回一个默认素材，或标记为待生成"""
        return {
            "type": "placeholder",
            "description": block["description"],
            "duration": block.get("duration", 3.0),
            "style": "unknown",
            "source": "library_fallback"
        }