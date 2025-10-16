# nodes/asset_request_node.py
"""
素材请求与生成节点 (Node 5 in VGP Workflow)

这是视频生成流程中的核心素材节点，负责：
1. 对接素材供给系统（materials_supplies）
2. 智能匹配并生成所有类型的素材（视频、图片、音频等）
3. 将生成的素材信息传递给后续节点进行处理

工作流位置：
- 输入：来自 node_3_shot_blocks (分镜块生成节点)
- 输出：传递给 node_6_filter_application (滤镜应用节点)

重要说明：
- 此节点是整个工作流中素材生成的核心环节
- 调用 match_intelligent_video() 进行智能素材匹配
- 后续节点（6-8）基于生成的素材进行各种处理（滤镜、效果、转场）
- 节点9-15只负责生成对应类型的素材，不进行最终合成
- 最终合成在 node_16_timeline_integration 中完成
"""

from video_generate_protocol import BaseNode
import logging

logger = logging.getLogger(__name__)

from typing import Dict, List, Any
import requests
import uuid
import time
from datetime import datetime
import asyncio

from materials_supplies import match_intelligent_video

# TTS 语速（字/秒）
CHARS_PER_SECOND = 4.0


class AssetRequestNode(BaseNode):
    """
    素材请求与生成节点

    这是VGP工作流中的核心素材节点（Node 5），负责：
    1. 接收分镜块列表（shot_blocks）
    2. 估算每个分镜的时长
    3. 调用素材供给系统智能匹配素材
    4. 返回包含素材信息的初步序列

    此节点生成的素材将被后续节点使用和处理。
    """
    required_inputs = [
        {
            "name": "shot_blocks_id",
            "label": "分镜块列表",
            "type": list,
            "required": True,
            "desc": "包含分镜描述的结构化列表",
            "field_type": "json"
        },
        {
            "name": "user_description_id",
            "label": "用户原始输入",
            "type": str,
            "required": True,
            "default": "",
            "desc": "用户原始输入",
            "field_type": "textarea"
        },
        {
            "name": "tts_text_id",
            "label": "语音文本（用于时长估算）",
            "type": str,
            "required": False,
            "desc": "用于估算镜头时长的完整语音文本",
            "field_type": "text"
        },
        ##TODO:这里视频描述要兼容链接
        {
            "name": "video_description_id",
            "label": "用户视频描述（可选）",
            "type": str,
            "required": False,
            "desc": "如果用户上传了参考视频，提供其内容描述，用于优先匹配",
            "field_type": "text"
        }
    ]

    output_schema=[
         {
            "name": "preliminary_sequence_id",
            "label": "分镜块列表",
            "type": list,
            "required": True,
            "desc": "包含镜头信息的剪辑序列，如 [{'shot_id': 's1', 'shot_type': '全景', 'pacing': '快剪', 'emotion_hint': '激昂'}]",
            "field_type": "json"
        },
        {
            "name": "total_main_duration_id",
            "label": "视频主片段总时长",
            "type": float,
            "required": True,
            "desc": "视频主片段总时长",
            "field_type": "number"
        }
    ]

    system_parameters = {
        "default_duration": 5.0,
        "use_tts_duration": True,
        "apply_default_transition": True,
        "transition_duration": 0.5,
        "match_user_video_strategy": "all"  # 或 "keyword_based" 等策略
    }

    def __init__(self, node_id: str, name: str = "素材需求解析"):
        super().__init__(node_id=node_id, node_type="asset", name=name)


    

    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_context(context)

        # 提取输入
        shot_blocks = context.get("shot_blocks_id", [])
        tts_text = context.get("tts_text_id", "").strip()
        video_description = context.get("video_description_id", "").strip() or None

        if not shot_blocks:
            raise ValueError("缺少输入：shot_blocks 不能为空")

        # === 第一步：预估每个分镜的时长，并处理用户视频匹配标记 ===
        processed_shot_blocks = []
        current_time = 0.0

        for idx, block in enumerate(shot_blocks):
            block = block.copy()  # 避免修改原 context
            description = block.get("visual_description", "").strip()
            if not description:
                continue

            # --- 1. 计算 duration ---
            if "duration" in block:
                duration = float(block["duration"])
            elif self.system_parameters["use_tts_duration"] and tts_text:
                total_chars = len(tts_text)
                block_chars = len(description)
                estimated_duration = (block_chars / max(total_chars, 1)) * (len(tts_text) / CHARS_PER_SECOND)
                duration = max(2.0, estimated_duration)
            else:
                duration = self.system_parameters["default_duration"]

            block["duration"] = duration

            # --- 2. 判断是否要强制匹配用户上传视频 ---
            # 这里可以根据策略决定：比如全部匹配、或只匹配包含某些关键词的
            should_match_user_video = False
            strategy = self.system_parameters.get("match_user_video_strategy", "none")

            if video_description and strategy == "all":
                should_match_user_video = True
            # elif strategy == "keyword_based":
            #     should_match_user_video = any(kw in description for kw in ["我拍摄", "我的视频", "实拍"])

            if should_match_user_video:
                block["asset_status"] = "matched"
                block["scheduled_asset"] = {
                    "source": "user_upload",
                    "description": video_description,
                    "reason": f"Forced by video_description match (block {idx})"
                }

            processed_shot_blocks.append(block)

            current_time += duration
            if (self.system_parameters["apply_default_transition"] and
                    idx < len(processed_shot_blocks) - 1):
                current_time += self.system_parameters["transition_duration"]

        # === 第二步：将处理后的 shot_blocks 写回 context，传给 match_intelligent_video ===
        # 注意：match_intelligent_video 接收的是完整 context，不是单独 shot_list
        enhanced_context = context.copy()
        enhanced_context["shot_blocks_id"] = processed_shot_blocks
        # 可选：添加 metadata
        enhanced_context["asset_request_metadata"] = {
            "node_id": self.node_id,
            "processed_at": datetime.now().isoformat(),
            "total_blocks": len(processed_shot_blocks)
        }

        # === 第三步：调用异步智能匹配（传入完整 context）===
        try:
            match_result = await match_intelligent_video(enhanced_context)
        except Exception as e:
            logger.info(f"❌ 调用 match_intelligent_video 失败: {str(e)}")
            match_result = {"success": False}

        if not match_result or not match_result.get("success"):
            logger.info(f"⚠️ 智能素材匹配失败，使用占位符序列")
            preliminary_sequence = self._create_placeholder_sequence(processed_shot_blocks)
        else:
            preliminary_sequence = match_result.get("data", {}).get("clips", [])
            # 可选：校验返回结构

        # === 第四步：根据目标时长计算需要的视频段数 ===
        import os
        import sys
        import math
        from urllib.parse import quote
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

        # 获取目标时长
        target_duration = context.get("target_duration_id", 60)

        video_clips = []
        keyframes = []

        try:
            # ✅ 使用万相图生视频 API 生成（OSS预存视频已注释，用于节省成本）
            # ========== OSS 预存视频部分（已注释） ==========
            # oss_base_url = "https://ai-movie-cloud-v2.oss-cn-shanghai.aliyuncs.com/aura_render_temp/"
            # available_oss_videos = 6
            # use_oss_videos = available_oss_videos > 0
            # if use_oss_videos:
            #     logger.info(f"✅ [Node 5] 使用 OSS 视频素材，共 {available_oss_videos} 个可用")
            #     current_time = 0.0
            #     for idx in range(videos_needed):
            #         video_idx = (idx % available_oss_videos) + 1
            #         video_filename = f"视频{video_idx}.mp4"
            #         video_url = oss_base_url + quote(video_filename)
            #         video_clip = {...}
            #         video_clips.append(video_clip)
            # ========== OSS 部分结束 ==========

            # ✅ 使用完整的一致性工作流（Node 4A/4B/4C + 图生视频）
            # 使用所有生成的镜头，不要截断（每个镜头生成一段5秒视频）
            selected_shot_blocks = processed_shot_blocks

            logger.info(f"🎯 [Node 5] 目标时长 {target_duration}秒")
            logger.info(f"🎨 [Node 5] 使用所有 {len(selected_shot_blocks)} 个镜头生成视频（含一致性保障）")
            logger.info(f"📊 [Node 5] 预计生成 {len(selected_shot_blocks)} 段视频，总时长约 {len(selected_shot_blocks) * 5}秒")

            # === 步骤1: Node 4A - 提取全局视觉基因 ===
            from video_generate_protocol.nodes.visual_dna_node import VisualDNANode

            node_4a = VisualDNANode(node_id="node_4a", name="全局视觉基因提取")
            visual_dna_context = {
                "user_description_id": context.get("user_description_id", ""),
                "shot_blocks_id": selected_shot_blocks,
                "reference_media": context.get("reference_media", {})  # ✅ 传递产品图片
            }
            visual_dna_result = await node_4a.generate(visual_dna_context)
            visual_dna = visual_dna_result["visual_dna_id"]

            # === 步骤2: Node 4B - 首帧细化 ===
            from video_generate_protocol.nodes.keyframe_refinement_node import KeyframeRefinementNode

            node_4b = KeyframeRefinementNode(node_id="node_4b", name="首帧细化")
            keyframe_context = {
                "shot_blocks_id": selected_shot_blocks,
                "visual_dna_id": visual_dna
            }
            keyframe_result = await node_4b.generate(keyframe_context)
            refined_keyframes = keyframe_result["refined_keyframes_id"]

            # === 步骤3: Node 4C - 一致性检查 ===
            from video_generate_protocol.nodes.consistency_check_node import ConsistencyCheckNode

            node_4c = ConsistencyCheckNode(node_id="node_4c", name="一致性检查")
            consistency_context = {
                "refined_keyframes_id": refined_keyframes,
                "reference_media": context.get("reference_media", {})  # ✅ 传递产品图片
            }
            consistency_result = await node_4c.generate(consistency_context)
            keyframes_with_strategy = consistency_result["keyframes_with_strategy_id"]

            # === 步骤4: 提取产品图片URL（如果有）===
            product_image_url = None
            reference_media = context.get("reference_media", {})
            if reference_media:
                product_images = reference_media.get("product_images", [])
                if product_images and len(product_images) > 0:
                    product_image_url = product_images[0].get("url")
                    logger.info(f"📦 [Node 5] 检测到产品参考图: {product_image_url}")

            # === 步骤5: 使用万相图生视频（支持图生图） ===
            from video_generate_protocol.nodes.qwen_integration import StoryboardToVideoProcessor

            qwen_key = os.getenv('DASHSCOPE_API_KEY') or os.getenv('AI__DASHSCOPE_API_KEY')
            if not qwen_key:
                raise ValueError("缺少千问API密钥")

            video_processor = StoryboardToVideoProcessor(qwen_key)

            logger.info(f"🎥 [Node 5] Generating {len(keyframes_with_strategy)} video clips with consistency...")
            video_clips = await video_processor.process_keyframes_with_consistency(
                keyframes_with_strategy,
                f"/tmp/video_clips_node5_{uuid.uuid4().hex[:8]}",
                product_image_url=product_image_url  # 传递产品图片
            )
            logger.info(f"✅ [Node 5] Generated {len(video_clips)} video clips from {len(selected_shot_blocks)} shots (target duration: {target_duration}s)")

            # 保存关键帧信息（供调试）
            keyframes = keyframes_with_strategy

            # 打印所有视频片段URL汇总
            if video_clips:
                logger.info(f"\n{'='*80}")
                logger.info(f"📹 生成的视频片段URL汇总 (共{len(video_clips)}个):")
                logger.info(f"{'='*80}")
                for idx, clip in enumerate(video_clips, 1):
                    logger.info(f"{idx}. {clip.get('url', 'N/A')}")
                logger.info(f"{'='*80}\n")

        except Exception as e:
            logger.info(f"❌ [Node 5] Video generation failed: {e}")
            import traceback
            traceback.print_exc()
            # 视频生成失败时，继续返回匹配结果，但标记失败
            video_clips = []
            keyframes = []

        return {
            "preliminary_sequence_id": preliminary_sequence,
            "total_main_duration_id": current_time,
            "video_clips": video_clips,  # 生成的视频片段列表
            "keyframes": keyframes,  # 生成的关键帧列表
            "video_generation_success": len(video_clips) > 0,  # 视频生成是否成功
            # "timestamp_id": datetime.now().isoformat(),
            # "processed_shot_blocks": processed_shot_blocks,  # 调试用
            # "video_description_matched": bool(video_description)  # 日志标记
        }

    def _create_placeholder_sequence(self, shot_blocks: List[Dict]) -> List[Dict]:
        """生成占位序列（用于 fallback）"""
        sequence = []
        current_time = 0.0
        transition_dur = self.system_parameters["transition_duration"] if self.system_parameters["apply_default_transition"] else 0

        for idx, block in enumerate(shot_blocks):
            duration = block["duration"]
            clip = {
                "id": f"clip_{uuid.uuid4().hex[:8]}",
                "index": idx,
                "asset_id": f"placeholder_{idx}",
                "source_url": "https://example.com/assets/placeholder.mp4",
                "start": round(current_time, 3),
                "end": round(current_time + duration, 3),
                "duration": round(duration, 3),
                "source": {"in": 0.0, "out": round(duration, 3)},
                "transition_in": {
                    "type": "cross_dissolve",
                    "duration": transition_dur if idx > 0 else 0
                },
                "transition_out": {
                    "type": "cross_dissolve",
                    "duration": transition_dur if idx < len(shot_blocks) - 1 else 0
                },
                "metadata": {
                    "description": block.get("visual_description", "")[:100],
                    "provider": "system_placeholder",
                    "original_block": block
                },
                "transform": {"scale": 1.0, "position": "center"}
            }
            sequence.append(clip)
            current_time += duration
            if idx < len(shot_blocks) - 1:
                current_time += transition_dur

        return sequence

    # regenerate 可后续改为 async，或仅用于调试
    def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"⚠️ regenerate 暂不支持异步流程，建议在上层统一使用 async generate")
        return self._create_placeholder_sequence(context.get("shot_blocks_id", []))