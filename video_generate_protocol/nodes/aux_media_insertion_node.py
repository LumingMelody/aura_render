# nodes/aux_media_insertion_node.py

from video_generate_protocol import BaseNode
from typing import Dict, List, Any
import re
import os
import json
from datetime import datetime

from materials_supplies import SupplementRequest, SupplementResponse,match_supplement
from llm import QwenLLM  # 假设这是你封装好的 Qwen 调用模块
import asyncio
import hashlib

# 模拟媒体素材库（实际项目中可对接数据库或向量检索）
AUX_MEDIA_LIBRARY = [
    {
        "id": "img_map_001",
        "keywords": "地图|位置|城市|路线|地理",
        "type": "image",
        "category": "map",
        "file_path": "assets/maps/world_map.jpg",
        "duration": 5.0,
        "suggested_duration": 4.0,
        "tags": ["地理", "导航", "探索"],
        "relevance_score": 0.9,
        "usage_count": 120
    },
    {
        "id": "vid_historical_002",
        "keywords": "历史|过去|老照片|黑白|年代",
        "type": "video",
        "category": "archive",
        "file_path": "assets/videos/archival_footage_1940s.mp4",
        "duration": 15.0,
        "suggested_duration": 8.0,
        "tags": ["历史", "纪录片"],
        "relevance_score": 0.85,
        "usage_count": 95
    },
    {
        "id": "img_chart_003",
        "keywords": "数据|统计|图表|增长|分析",
        "type": "image",
        "category": "data_visualization",
        "file_path": "assets/charts/bar_chart_sample.png",
        "duration": 6.0,
        "suggested_duration": 5.0,
        "tags": ["数据", "商业", "教育"],
        "relevance_score": 0.92,
        "usage_count": 130
    },
    {
        "id": "vid_atmosphere_004",
        "keywords": "氛围|自然|森林|海浪|宁静",
        "type": "video",
        "category": "atmosphere",
        "file_path": "assets/videos/forest_dawn_loop.mp4",
        "duration": 30.0,
        "suggested_duration": 10.0,
        "tags": ["氛围", "冥想", "背景"],
        "relevance_score": 0.8,
        "usage_count": 88,
        "loopable": True
    },
    {
        "id": "img_timeline_005",
        "keywords": "时间线|年份|事件|发展|历程",
        "type": "image",
        "category": "timeline",
        "file_path": "assets/timelines/timeline_modern_history.png",
        "duration": 8.0,
        "suggested_duration": 6.0,
        "tags": ["时间线", "教育", "叙事"],
        "relevance_score": 0.88,
        "usage_count": 105
    },
    {
        "id": "vid_reenactment_006",
        "keywords": "重现|模拟|实验|过程|演示",
        "type": "video",
        "category": "reenactment",
        "file_path": "assets/videos/science_experiment_sim.mp4",
        "duration": 20.0,
        "suggested_duration": 12.0,
        "tags": ["演示", "科学", "教学"],
        "relevance_score": 0.9,
        "usage_count": 110
    }
]

# 视频类型对应的默认推荐（如纪录片多用历史影像）
GENRE_TO_MEDIA_HINT = {
    "documentary": ["archive", "timeline", "map"],
    "educational": ["data_visualization", "timeline", "reenactment"],
    "tech_review": ["data_visualization", "atmosphere"],
    "vlog": ["atmosphere", "map"],
    "storytelling": ["atmosphere", "archive"]
}

# 默认建议时长（秒）
DEFAULT_SUGGESTED_DURATION = 5.0


class AuxMediaInsertionNode(BaseNode):
   
    required_inputs = [
        {
            "name": "shot_blocks_id",
            "label": "分镜块描述",
            "type": List[Dict],
            "required": True,
            "desc": "包含镜头类型、时长、视觉描述、节奏和字幕的分镜列表",
            "field_type": "json",
            "example": [
                {
                    "shot_type": "中景",
                    "duration": 8,
                    "visual_description": "讲师站在白板前微笑，手指向屏幕上的课程总结要点",
                    "pacing": "常规",
                    "caption": "我们已经走过了这段旅程的关键点。"
                }
            ]
        },
        {
            "name": "video_genre",
            "label": "视频类型",
            "type": str,
            "required": False,
            "desc": "视频风格类型，用于推荐合适的辅助媒体类别，如 'educational', 'documentary', 'vlog'",
            "field_type": "text",
            "options": ["educational", "documentary", "tech_review", "vlog", "storytelling"]
        },
        {
            "name": "outline",
            "label": "视频大纲",
            "type": str,
            "required": False,
            "desc": "整体叙事结构，帮助理解上下文和关键信息点",
            "field_type": "textarea"
        }
    ]

    output_schema=[
        {
            "name": "auxiliary_track_id",
            "label": "额外媒体轨道",
            "type": str,
            "desc": "额外媒体轨道，如补图，说明性视频等"
        },
       
    ]

    file_upload_config = {
        "image": {
            "enabled": True,
            "accept": ".jpg,.jpeg,.png,.gif,.svg",
            "desc": "可上传自定义图片"
        },
        "video": {
            "enabled": True,
            "accept": ".mp4,.mov,.avi,.webm",
            "desc": "可上传自定义视频"
        }
    }

    system_parameters = {
        "max_clips_per_scene": 1,
        "min_match_score": 0.7,
        "default_placement": "overlay",
        "default_position": "center"
    }

    def __init__(self, node_id: str, name: str = "额外插入图片/视频"):
        super().__init__(node_id=node_id, node_type="aux_media_insertion", name=name)
        self.qwen = QwenLLM()  # 使用你封装的 Qwen

    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        支持异步的 generate 方法（因为 match_supplement 是 async）
        """
        self.validate_context(context)
        shot_blocks: List[Dict] = context["shot_blocks_id"]
        video_genre: str = context.get("video_genre", "educational")
        outline: str = context.get("outline", "")

        aux_track = {
            "track_name": "auxiliary",
            "track_type": "overlay_media",
            "clips": []
        }

        current_time = 0.0

        for block in shot_blocks:
            duration = block["duration"]
            end_time = current_time + duration
            visual_desc = block["visual_description"]
            caption = block.get("caption", "")
            full_desc = visual_desc + " " + caption

            # ✅ 第一步：用 Qwen 判断是否需要插入辅助媒体
            if not await self._should_insert_aux_media(block):
                current_time = end_time
                continue

            # ✅ 第二步：决定插入什么类型的素材（可基于 genre 或描述）
            category = self._infer_category_from_genre(video_genre)
            request = SupplementRequest(
                description=full_desc,
                category=category,
                duration=duration
            )

            # ✅ 第三步：调用素材系统匹配
            supplements = await match_supplement(request)
            supplements = [s for s in supplements if s.match_score >= self.system_parameters["min_match_score"]]
            supplements = supplements[:self.system_parameters["max_clips_per_scene"]]

            for supp in supplements:
                clip = self._create_clip_from_response(supp, current_time, duration)
                aux_track["clips"].append(clip)

            current_time = end_time

        return {"auxiliary_track_id": aux_track}


    async def _should_insert_aux_media(self, shot: Dict[str, str]) -> bool:
        prompt = f"""
你是一个视频叙事结构专家。请分析以下镜头描述，判断是否适合插入一张辅助性视觉元素（如图表、示意图、图标、地图、数据可视化等）来增强信息传达。

如果适合，请回答“是”；否则回答“否”。

镜头类型：{shot['shot_type']}
视觉描述：{shot['visual_description']}
字幕：{shot.get('caption', '')}
节奏：{shot['pacing']}

请只回答“是”或“否”。
"""
        response = self.qwen.generate(prompt=prompt)
        return "是" in response.strip()

    def _infer_category_from_genre(self, genre: str) -> str:
        """根据视频类型推断推荐的素材类别"""
        mapping = {
            "documentary": "archive",
            "educational": "data_visualization",
            "tech_review": "data_visualization",
            "vlog": "atmosphere",
            "storytelling": "atmosphere"
        }
        return mapping.get(genre, "overlay")

    def _create_clip_from_response(self, supp: SupplementResponse, start_time: float, scene_duration: float) -> Dict:
        """将 SupplementResponse 转为轨道片段"""
        duration = min(supp.duration, scene_duration * 0.8)  # 不超过镜头 80%
        return {
            "media_id": f"supp_{hash(supp.url) % 10000}",
            "title": supp.url.split("/")[-1].split(".")[0],
            "file_path": supp.url,
            "thumbnail": supp.thumbnail,
            "start": start_time,
            "duration": duration,
            "type": supp.media_type,
            "category": self._infer_category_from_genre("educational"),  # 可优化
            "placement": self.system_parameters["default_placement"],
            "position": self.system_parameters["default_position"],
            "opacity": 0.95 if supp.media_type == "image" else 1.0,
            "source": "material_service",
            "match_score": supp.match_score
        }


    def _load_custom_media(self) -> List[Dict]:
        """加载用户上传的自定义图片/视频"""
        custom_list = []
        if not self.uploaded_files:
            return custom_list

        for file_info in self.uploaded_files:
            file_type = file_info["type"]
            filename = file_info["filename"]

            if file_type == "image" and filename.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
                custom_list.append({
                    "id": f"custom_img_{len(custom_list)+1}",
                    "keywords": filename.split(".")[0],
                    "type": "image",
                    "category": "custom_image",
                    "file_path": file_info["path"],
                    "duration": 5.0,
                    "suggested_duration": 4.0,
                    "relevance_score": 0.95,
                    "source": "uploaded"
                })
            elif file_type == "video" and filename.lower().endswith(('.mp4', '.mov', '.avi')):
                custom_list.append({
                    "id": f"custom_vid_{len(custom_list)+1}",
                    "keywords": filename.split(".")[0],
                    "type": "video",
                    "category": "custom_video",
                    "file_path": file_info["path"],
                    "duration": 10.0,
                    "suggested_duration": 8.0,
                    "relevance_score": 0.95,
                    "source": "uploaded"
                })

        return custom_list

    def _match_media(self, desc: str, library: List[Dict], genre_hints: List[str]) -> List[Dict]:
        """根据描述和类型匹配辅助媒体"""
        matched = []

        for media in library:
            # 关键词匹配
            if re.search(media["keywords"], desc, re.IGNORECASE):
                score = media["relevance_score"]
                # 类型加权
                if media["category"] in genre_hints:
                    score += 0.1  # 类型匹配则加分
                matched.append((media, score))

        # 按相关性排序
        matched.sort(key=lambda x: x[1], reverse=True)

        # 过滤低分项
        min_score = self.system_parameters["min_relevance_score"]
        return [item[0] for item in matched if item[1] >= min_score]

    def _create_aux_clip(self, media: Dict, start_time: float, scene_duration: float) -> Dict:
        """创建辅助媒体片段"""
        # 计算插入时间（可微调提前/延后）
        insert_time = start_time  # 默认从镜头开始插入

        # 计算时长
        if self.system_parameters["auto_duration"]:
            suggested = media.get("suggested_duration", DEFAULT_SUGGESTED_DURATION)
            duration = min(suggested, scene_duration * 0.8)  # 不超过镜头时长80%
        else:
            duration = self.system_parameters["default_duration"]

        # 确保不超时
        duration = min(duration, media["duration"])

        return {
            "media_id": media["id"],
            "title": media.get("title", media["keywords"].split("|")[0]),
            "file_path": media["file_path"],
            "start": insert_time,
            "duration": duration,
            "type": media["type"],
            "category": media["category"],
            "tags": media.get("tags", []),
            "placement": "overlay",  # 可为 overlay, pip (画中画), transition 等
            "position": "center",   # 或 "top-left", "bottom-right"
            "opacity": 0.95 if media["type"] == "image" else 1.0,
            "source": media.get("source", "library")
        }

    # def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
    #     """支持用户干预"""
    #     super().regenerate(context, user_intent)

    #     override = user_intent.get("aux_override")
    #     if not override:
    #         return self.generate(context)

    #     result = self.generate(context)
    #     aux_track = result["auxiliary_track"]

    #     # 手动添加
    #     if "add_manual_clip" in override:
    #         manual = override["add_manual_clip"]
    #         aux_track["clips"].append({
    #             "media_id": f"manual_{len(aux_track['clips'])+1}",
    #             "title": manual["title"],
    #             "file_path": manual["file_path"],
    #             "start": manual["time"],
    #             "duration": manual.get("duration", 5.0),
    #             "type": manual["type"],
    #             "placement": manual.get("placement", "overlay"),
    #             "source": "manual"
    #         })

    #     # 移除某类
    #     if "remove_category" in override:
    #         category = override["remove_category"]
    #         aux_track["clips"] = [
    #             c for c in aux_track["clips"] if c["category"] != category
    #         ]

    #     return result
    
    
    def _hash_shot(shot: Dict) -> str:
        """为每个镜头生成唯一指纹，用于比较是否变化"""
        content = f"{shot.get('visual_description', '')}|{shot.get('caption', '')}|{shot.get('shot_type', '')}|{shot['duration']}"
        return hashlib.md5(content.encode('utf-8')).hexdigest()[:8]
    
    
    async def regenerate_async(
        self,
        context: Dict[str, Any],
        existing_result: Dict[str, Any],
        modified_shots: List[int] = None
    ) -> Dict[str, Any]:
        """
        增量重生成：仅对发生变化的镜头重新生成辅助媒体

        :param context: 新的上下文（含更新后的 shot_blocks）
        :param existing_result: 上一次 generate 的结果
        :param modified_shots: 明确指定哪些镜头索引被修改（可选）
        :return: 更新后的辅助媒体轨道
        """
        shot_blocks: List[Dict] = context["shot_blocks"]
        video_genre: str = context.get("video_genre", "educational")
        outline: str = context.get("outline", "")

        # 获取已有轨道，准备更新
        aux_track = existing_result.get("auxiliary_track", {
            "track_name": "auxiliary",
            "track_type": "overlay_media",
            "clips": []
        })

        # 记录每个镜头的起始时间
        current_time = 0.0
        shot_intervals = []
        for block in shot_blocks:
            duration = block["duration"]
            shot_intervals.append((current_time, current_time + duration))
            current_time += duration

        # 计算当前镜头哈希
        current_hashes = [self._hash_shot(block) for block in shot_blocks]

        # 解析已有 clips 的归属（哪个镜头）
        existing_clips_by_shot = {}  # {index: [clip, ...]}
        for clip in aux_track["clips"]:
            # 假设 clip 中记录了它所属的 time_range 或 shot_index（generate 时可添加）
            # 这里我们用时间区间反推属于哪个镜头
            clip_start = clip.get("start", 0.0)
            for idx, (s, e) in enumerate(shot_intervals):
                if s <= clip_start < e:
                    if idx not in existing_clips_by_shot:
                        existing_clips_by_shot[idx] = []
                    existing_clips_by_shot[idx].append(clip)
                    break

        # 确定需要重新生成的镜头索引
        if modified_shots is not None:
            indices_to_update = set(modified_shots)
        else:
            # 自动检测变化的镜头
            old_hashes = context.get("_old_shot_hashes", [])
            indices_to_update = {
                i for i in range(len(shot_blocks))
                if i >= len(old_hashes) or current_hashes[i] != old_hashes[i]
            }

        # 存储新 clips
        new_clips = []

        current_time = 0.0
        for idx, block in enumerate(shot_blocks):
            duration = block["duration"]
            end_time = current_time + duration

            if idx in indices_to_update:
                # 需要重新生成
                if await self._should_insert_aux_media(block):
                    full_desc = block["visual_description"] + " " + block.get("caption", "")
                    category = self._infer_category_from_genre(video_genre)
                    request = SupplementRequest(
                        description=full_desc,
                        category=category,
                        duration=duration
                    )
                    supplements = await match_supplement(request)
                    supplements = [s for s in supplements if s.match_score >= self.system_parameters["min_match_score"]]
                    supplements = supplements[:self.system_parameters["max_clips_per_scene"]]

                    for supp in supplements:
                        clip = self._create_clip_from_response(supp, current_time, duration)
                        new_clips.append(clip)
                # 注意：不保留旧 clip
            else:
                # 保留旧 clips
                if idx in existing_clips_by_shot:
                    offset = current_time - shot_intervals[idx][0]  # 时间轴偏移校正
                    for clip in existing_clips_by_shot[idx]:
                        adjusted_clip = clip.copy()
                        adjusted_clip["start"] += offset
                        new_clips.append(adjusted_clip)

            current_time = end_time

        # 更新哈希记录，供下次比较
        context["_old_shot_hashes"] = current_hashes

        aux_track["clips"] = new_clips
        return {"auxiliary_track": aux_track}
    
    def regenerate(
        self,
        context: Dict[str, Any],
        existing_result: Dict[str, Any],
        modified_shots: List[int] = None
    ) -> Dict[str, Any]:
        """
        同步入口：用于外部调用的 regenerate
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        return loop.run_until_complete(
            self.regenerate_async(context, existing_result, modified_shots)
        )