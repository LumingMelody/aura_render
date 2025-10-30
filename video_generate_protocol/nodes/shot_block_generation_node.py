# nodes/shot_block_generation_node.py

from video_generate_protocol import BaseNode
import logging

logger = logging.getLogger(__name__)

from video_generate_protocol.prompt_manager import get_prompt_manager
from typing import Dict, List, Any, Optional
import json
import json5
import re
import hashlib
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
from functools import wraps

# 假设你有一个 QwenLLM 类封装了大模型调用
# from llm import QwenLLM  # 示例导入，根据实际路径调整
from llm import QwenLLM  # 请确保这个模块存在

# ✨ 新增：导入12步提示词优化器
try:
    from video_generate_protocol.prompt_optimizer import VideoPromptOptimizer
    OPTIMIZER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"⚠️ 提示词优化器导入失败: {e}，将使用旧版分镜生成")
    OPTIMIZER_AVAILABLE = False


VIDEO_TYPE_STYLES = {
    # —————— 商业类 ——————
    "产品广告": {
        "tone": "高能、吸引眼球、突出卖点",
        "shots": "特写、慢镜头、对比镜头、动态展示",
        "pace": "快剪为主，关键点慢镜头强调",
        "caption": "简短有力的口号式文案，如‘快！准！狠！’"
    },
    "品牌宣传": {
        "tone": "大气、有故事感、有情感共鸣",
        "shots": "航拍、推镜头、人物群像、光影变化",
        "pace": "慢→快→慢，有节奏递进",
        "caption": "富有哲理或品牌精神的语句"
    },
    "促销视频": {
        "tone": "紧迫感、优惠强调、行动号召",
        "shots": "价格标签特写、人群抢购、倒计时、弹幕式文字",
        "pace": "极快剪辑，节奏紧凑",
        "caption": "‘限时5折！’‘仅剩100件！’等促销语言"
    },

    # —————— 教育类 ——————
    "知识讲解": {
        "tone": "清晰、逻辑强、沉稳",
        "shots": "中景讲解、PPT/图表叠加、标注动画",
        "pace": "常规，重点处暂停或慢放",
        "caption": "知识点标题或关键词，如‘关键：梯度下降’"
    },
    "技能教学": {
        "tone": "步骤清晰、可操作性强",
        "shots": "手部特写、分屏对比、画中画演示",
        "pace": "中等，配合操作节奏",
        "caption": "‘第一步：导入数据’‘注意：参数设置’"
    },
    "在线课程": {
        "tone": "亲和力强、有互动感",
        "shots": "讲师中景讲解+课件画中画",
        "pace": "平稳，留出思考时间",
        "caption": "章节标题、思考题、小测验提示"
    },

    # —————— 娱乐类 ——————
    "微电影": {
        "tone": "有剧情张力、情感丰富",
        "shots": "多角度切换、特写表情、空镜头过渡",
        "pace": "随情节变化，高潮快剪，抒情慢镜头",
        "caption": "对白字幕或旁白"
    },
    "短视频故事": {
        "tone": "反转强、开头抓人",
        "shots": "第一视角、快速转场、夸张表情",
        "pace": "极快，3秒内出冲突",
        "caption": "悬念式文字，如‘下一秒他惊呆了…’"
    },
    "动画短片": {
        "tone": "创意十足、风格化",
        "shots": "动态运镜、变形转场、色彩夸张",
        "pace": "灵活，配合音效节奏",
        "caption": "拟声词、角色对话气泡"
    },

    # —————— 社交类 ——————
    "VLOG": {
        "tone": "真实、生活化、有陪伴感",
        "shots": "手持拍摄感、自拍视角、环境空镜",
        "pace": "自然流动，少量快剪",
        "caption": "内心独白、时间地点标注"
    },
    "社交媒体内容": {
        "tone": "潮流、年轻化、易传播",
        "shots": "竖屏构图、滤镜特效、文字弹幕",
        "pace": "快，前3秒必须吸引人",
        "caption": "网络热词、挑战标签、互动提问"
    },
    "直播回放": {
        "tone": "还原现场、突出高光",
        "shots": "主播中景、观众反应、屏幕分享",
        "pace": "剪辑高光片段，跳过冗余",
        "caption": "‘高能预警！’‘此处有福利’"
    },

    # —————— 专业类 ——————
    "新闻播报": {
        "tone": "权威、简洁、客观",
        "shots": "主持人固定机位、新闻画面插播、字幕条",
        "pace": "稳定，每条新闻节奏一致",
        "caption": "时间、地点、事件关键词"
    },
    "访谈节目": {
        "tone": "深度、有交流感",
        "shots": "双人中景、单人特写切换、背景虚化",
        "pace": "随对话节奏，提问稍停顿",
        "caption": "嘉宾姓名+头衔、金句高亮"
    },
    "纪录片": {
        "tone": "真实、厚重、有旁白",
        "shots": "长镜头、航拍地理、历史资料画面",
        "pace": "缓慢推进，留白空间",
        "caption": "时间地点标注、旁白字幕"
    }
}

SUPPORTED_VIDEO_TYPES = list(VIDEO_TYPE_STYLES.keys())

# 缓存和重试机制
class ShotBlockCache:
    """分镜块生成结果缓存"""
    def __init__(self, max_size: int = 100, ttl: int = 3600):
        self.cache = {}
        self.timestamps = {}
        self.max_size = max_size
        self.ttl = ttl

    def _generate_key(self, emotions: Dict[str, int], video_type: str, structure: Dict[str, Any], target_duration: int = None) -> str:
        """生成缓存键"""
        duration_part = f"_{target_duration}" if target_duration else ""
        content = f"{json.dumps(emotions, sort_keys=True)}_{video_type}_{json.dumps(structure, sort_keys=True)}{duration_part}"
        return hashlib.md5(content.encode()).hexdigest()

    def get(self, emotions: Dict[str, int], video_type: str, structure: Dict[str, Any], target_duration: int = None) -> Optional[List[Dict]]:
        """获取缓存结果"""
        key = self._generate_key(emotions, video_type, structure, target_duration)

        if key not in self.cache:
            return None

        # 检查TTL
        if time.time() - self.timestamps[key] > self.ttl:
            self._remove(key)
            return None

        return self.cache[key]

    def set(self, emotions: Dict[str, int], video_type: str, structure: Dict[str, Any], result: List[Dict], target_duration: int = None):
        """设置缓存"""
        key = self._generate_key(emotions, video_type, structure, target_duration)

        # 如果缓存满了，删除最老的条目
        if len(self.cache) >= self.max_size:
            oldest_key = min(self.timestamps, key=self.timestamps.get)
            self._remove(oldest_key)

        self.cache[key] = result.copy()
        self.timestamps[key] = time.time()

    def _remove(self, key: str):
        """删除缓存条目"""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)

def async_retry_shot_gen(max_attempts: int = 3, delay: float = 1.0, backoff: float = 2.0):
    """异步重试装饰器"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(max_attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < max_attempts - 1:
                        wait_time = delay * (backoff ** attempt)
                        logger.info(f"⚠️ {func.__name__} 第{attempt + 1}次尝试失败: {e}")
                        logger.info(f"⏳ 等待 {wait_time:.1f}s 后重试...")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.info(f"❌ {func.__name__} 经过{max_attempts}次尝试后最终失败")

            raise last_exception
        return wrapper
    return decorator


class ShotBlockGenerationNode(BaseNode):

    required_inputs = [
        {
            "name": "emotions_id",
            "label": "情感分布",
            "type": dict,
            "required": True,
            "desc": "情感标签及权重，如 {'励志': 50, '冷静': 30}",
            "field_type": "json"
        },
        {
            "name": "structure_template_id",
            "label": "视频结构模板",
            "type": dict,
            "required": True,
            "desc": "视频结构模板",
            "field_type": "json"
        },
        {
            "name": "video_type_id",
            "label": "视频类型",
            "required": True,
            "type": str,
            "desc": "视频类型，如'宣传片'、'VLOG'等",
            "field_type": "json",
        },
    ]
    output_schema=[
         {
            "name": "shot_blocks_id",
            "label": "分镜块列表",
            "type": list,
            "required": True,
            "desc": "分镜块列表，包含视觉描述、字幕、节奏等信息，用于提取情感与音乐锚点",
            "field_type": "json"
        }
    ]

    file_upload_config = {
        "image": {"enabled": False},
        "video": {"enabled": False}
    }

    system_parameters = {
        "min_shot_duration": 2,
        "max_shot_duration": 10,
        "use_advanced_optimizer": True  # ✨ 新增：是否使用12步提示词优化器
    }

    def __init__(self, node_id: str, name: str = "分镜块生成"):
        super().__init__(node_id=node_id, node_type="shot_block_generation", name=name)

        # 初始化缓存
        self.cache = ShotBlockCache(max_size=100, ttl=3600)

        # ✨ 新增：初始化12步优化器
        self.optimizer = None
        if OPTIMIZER_AVAILABLE and self.system_parameters.get("use_advanced_optimizer", True):
            try:
                self.optimizer = VideoPromptOptimizer()
                # 只在非重载进程中打印日志，避免开发模式重复日志
                import os
                if os.environ.get('RUN_MAIN') != 'true':  # 只在主进程打印
                    logger.info("✅ 12步提示词优化器已启用")
            except Exception as e:
                logger.warning(f"⚠️ 优化器初始化失败: {e}，将使用旧版生成")
                self.optimizer = None

        # 性能统计
        self.stats = {
            "total_requests": 0,
            "cache_hits": 0,
            "llm_calls": 0,
            "fallback_calls": 0,
            "json_parse_failures": 0,
            "avg_response_time": 0.0,
            "optimizer_calls": 0  # ✨ 新增：优化器调用次数
        }

    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        start_time = time.time()
        self.stats["total_requests"] += 1

        try:
            # 验证输入上下文
            self.validate_context(context)

            # ✅ 修复：优先读取 target_duration_id，如果没有则尝试 target_duration
            total_duration = context.get("target_duration_id") or context.get("target_duration", 60)

            # ✨ 新增：如果启用了优化器，使用12步优化流程
            if self.optimizer:
                logger.info(f"🎨 使用12步提示词优化器生成分镜...")
                return await self._generate_with_optimizer(context, total_duration)

            # 原有的旧版生成逻辑
            logger.info(f"🎬 使用旧版分镜生成...")
            return await self._generate_legacy(context, total_duration)

        except Exception as e:
            logger.error(f"❌ ShotBlockGenerationNode.generate 失败: {e}")
            import traceback
            traceback.print_exc()
            # 使用fallback
            fallback_shots = self._fallback_shots("通用", 60, "冷静")
            self.stats["fallback_calls"] += 1
            return {"shot_blocks_id": fallback_shots}

        finally:
            # 更新性能统计
            response_time = time.time() - start_time
            self._update_stats(response_time)

    async def _generate_with_optimizer(self, context: Dict[str, Any], total_duration: int) -> Dict[str, Any]:
        """使用12步优化器生成分镜"""
        try:
            # 提取产品信息
            product_name = context.get("keywords_id", ["产品"])[0] if isinstance(context.get("keywords_id"), list) else "产品"
            user_description = context.get("user_description_id", "")

            # 调用优化器
            logger.info(f"📦 产品: {product_name}, 目标时长: {total_duration}秒")
            optimized_result = await self.optimizer.optimize(
                product_name=product_name,
                user_input=user_description,
                target_duration=total_duration  # ✅ 传递目标时长
            )

            self.stats["optimizer_calls"] += 1

            logger.info(f"✅ 优化器生成完成:")
            logger.info(f"   视觉风格: {optimized_result.visual_style.target_style}")
            logger.info(f"   分镜数量: {len(optimized_result.storyboard)}")
            logger.info(f"   总时长: {optimized_result.total_duration}秒")

            # 转换为节点期望的格式
            shot_blocks = []
            for shot in optimized_result.storyboard:
                # 使用优化后的细化描述
                visual_desc = shot.first_frame_clean or shot.first_frame_refined or shot.first_frame or shot.description

                shot_block = {
                    "shot_type": "特写",  # 默认类型，可以从description中推断
                    "duration": shot.duration,
                    "visual_description": visual_desc,
                    "pacing": "慢镜头" if shot.duration >= 3 else "常规",
                    "caption": shot.reason[:20] if shot.reason else "",  # 使用设计理由作为字幕
                    "start_time": shot_blocks[-1]["end_time"] if shot_blocks else 0.0,
                    "end_time": 0.0,  # 将在下面计算

                    # ✨ 新增：保存优化后的详细信息
                    "_optimized": {
                        "first_frame_refined": shot.first_frame_refined,
                        "middle_process_refined": shot.middle_process_refined,
                        "generation_strategy": shot.generation_strategy,
                        "reference_source": shot.reference_source,
                        "visual_style": {
                            "target_style": optimized_result.visual_style.target_style,
                            "core_theme": optimized_result.visual_style.core_theme,
                            "core_emotion": optimized_result.visual_style.core_emotion,
                            "color_palette": optimized_result.visual_style.color_palette,
                            "lighting_rules": optimized_result.visual_style.lighting_rules
                        }
                    }
                }

                # 计算结束时间
                shot_block["end_time"] = shot_block["start_time"] + shot_block["duration"]
                shot_blocks.append(shot_block)

            # ✅ 时长校验和调整：确保总时长符合 target_duration
            actual_duration = sum(shot["duration"] for shot in shot_blocks)
            logger.info(f"📊 [优化器] 生成的分镜总时长: {actual_duration:.1f} 秒（目标: {total_duration} 秒）")

            # ✅ 修复：根据万相固定5秒，重新调整分镜数量而不是缩减时长
            # 万相视频固定5秒，所以应该生成 target_duration/5 个分镜
            import math
            target_shot_count = math.ceil(total_duration / 5.0)
            current_shot_count = len(shot_blocks)

            logger.info(f"🎯 [优化器] 万相固定5秒/视频，目标分镜数: {target_shot_count}，当前分镜数: {current_shot_count}")

            if current_shot_count != target_shot_count:
                logger.info(f"⚠️  [优化器] 分镜数量不匹配，需要调整...")

                if current_shot_count > target_shot_count:
                    # 分镜过多，需要合并
                    logger.info(f"   🔗 合并分镜: {current_shot_count} → {target_shot_count}")
                    shot_blocks = self._merge_shots(shot_blocks, target_shot_count)
                else:
                    # 分镜过少，需要拆分（罕见情况）
                    logger.info(f"   ✂️ 拆分分镜: {current_shot_count} → {target_shot_count}")
                    shot_blocks = self._split_shots(shot_blocks, target_shot_count)

                # 调整每个分镜的时长为5秒，并重新计算时间戳
                current_time = 0.0
                for shot in shot_blocks:
                    shot["duration"] = 5.0
                    shot["start_time"] = round(current_time, 1)
                    shot["end_time"] = round(current_time + 5.0, 1)
                    current_time += 5.0

                logger.info(f"✅ [优化器] 调整后: {len(shot_blocks)} 个分镜，每个5秒，总时长 {current_time:.1f}秒")
            else:
                # 分镜数量正确，但可能时长不对，统一调整为5秒
                logger.info(f"✅ [优化器] 分镜数量正确，调整每个分镜为5秒")
                current_time = 0.0
                for shot in shot_blocks:
                    shot["duration"] = 5.0
                    shot["start_time"] = round(current_time, 1)
                    shot["end_time"] = round(current_time + 5.0, 1)
                    current_time += 5.0

            return {"shot_blocks_id": shot_blocks}

        except Exception as e:
            logger.error(f"❌ 优化器生成失败: {e}，降级为旧版生成")
            import traceback
            traceback.print_exc()
            return await self._generate_legacy(context, total_duration)

    async def _generate_legacy(self, context: Dict[str, Any], total_duration: int) -> Dict[str, Any]:
        """旧版分镜生成逻辑（原有代码）"""
        emotions: Dict[str, int] = context["emotions_id"].get("emotions", {})
        structure_template: Dict[str, Any] = context["structure_template_id"]
        user_video_type: str = context["video_type_id"]

        if not structure_template:
            raise ValueError("video_content 中缺少 structure_template")

        logger.info(f"🎯 [Node 3] 目标视频时长: {total_duration} 秒")

        # 检查缓存（包含目标时长）
        cached_result = self.cache.get(emotions, user_video_type, structure_template, total_duration)
        if cached_result:
            self.stats["cache_hits"] += 1
            logger.info(f"✅ 缓存命中（时长: {total_duration}s），跳过LLM调用")
            return {"shot_blocks_id": cached_result}

        # 生成分镜块
        shot_blocks = await self._generate_shot_blocks_with_retry(
            emotions, structure_template, user_video_type, total_duration
        )

        # 存储到缓存（包含目标时长）
        self.cache.set(emotions, user_video_type, structure_template, shot_blocks, total_duration)
        self.stats["llm_calls"] += 1

        return {"shot_blocks_id": shot_blocks}

    @async_retry_shot_gen(max_attempts=3, delay=1.0, backoff=2.0)
    async def _generate_shot_blocks_with_retry(self, emotions: Dict[str, int],
                                             structure_template: Dict[str, Any],
                                             user_video_type: str, total_duration: int) -> List[Dict]:
        """带重试机制的分镜块生成"""
        dominant_emotion = self._get_dominant_emotion(emotions)

        # 使用异步方式调用 Qwen 映射视频类型
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        qwen = QwenLLM()
        standard_video_type = await loop.run_in_executor(
            executor,
            lambda: self._map_to_supported_type(user_video_type, qwen)
        )
        logger.info(f"[类型映射] '{user_video_type}' → '{standard_video_type}'")

        # ✅ 根据目标时长计算需要的镜头数量
        shots_needed = self._calculate_shots_needed(total_duration)

        # ✅ 修复：按段落分配镜头数量
        num_segments = len(structure_template)

        # 如果目标镜头数少于段落数，只为前N个段落分配镜头
        if shots_needed < num_segments:
            shots_per_segment = 0
            remaining_shots = shots_needed
            logger.info(f"📊 [Node 3] 共 {num_segments} 个段落，但只需要 {shots_needed} 个镜头，将为前 {shots_needed} 个段落各分配1个镜头")
        else:
            shots_per_segment = shots_needed // num_segments
            remaining_shots = shots_needed % num_segments
            logger.info(f"📊 [Node 3] 共 {num_segments} 个段落，每个段落约 {shots_per_segment} 个镜头")

        shot_blocks = []
        current_time = 0.0
        shots_generated = 0

        for idx, (seg_name, content) in enumerate(structure_template.items()):
            seg_label = self._format_segment_name(seg_name)

            # 计算这个段落应该生成多少镜头
            segment_shots_count = shots_per_segment
            if idx < remaining_shots:  # 余数分配给前几个段落
                segment_shots_count += 1

            # 如果这个段落不需要镜头，跳过
            if segment_shots_count == 0:
                logger.info(f"⏭️  [Node 3] 跳过段落 {idx+1} ({seg_label})，不生成镜头")
                continue

            # 每个镜头约 5 秒
            segment_duration = segment_shots_count * 5.0

            # 生成单个段落的分镜
            segment_shots = await self._generate_segment_shots_async(
                seg_label, content, segment_duration, dominant_emotion, standard_video_type,
                target_shot_count=segment_shots_count  # 传入目标镜头数
            )

            # 设置时间戳
            for shot in segment_shots:
                shot["start_time"] = round(current_time, 1)
                end_time = current_time + shot["duration"]
                shot["end_time"] = round(end_time, 1)
                current_time = end_time
                shot_blocks.append(shot)
                shots_generated += 1

        # ✅ 时长校验和调整：确保总时长符合 target_duration
        actual_duration = sum(shot["duration"] for shot in shot_blocks)
        logger.info(f"📊 [Node 3] 生成的分镜总时长: {actual_duration:.1f} 秒（目标: {total_duration} 秒）")

        if actual_duration > total_duration:
            # 时长超出，需要按比例缩减
            scale_factor = total_duration / actual_duration
            logger.info(f"⚠️  [Node 3] 时长超出，将按比例缩减: {scale_factor:.2f}")

            current_time = 0.0
            for shot in shot_blocks:
                shot["duration"] = round(shot["duration"] * scale_factor, 1)
                shot["start_time"] = round(current_time, 1)
                shot["end_time"] = round(current_time + shot["duration"], 1)
                current_time += shot["duration"]

            logger.info(f"✅ [Node 3] 调整后总时长: {sum(shot['duration'] for shot in shot_blocks):.1f} 秒")
        elif actual_duration < total_duration * 0.8:
            # 时长过短（少于目标的80%），警告但不调整
            logger.info(f"⚠️  [Node 3] 生成的时长偏短（{actual_duration:.1f}s < {total_duration}s），建议检查生成逻辑")

        return shot_blocks

    async def _generate_segment_shots_async(self, seg_label: str, content: str,
                                          duration: float, emotion: str, video_type: str,
                                          target_shot_count: int = None) -> List[Dict]:
        """异步生成单个段落的分镜"""
        prompt = self._build_prompt(seg_label, content, duration, emotion, video_type, target_shot_count)

        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        try:
            qwen = QwenLLM()
            response = await loop.run_in_executor(
                executor,
                lambda: qwen.generate(prompt=prompt)
            )

            cleaned_response = response.strip().replace('\n', ' ').replace('\r', ' ')

            # 增强的JSON解析
            try:
                shots_json = self._extract_json_from_response_enhanced(cleaned_response)
                return self._parse_shots_from_json(shots_json, duration)
            except Exception as e:
                self.stats["json_parse_failures"] += 1
                logger.info(f"[JSON解析失败] {e}，尝试修复...")
                repaired_json = self._repair_json_heuristically(cleaned_response)
                return self._parse_shots_from_json(repaired_json, duration)

        except Exception as e:
            logger.info(f"[段落分镜生成失败] {e}")
            return self._fallback_shots(seg_label, duration, emotion)

    def _extract_json_from_response_enhanced(self, text: str) -> List[Dict]:
        """增强版JSON提取，更好的错误处理"""
        try:
            # 方法1: 直接解析
            return json5.loads(text.strip())
        except:
            pass

        try:
            # 方法2: 提取JSON数组
            json_match = re.search(r'(\[[\s\S]*?\])', text, re.DOTALL)
            if json_match:
                return json5.loads(json_match.group(1))
        except:
            pass

        try:
            # 方法3: 查找多个JSON对象并组合
            objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text)
            if objects:
                parsed_objects = []
                for obj in objects:
                    try:
                        parsed_objects.append(json5.loads(obj))
                    except:
                        continue
                if parsed_objects:
                    return parsed_objects
        except:
            pass

        raise ValueError(f"无法从响应中提取有效JSON: {text[:200]}...")

    def _get_dominant_emotion(self, emotions: Dict[str, int]) -> str:
        """获取主情感"""
        return max(emotions, key=emotions.get) if emotions else "冷静"

    def _allocate_time(self, template: Dict[str, str], total: int) -> Dict[str, float]:
        """为每个段落分配时间（简单平均或可扩展为加权）"""
        n = len(template)
        base_duration = total / n
        # 至少保证最小镜头时长
        return {self._format_segment_name(k): max(base_duration, 2.0) for k in template}

    def _calculate_shots_needed(self, total_duration: int) -> int:
        """
        根据目标时长计算需要的 shot blocks 数量
        每段视频固定 5 秒（万相API限制）
        """
        import math
        shots_needed = math.ceil(total_duration / 5.0)
        logger.info(f"🎯 [Node 3] 目标时长 {total_duration}秒，需要生成 {shots_needed} 个镜头（每个约5秒）")
        return shots_needed

    def _merge_shots(self, shot_blocks: List[Dict], target_count: int) -> List[Dict]:
        """
        合并分镜：将过多的分镜合并为目标数量

        例如：4个分镜 → 2个分镜，每2个合并为1个
        """
        if len(shot_blocks) <= target_count:
            return shot_blocks

        merged = []
        merge_ratio = len(shot_blocks) / target_count

        for i in range(target_count):
            # 计算这个合并分镜应该包含哪些原始分镜
            start_idx = int(i * merge_ratio)
            end_idx = int((i + 1) * merge_ratio)

            # 取这些分镜中的第一个作为基础
            base_shot = shot_blocks[start_idx].copy()

            # 合并描述（取第一个的描述，或者拼接）
            descriptions = [shot_blocks[j]["visual_description"] for j in range(start_idx, min(end_idx, len(shot_blocks)))]
            if len(descriptions) > 1:
                # 简单拼接前两个描述
                base_shot["visual_description"] = f"{descriptions[0][:50]}... + {descriptions[1][:30]}..."

            # ✅ 合并字幕 - 保留完整内容，让后续节点根据时长智能处理
            captions = [shot_blocks[j].get("caption", "") for j in range(start_idx, min(end_idx, len(shot_blocks)))]
            base_shot["caption"] = " ".join([c for c in captions if c])  # 去掉[:30]截断

            merged.append(base_shot)

        logger.info(f"   🔗 已合并: {len(shot_blocks)} 个分镜 → {len(merged)} 个分镜")
        return merged

    def _split_shots(self, shot_blocks: List[Dict], target_count: int) -> List[Dict]:
        """
        拆分分镜：将过少的分镜拆分为目标数量（罕见情况）

        例如：1个分镜 → 2个分镜，复制并调整描述
        """
        if len(shot_blocks) >= target_count:
            return shot_blocks

        result = []
        for shot in shot_blocks:
            result.append(shot.copy())
            # 如果还需要更多分镜，复制当前分镜
            while len(result) < target_count and len(result) < target_count:
                duplicated = shot.copy()
                duplicated["visual_description"] = f"{shot['visual_description'][:60]}... (续)"
                result.append(duplicated)
                if len(result) >= target_count:
                    break

        logger.info(f"   ✂️ 已拆分: {len(shot_blocks)} 个分镜 → {len(result)} 个分镜")
        return result[:target_count]  # 确保不超过目标数量

    def _format_segment_name(self, key: str) -> str:
        mapping = {"intro": "引入", "body": "主体", "conclusion": "结尾"}
        return mapping.get(key, key)



    def _repair_json_heuristically(self, text: str) -> dict:
        """
        启发式修复可能损坏的 JSON 字符串。
        """
        # 1. 提取最可能的 JSON 主体
        json_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if not json_match:
            raise ValueError("未检测到 JSON 结构")

        json_str = json_match.group(1)

        # 2. 清理常见干扰字符
        json_str = json_str.strip()
        json_str = json_str.replace('\\n', ' ').replace('\n', ' ').replace('\t', ' ')
        json_str = re.sub(r',\s*}', '}', json_str)  # 去除尾部多余逗号
        json_str = re.sub(r',\s*]', ']', json_str)

        # 3. 尝试解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError:
            pass

        # 4. 尝试修复括号缺失
        stack = []
        for char in json_str:
            if char in '{[':
                stack.append(char)
            elif char in '}]':
                if stack and ((char == '}' and stack[-1] == '{') or (char == ']' and stack[-1] == '[')):
                    stack.pop()
                else:
                    stack.append(char)

        # 从右往左补 }
        while stack:
            top = stack.pop()
            if top == '{':
                json_str += '}'
            elif top == '[':
                json_str += ']'
            elif top == '}':
                json_str = '{' + json_str
            elif top == ']':
                json_str = '[' + json_str

        # 5. 修复引号（简单处理奇数个双引号）
        if json_str.count('"') % 2 == 1:
            if not json_str.endswith('"'):
                json_str += '"'

        # 6. 最后一次尝试解析
        try:
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            raise ValueError(f"JSON 修复失败: {e}, 原始文本片段: {text[:200]}...") from e

    def _map_to_supported_type(self, user_type: str, qwen: QwenLLM) -> str:
        """
        使用 Qwen 将用户输入的 video_type 映射到支持的标准类型
        """
        if user_type in SUPPORTED_VIDEO_TYPES:
            return user_type

        prompt = f"""
    你是一个视频内容分类专家。请将用户描述的视频类型匹配到以下标准类别中最接近的一个。

    📌 支持的标准视频类型：
    {', '.join(SUPPORTED_VIDEO_TYPES)}

    ⚠️ 注意：
    - 只能选择上述列表中的一个，不能自创
    - 返回标准类型名称即可，不要解释

    🎯 用户输入的视频类型：{user_type}

    请输出最匹配的标准类型：
    """
        try:
            response = qwen.generate(prompt=prompt)
            predicted = response.strip().strip("“”\"'").strip()
            # 验证返回值是否在支持列表中
            if predicted in SUPPORTED_VIDEO_TYPES:
                return predicted
            else:
                # 再次尝试：让模型重新选择（防止输出带解释）
                prompt += "\n\n注意：只输出标准类型名称，不要任何其他文字！"
                response = qwen.generate(prompt=prompt)
                predicted = response.strip().strip("“”\"'").strip()
                return predicted if predicted in SUPPORTED_VIDEO_TYPES else "知识讲解"
        except Exception as e:
            logger.info(f"[警告] 类型映射失败，使用默认类型: {e}")
            return "知识讲解"

    def _build_prompt(self, segment_name: str, segment_content: str,
                      duration: float, emotion: str, video_type: str, target_shot_count: int = None) -> str:
        """构建智能适配视频类型的 prompt，集成专业分镜指导"""

        # 获取专业分镜指导提示词
        prompt_manager = get_prompt_manager()
        professional_guidance = prompt_manager.get_prompt_for_node(
            "shot_block_generation",
            context={
                "product": segment_content,
                "input": segment_content,
                "output": f"{video_type}类视频的{segment_name}段落"
            }
        )

        style = VIDEO_TYPE_STYLES.get(video_type, VIDEO_TYPE_STYLES["知识讲解"])  # 默认教育类

        # 如果没有指定镜头数量，使用默认逻辑
        if target_shot_count is None:
            target_shot_count = max(1, int(duration / 5))

        # 构建具体任务要求
        specific_task = f"""
请为【{video_type}】类视频的【{segment_name}】段落生成分镜镜头。

📌 视频类型特征：
- 风格基调：{style['tone']}
- 常用镜头：{style['shots']}
- 节奏特点：{style['pace']}
- 字幕风格：{style['caption']}

🎯 情感导向：{emotion}（请融入镜头设计中）

📄 段落内容：
{segment_content}

📝 输出要求：
1. ⚠️ 重要：必须生成 **恰好 {target_shot_count} 个镜头**，不多不少！
2. 每个镜头约 5 秒，总时长约 {int(duration)} 秒；
3. 每个镜头包含字段：
   - "shot_type": 镜头类型（如特写、全景、跟拍等）
   - "duration": 时长（约5秒，可微调在4-6秒之间）
   - "visual_description": 画面内容描述（具体、可视化）
   - "pacing": 节奏（从 ['快剪', '常规', '慢镜头', '定格'] 中选择）
   - "caption": 字幕文案（符合该视频类型的表达风格）
4. 输出为严格 JSON 列表，不加任何解释。

示例格式：
[
  {{
    "shot_type": "特写",
    "duration": 4,
    "visual_description": "手机屏幕亮起，显示'5折优惠'弹窗",
    "pacing": "快剪",
    "caption": "限时优惠，仅此一天！"
  }}
]

请开始输出：
"""

        # 使用PromptManager增强提示词
        enhanced_prompt = prompt_manager.enhance_prompt(
            specific_task,
            "shot_block_generation",
            context={
                "product": segment_content,
                "input": segment_content
            }
        )

        return enhanced_prompt

    def _extract_json_from_response(self, text: str) -> List[Dict]:
        """从 Qwen 响应中提取 JSON 列表"""
        try:
            # 尝试直接解析整个文本为 JSON
            return json5.loads(text.strip())
        except json.JSONDecodeError:
            # 查找第一个 [ 和最后一个 ] 之间的内容
            start = text.find("[")
            end = text.rfind("]") + 1
            if start != -1 and end != 0:
                json_str = text[start:end]
                return json5.loads(json_str)
            raise ValueError("无法从响应中提取有效 JSON")

    def _parse_shots_from_json(self, shots: List[Dict], max_duration: float) -> List[Dict]:
        """标准化并限制总时长"""
        total = 0.0
        result = []
        for item in shots:
            dur = float(item.get("duration", 3))
            if total + dur > max_duration and result:
                # 调整最后一项时长
                result[-1]["duration"] += (max_duration - total)
                result[-1]["duration"] = round(result[-1]["duration"], 1)
                break
            item["duration"] = round(dur, 1)
            item["visual_description"] = item.get("visual_description", "无描述")
            item["shot_type"] = item.get("shot_type", "中景")
            item["pacing"] = item.get("pacing", "常规")
            item["caption"] = item.get("caption", "")
            result.append(item)
            total += dur
        return result

    def _fallback_shots(self, seg_name: str, duration: float, emotion: str) -> List[Dict]:
        """模型失败时的简单回退策略"""
        return [{
            "segment": seg_name,
            "duration": round(duration, 1),
            "visual_description": f"[AI生成失败] {seg_name} 段落，情感：{emotion}",
            "shot_type": "中景",
            "pacing": "常规",
            "caption": f"正在加载内容...",
            "emotion_hint": emotion
        }]

    def _update_stats(self, response_time: float):
        """更新性能统计"""
        if self.stats["total_requests"] > 1:
            current_avg = self.stats["avg_response_time"]
            n = self.stats["total_requests"] - 1
            self.stats["avg_response_time"] = (current_avg * n + response_time) / self.stats["total_requests"]
        else:
            self.stats["avg_response_time"] = response_time

    def get_performance_stats(self) -> Dict[str, Any]:
        """获取性能统计信息"""
        total = self.stats["total_requests"]
        if total == 0:
            return {"message": "暂无统计数据"}

        cache_hit_rate = (self.stats["cache_hits"] / total) * 100 if total > 0 else 0
        json_failure_rate = (self.stats["json_parse_failures"] / self.stats["llm_calls"]) * 100 if self.stats["llm_calls"] > 0 else 0

        return {
            "total_requests": total,
            "cache_hits": self.stats["cache_hits"],
            "cache_hit_rate": f"{cache_hit_rate:.1f}%",
            "llm_calls": self.stats["llm_calls"],
            "fallback_calls": self.stats["fallback_calls"],
            "json_parse_failures": self.stats["json_parse_failures"],
            "json_failure_rate": f"{json_failure_rate:.1f}%",
            "avg_response_time": f"{self.stats['avg_response_time']:.3f}s"
        }

    def clear_cache(self):
        """清空缓存"""
        self.cache.cache.clear()
        self.cache.timestamps.clear()
        logger.info(f"✅ 分镜块生成缓存已清空")

    async def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        支持用户干预，如指定镜头类型或节奏
        示例 user_intent: {"shot_override": {"shot_type": "特写", "pacing": "快剪"}}
        """
        # 验证输入上下文
        self.validate_context(context)

        override = user_intent.get("shot_override")
        if not override or not isinstance(override, dict):
            return await self.generate(context)

        result = await self.generate(context)
        blocks = result["shot_blocks_id"]

        shot_type_override = override.get("shot_type")
        pacing_override = override.get("pacing")

        for block in blocks:
            if shot_type_override:
                block["shot_type"] = shot_type_override
            if pacing_override:
                block["pacing"] = pacing_override

        return result


# =============================
# ✅ Main 测试入口
# =============================

if __name__ == "__main__":
    # 模拟输入 context
    context = {
        "emotions": {
            "emotions": {"励志": 50, "冷静": 30, "激昂": 20}
        },
        "video_content": {
            "video_type": "教育类",
            "structure_template": {
                "intro": "你知道如何使用python快速构建你的第一个机器学习模型吗？",
                "body": "介绍python的优势，并演示导入库、加载数据、训练模型等关键步骤。",
                "conclusion": "总结内容，鼓励观众动手实践：'现在轮到你了！开启你的机器学习之旅吧。'"
            }
        },
        "target_duration": 60  # 可选注入
    }

    # 创建节点
    node = ShotBlockGenerationNode(node_id="shot_gen_1", name="AI分镜生成器")

    # 调用 generate
    result = node.generate(context)

    logger.info(f"🎬 分镜生成结果：\n")
    for i, block in enumerate(result["shot_blocks_id"]):
        logger.info(f"镜头 {i+1}:")
        logger.info(f"  时长: {block['duration']}s [{block['start_time']} → {block['end_time']}]")
        logger.info(f"  镜头: {block['shot_type']} | 节奏: {block['pacing']}")
        logger.info(f"  画面: {block['visual_description']}")
        logger.info(f"  字幕: {block['caption']}")
        logger.info(f"-" * 60)

    # 示例：用户干预
    logger.info(f"\n🔄 用户干预：全部改为‘特写’和‘快剪’\n")
    user_intent = {
        "shot_override": {
            "shot_type": "特写",
            "pacing": "快剪"
        }
    }
    regenerated = node.regenerate(context, user_intent)
    for i, block in enumerate(regenerated["shot_blocks_id"]):
        logger.info(f"镜头 {i+1}: {block['shot_type']} | {block['pacing']} | {block['visual_description']}")
        logger.info(f"-" * 60)