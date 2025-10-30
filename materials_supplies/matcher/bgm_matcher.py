# matcher/bgm_matcher.py
from materials_supplies.models import BGMRequest, BGMResponse
from materials_supplies.material_library_client import get_material_library_client
import random
import logging
from typing import List
import subprocess
import json

logger = logging.getLogger(__name__)


async def match_bgm(request: BGMRequest) -> List[BGMResponse]:
    """
    从素材库匹配BGM音频

    Args:
        request: BGM请求，包含:
            - description: 描述信息 (如"冷静情绪，使用合成贝斯...")
            - category: 音乐类型 (如"极简电子")
            - duration: 需要的时长(秒)

    Returns:
        BGM响应列表，包含匹配到的音频URL和裁剪信息
    """
    client = get_material_library_client()

    if not client:
        logger.warning("素材库客户端未初始化，BGM匹配失败")
        return []

    # 从description中提取mood（情绪）
    mood = ""
    if "情绪" in request.description:
        mood = request.description.split("情绪")[0].strip()

    # 构造搜索策略 (多级fallback)
    search_strategies = [
        {"tag": request.category, "keyword": ""},           # 优先: 精确风格 (如"极简电子")
        {"tag": mood, "keyword": ""},                       # 次选: 情绪匹配 (如"冷静")
        {"tag": request.category, "keyword": mood},         # 组合: 风格+情绪
        {"tag": "", "keyword": "背景音乐"},                  # 兜底: 任意BGM
    ]

    logger.info(f"🎵 开始匹配BGM: category={request.category}, mood={mood}, duration={request.duration}秒")

    # 逐个尝试搜索策略
    for idx, strategy in enumerate(search_strategies):
        tag = strategy["tag"]
        keyword = strategy["keyword"]

        logger.debug(f"   尝试策略 {idx+1}: tag='{tag}', keyword='{keyword}'")

        # 调用素材库API
        audio_list = client.search_audios(
            keyword=keyword if keyword else None,
            tag=tag if tag else None,
            page_size=5
        )

        if audio_list and len(audio_list) > 0:
            logger.info(f"   ✅ 策略 {idx+1} 找到 {len(audio_list)} 个候选音频")

            # 转换为BGMResponse
            responses = []
            for item in audio_list[:3]:  # 取前3个作为候选
                audio_url = item.get("url", "")

                if not audio_url:
                    logger.warning(f"   跳过无效音频: {item.get('name')}")
                    continue

                # 获取音频时长
                audio_duration = await _get_audio_duration(audio_url)

                if audio_duration is None or audio_duration <= 0:
                    logger.warning(f"   无法获取音频时长，跳过: {audio_url}")
                    continue

                # 随机裁剪片段
                if audio_duration >= request.duration:
                    cut_start = random.uniform(0, audio_duration - request.duration)
                    cut_end = cut_start + request.duration
                else:
                    # 音频时长不够，使用全部
                    cut_start = 0.0
                    cut_end = audio_duration
                    logger.warning(f"   音频时长不足: {audio_duration}秒 < {request.duration}秒")

                responses.append(BGMResponse(
                    url=audio_url,
                    cut_start=round(cut_start, 2),
                    cut_end=round(cut_end, 2),
                    duration=audio_duration
                ))

                logger.info(f"   添加候选: {item.get('name')} [{cut_start:.1f}s - {cut_end:.1f}s]")

            if responses:
                logger.info(f"🎉 BGM匹配成功，返回 {len(responses)} 个候选")
                return responses

    # 所有策略都失败
    logger.warning("⚠️ 所有BGM搜索策略都失败，返回空列表（视频将不包含BGM）")
    return []


async def _get_audio_duration(audio_url: str) -> float:
    """
    使用ffprobe获取音频文件的时长

    Args:
        audio_url: 音频文件URL

    Returns:
        音频时长(秒)，失败返回None
    """
    try:
        # 使用ffprobe获取音频信息
        cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'json',
            audio_url
        ]

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode == 0:
            data = json.loads(result.stdout)
            duration = float(data['format']['duration'])
            logger.debug(f"   音频时长: {duration:.2f}秒")
            return duration
        else:
            logger.warning(f"   ffprobe失败: {result.stderr}")
            return None

    except subprocess.TimeoutExpired:
        logger.warning(f"   ffprobe超时: {audio_url}")
        return None
    except Exception as e:
        logger.warning(f"   获取音频时长失败: {e}")
        # 如果ffprobe失败，返回默认时长
        return 120.0  # 默认假设2分钟


    def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        super().generate(context)

        shot_blocks: List[Dict] = context["shot_blocks"]
        target_duration: float = context["target_duration"]

        # --- 1. 自动分析分镜 → 情感、节奏、主题 ---
        analysis = self._analyze_shot_blocks(shot_blocks)
        primary_mood = analysis["primary_mood"]
        avg_bpm_hint = analysis["estimated_bpm"]
        dominant_theme = analysis["dominant_theme"]

        print(f"🔍 分析结果: 主情绪={primary_mood}, 建议BPM={avg_bpm_hint:.1f}, 主题={dominant_theme}")

        # --- 2. 优先使用用户上传音频 ---
        if self.uploaded_files:
            for file_info in self.uploaded_files:
                if file_info["type"] == "audio" and file_info["filename"].endswith((".mp3", ".wav", ".aiff")):
                    bgm_track = self._create_custom_track(file_info["path"], target_duration)
                    return {"bgm_track": bgm_track}

        # --- 3. 调用音乐 API 获取候选 ---
        candidate_tracks = self._fetch_matching_tracks(
            mood=primary_mood,
            bpm=avg_bpm_hint,
            duration=target_duration,
            theme=dominant_theme
        )

        if not candidate_tracks:
            print("⚠️ 未找到匹配音乐，尝试通用‘励志’类音乐...")
            candidate_tracks = self._fetch_matching_tracks(mood="励志", bpm=100, duration=target_duration)

        if not candidate_tracks:
            print("⚠️ 仍无结果，使用默认静音或提示音")
            return {
                "bgm_track": {
                    "source": "fallback",
                    "title": "No BGM Available",
                    "duration": target_duration,
                    "file_path": None,
                    "volume_db": -20.0
                }
            }

        # --- 4. 选评分最高的 ---
        best_track = max(candidate_tracks, key=lambda t: t.get("match_score", 0.5))
        bgm_track = self._create_library_track(best_track, target_duration)

        return {"bgm_track": bgm_track}

    def _analyze_shot_blocks(self, shots: List[Dict]) -> Dict[str, Any]:
        """从分镜中推断情绪、BPM、主题"""
        total_duration = 0.0
        mood_counter = {}
        pacing_weights = {"快": 1.8, "常规": 1.0, "慢镜头": 0.6}
        theme_counter = {}

        for shot in shots:
            duration = shot.get("duration", 5.0)
            total_duration += duration

            # 1. 情绪推断（基于 shot_type + visual_description + caption 关键词）
            mood = self._infer_mood_from_text(shot)
            mood_counter[mood] = mood_counter.get(mood, 0) + duration * MOOD_WEIGHTS.get(mood, 0.5)

            # 2. 主题推断（教育、科技、生活、励志等）
            theme = self._infer_theme_from_text(shot)
            theme_counter[theme] = theme_counter.get(theme, 0) + duration

            # 3. 节奏权重（用于BPM估算）
            pacing = shot.get("pacing", "常规")
            speed_factor = pacing_weights.get(pacing, 1.0)

        # 主情绪
        primary_mood = max(mood_counter, key=mood_counter.get) if mood_counter else "冷静"

        # 主题
        dominant_theme = max(theme_counter, key=theme_counter.get) if theme_counter else "通用"

        # BPM 估算：基于剪辑密度 × 节奏因子
        cuts_per_minute = (len(shots) / (total_duration or 1)) * 60
        estimated_bpm = cuts_per_minute * 4  # 每小节4拍
        estimated_bpm *= pacing_weights.get("常规", 1.0)  # 可加入 pacing 调整

        return {
            "primary_mood": primary_mood,
            "estimated_bpm": round(estimated_bpm, 1),
            "dominant_theme": dominant_theme,
            "mood_dist": {k: v / sum(mood_counter.values()) for k, v in mood_counter.items()}
        }

    def _infer_mood_from_text(self, shot: Dict) -> str:
        """基于文本关键词推断情绪"""
        text = f"{shot.get('visual_description', '')} {shot.get('caption', '')} {shot.get('shot_type', '')}"
        text_lower = text.lower()

        mood_keywords = {
            "激昂": ["激情", "激动", "高潮", "突破", "挑战"],
            "励志": ["加油", "你可以", "坚持", "梦想", "努力", "旅程", "开启"],
            "感动": ["感动", "回忆", "温暖", "陪伴", "成长"],
            "温馨": ["温馨", "家庭", "微笑", "明亮", "整洁"],
            "幽默": ["搞笑", "滑稽", "调皮", "笑"],
            "冷静": ["分析", "数据", "逻辑", "思考", "白板"],
            "悬疑": ["秘密", "未知", "探索", "黑影"],
            "科技": ["AI", "机器学习", "代码", "电脑", "算法", "项目", "技术"]
        }

        for mood, keywords in mood_keywords.items():
            if any(k in text_lower for k in keywords):
                return mood
        return "冷静"  # 默认

    def _infer_theme_from_text(self, shot: Dict) -> str:
        """推断主题（可用于API过滤）"""
        text = f"{shot.get('visual_description', '')} {shot.get('caption', '')}".lower()
        if "学习" in text or "课程" in text or "学生" in text or "教育" in text:
            return "教育"
        elif "科技" in text or "AI" in text or "机器学习" in text or "编程" in text:
            return "科技"
        elif "家庭" in text or "家" in text or "生活" in text:
            return "生活"
        elif "运动" in text or "比赛" in text:
            return "运动"
        return "通用"

    def _fetch_matching_tracks(self, mood: str, bpm: float, duration: float, theme: str = "通用") -> List[Dict]:
        """调用外部音乐API搜索匹配曲目"""
        try:
            response = requests.post(MUSIC_SEARCH_API, json={
                "mood": mood,
                "bpm": bpm,
                "bpm_tolerance": BPM_TOLERANCE,
                "duration": duration,
                "duration_tolerance": 10.0,
                "genre_hint": theme,
                "limit": 10
            }, timeout=5)

            if response.status_code == 200:
                tracks_data = response.json().get("tracks", [])
                candidates = []
                for item in tracks_data:
                    # 计算匹配度评分
                    bpm_diff = abs(item["bpm"] - bpm)
                    bpm_match = 1.0 if bpm_diff <= BPM_TOLERANCE else max(0, 1 - bpm_diff / 20)
                    mood_match = 1.0 if mood in item.get("mood", []) else 0.4

                    stretch_ratio = duration / item["duration"]
                    if stretch_ratio < 0.8 or stretch_ratio > 1.3:
                        continue

                    score = (mood_match * 0.6 + bpm_match * 0.4)

                    if score >= self.system_parameters["min_match_score"]:
                        item["match_score"] = score
                        item["stretch_ratio"] = stretch_ratio
                        candidates.append(item)
                return candidates
            else:
                print(f"❌ API请求失败: {response.status_code} - {response.text}")
                return []

        except Exception as e:
            print(f"⚠️  请求音乐API出错: {e}")
            return []

    def _create_library_track(self, track: Dict, target_duration: float) -> Dict:
        """创建库内音乐的轨道配置"""
        stretch_ratio = target_duration / track["duration"]

        return {
            "source": "library_api",
            "track_id": track["id"],
            "title": track["title"],
            "file_path": track["file_path"],
            "original_bpm": track["bpm"],
            "applied_bpm": track["bpm"],
            "duration": target_duration,
            "stretch_ratio": round(stretch_ratio, 3),
            "fade_in": DEFAULT_FADE_IN,
            "fade_out": max(DEFAULT_FADE_OUT, target_duration * 0.05),
            "volume_db": self.system_parameters["default_volume"],
            "processing": {
                "pitch_preserved": True,
                "time_stretch": True
            },
            "metadata": {
                "genre": track.get("genre", []),
                "mood": track.get("mood", []),
                "key": track.get("key", "N/A"),
                "has_bass_drop": track.get("has_bass_drop", False)
            }
        }

    def _create_custom_track(self, file_path: str, target_duration: float) -> Dict:
        """创建自定义音频轨道"""
        return {
            "source": "custom",
            "title": "用户上传BGM",
            "file_path": file_path,
            "duration": target_duration,
            "stretch_ratio": 1.0,
            "fade_in": DEFAULT_FADE_IN,
            "fade_out": max(DEFAULT_FADE_OUT, target_duration * 0.05),
            "volume_db": self.system_parameters["default_volume"],
            "processing": {
                "pitch_preserved": True,
                "time_stretch": True
            },
            "metadata": {},
            "anchor_processing": []
        }
