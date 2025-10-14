# matcher/bgm_matcher.py
from materials_supplies.models import BGMRequest, BGMResponse
import random
from typing import List

async def match_bgm(request: BGMRequest) -> List[BGMResponse]:
    # 模拟从 Java 获取 BGM 候选
    candidates = [
        {"url": "https://audio.com/tech-bgm.mp3", "duration": 120.0},
        {"url": "https://audio.com/epic-bgm.mp3", "duration": 90.0}
    ]

    results = []
    for c in candidates:
        duration = c["duration"]
        if duration >= request.duration:
            cut_start = random.uniform(0, duration - request.duration)
            cut_end = cut_start + request.duration
        else:
            cut_start = 0.0
            cut_end = duration

        results.append(BGMResponse(
            url=c["url"],
            cut_start=cut_start,
            cut_end=cut_end,
            duration=c["duration"]
        ))
    return results


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
