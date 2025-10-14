"""
数字人统一管理器 - 确保整个视频使用同一数字人源
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import httpx
from concurrent.futures import ThreadPoolExecutor

from llm.qwen import QwenLLM


class Gender(Enum):
    """性别"""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


class AgeGroup(Enum):
    """年龄组"""
    YOUNG = "young"      # 18-30
    MIDDLE = "middle"    # 30-50
    SENIOR = "senior"    # 50+


class Style(Enum):
    """风格"""
    PROFESSIONAL = "professional"  # 专业
    CASUAL = "casual"              # 休闲
    FRIENDLY = "friendly"          # 亲和
    AUTHORITATIVE = "authoritative" # 权威
    CREATIVE = "creative"          # 创意


@dataclass
class DigitalHuman:
    """数字人配置"""
    avatar_id: str
    name: str
    gender: Gender
    age_group: AgeGroup
    style: Style
    voice_id: str
    language: str = "zh-CN"
    accent: str = "standard"
    speaking_rate: float = 1.0
    pitch: float = 0.0
    volume: float = 1.0
    avatar_url: Optional[str] = None
    thumbnail_url: Optional[str] = None
    description: str = ""
    supported_emotions: List[str] = None
    supported_backgrounds: List[str] = None


@dataclass
class SelectionCriteria:
    """选择标准"""
    preferred_gender: Optional[Gender] = None
    preferred_age: Optional[AgeGroup] = None
    preferred_style: Optional[Style] = None
    language_requirement: str = "zh-CN"
    brand_guidelines: Optional[Dict] = None
    audience_profile: Optional[Dict] = None
    content_type: str = "general"  # "educational", "commercial", "entertainment"


class DigitalHumanManager:
    """数字人统一管理器"""

    def __init__(self):
        self.qwen = QwenLLM()
        self.selected_avatar: Optional[DigitalHuman] = None
        self.avatar_selection_locked = False

        # API配置
        self.avatar_api_url = "https://api.digital-human.com/v1"
        self.tts_api_url = "https://api.tts-service.com/v1"

        # 可用数字人库
        self.available_avatars = self._initialize_avatar_library()

    def _initialize_avatar_library(self) -> List[DigitalHuman]:
        """初始化数字人库"""
        return [
            # 专业女性
            DigitalHuman(
                avatar_id="prof_female_01",
                name="李雅文",
                gender=Gender.FEMALE,
                age_group=AgeGroup.MIDDLE,
                style=Style.PROFESSIONAL,
                voice_id="zh-CN-XiaoxiaoNeural",
                description="专业知性的商务女性，适合教育和商业内容",
                supported_emotions=["neutral", "happy", "confident"],
                supported_backgrounds=["office", "studio", "classroom"]
            ),

            # 专业男性
            DigitalHuman(
                avatar_id="prof_male_01",
                name="张文博",
                gender=Gender.MALE,
                age_group=AgeGroup.MIDDLE,
                style=Style.PROFESSIONAL,
                voice_id="zh-CN-YunxiNeural",
                description="成熟稳重的商务男性，适合权威性内容",
                supported_emotions=["neutral", "serious", "confident"],
                supported_backgrounds=["office", "studio", "conference"]
            ),

            # 亲和女性
            DigitalHuman(
                avatar_id="friendly_female_01",
                name="小雨",
                gender=Gender.FEMALE,
                age_group=AgeGroup.YOUNG,
                style=Style.FRIENDLY,
                voice_id="zh-CN-XiaoyiNeural",
                description="活泼亲和的年轻女性，适合生活化内容",
                supported_emotions=["happy", "excited", "warm"],
                supported_backgrounds=["home", "outdoor", "casual"]
            ),

            # 权威男性
            DigitalHuman(
                avatar_id="auth_male_01",
                name="王教授",
                gender=Gender.MALE,
                age_group=AgeGroup.SENIOR,
                style=Style.AUTHORITATIVE,
                voice_id="zh-CN-YunyangNeural",
                description="权威专业的资深专家，适合学术和专业内容",
                supported_emotions=["serious", "neutral", "wise"],
                supported_backgrounds=["library", "classroom", "studio"]
            ),

            # 创意女性
            DigitalHuman(
                avatar_id="creative_female_01",
                name="艾米",
                gender=Gender.FEMALE,
                age_group=AgeGroup.YOUNG,
                style=Style.CREATIVE,
                voice_id="zh-CN-XiaohanNeural",
                description="富有创意的艺术女性，适合创意和娱乐内容",
                supported_emotions=["excited", "creative", "playful"],
                supported_backgrounds=["studio", "creative_space", "colorful"]
            ),

            # 休闲男性
            DigitalHuman(
                avatar_id="casual_male_01",
                name="小明",
                gender=Gender.MALE,
                age_group=AgeGroup.YOUNG,
                style=Style.CASUAL,
                voice_id="zh-CN-YunyeNeural",
                description="轻松随意的年轻男性，适合生活化和娱乐内容",
                supported_emotions=["relaxed", "happy", "casual"],
                supported_backgrounds=["home", "outdoor", "casual"]
            )
        ]

    async def select_optimal_avatar(
        self,
        scripts: List[str],
        criteria: SelectionCriteria,
        context: Dict[str, Any] = None
    ) -> DigitalHuman:
        """
        选择最优数字人
        """
        if self.avatar_selection_locked and self.selected_avatar:
            return self.selected_avatar

        # 如果已有全局设置，直接返回
        if self.selected_avatar:
            return self.selected_avatar

        # Step 1: 分析内容特征
        content_analysis = await self._analyze_content_characteristics(scripts)

        # Step 2: 根据标准筛选候选
        candidates = self._filter_candidates_by_criteria(criteria)

        # Step 3: 基于内容匹配评分
        scored_candidates = await self._score_candidates_for_content(
            candidates, content_analysis, criteria
        )

        # Step 4: 选择最佳候选
        best_avatar = self._select_best_candidate(scored_candidates)

        # Step 5: 锁定选择
        self.selected_avatar = best_avatar
        self.avatar_selection_locked = True

        return best_avatar

    async def _analyze_content_characteristics(self, scripts: List[str]) -> Dict[str, Any]:
        """分析内容特征"""
        combined_script = "\n".join(scripts[:5])  # 分析前5个脚本

        analysis_prompt = f"""
        请分析以下视频脚本的特征，判断最适合的数字人类型：

        【脚本内容】:
        {combined_script}

        请分析：
        1. 内容类型 (educational/commercial/entertainment/news)
        2. 语调风格 (formal/casual/friendly/authoritative)
        3. 目标受众 (young/middle/senior/general)
        4. 情感倾向 (serious/happy/neutral/excited)
        5. 专业程度 (high/medium/low)

        输出JSON格式：
        {{
            "content_type": "educational",
            "tone_style": "formal",
            "target_audience": "general",
            "emotional_tendency": "neutral",
            "professionalism_level": "high",
            "recommended_gender": "female",
            "recommended_age": "middle",
            "recommended_style": "professional"
        }}
        """

        try:
            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=1)

            response = await loop.run_in_executor(
                executor,
                lambda: self.qwen.generate(prompt=analysis_prompt, max_retries=3)
            )

            if response:
                import json
                return json.loads(response)

        except Exception as e:
            print(f"[内容分析] 失败: {e}")

        # 降级返回默认分析
        return {
            "content_type": "general",
            "tone_style": "neutral",
            "target_audience": "general",
            "emotional_tendency": "neutral",
            "professionalism_level": "medium",
            "recommended_gender": "female",
            "recommended_age": "middle",
            "recommended_style": "professional"
        }

    def _filter_candidates_by_criteria(self, criteria: SelectionCriteria) -> List[DigitalHuman]:
        """根据标准筛选候选数字人"""
        candidates = []

        for avatar in self.available_avatars:
            # 性别筛选
            if criteria.preferred_gender and avatar.gender != criteria.preferred_gender:
                continue

            # 年龄筛选
            if criteria.preferred_age and avatar.age_group != criteria.preferred_age:
                continue

            # 风格筛选
            if criteria.preferred_style and avatar.style != criteria.preferred_style:
                continue

            # 语言筛选
            if avatar.language != criteria.language_requirement:
                continue

            candidates.append(avatar)

        # 如果筛选后没有候选，放宽条件
        if not candidates:
            candidates = [avatar for avatar in self.available_avatars
                         if avatar.language == criteria.language_requirement]

        return candidates

    async def _score_candidates_for_content(
        self,
        candidates: List[DigitalHuman],
        content_analysis: Dict[str, Any],
        criteria: SelectionCriteria
    ) -> List[Tuple[DigitalHuman, float]]:
        """为候选数字人打分"""
        scored_candidates = []

        for candidate in candidates:
            score = self._calculate_match_score(candidate, content_analysis, criteria)
            scored_candidates.append((candidate, score))

        # 按分数排序
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        return scored_candidates

    def _calculate_match_score(
        self,
        candidate: DigitalHuman,
        content_analysis: Dict[str, Any],
        criteria: SelectionCriteria
    ) -> float:
        """计算匹配分数"""
        score = 0.0

        # 性别匹配 (20%)
        recommended_gender = content_analysis.get("recommended_gender")
        if recommended_gender and candidate.gender.value == recommended_gender:
            score += 0.2

        # 年龄匹配 (20%)
        recommended_age = content_analysis.get("recommended_age")
        if recommended_age and candidate.age_group.value == recommended_age:
            score += 0.2

        # 风格匹配 (25%)
        recommended_style = content_analysis.get("recommended_style")
        if recommended_style and candidate.style.value == recommended_style:
            score += 0.25

        # 内容类型匹配 (15%)
        content_type = content_analysis.get("content_type", "general")
        if self._is_suitable_for_content_type(candidate, content_type):
            score += 0.15

        # 专业程度匹配 (10%)
        professionalism = content_analysis.get("professionalism_level", "medium")
        if self._matches_professionalism_level(candidate, professionalism):
            score += 0.1

        # 情感表达能力 (10%)
        emotional_tendency = content_analysis.get("emotional_tendency", "neutral")
        if self._supports_emotion(candidate, emotional_tendency):
            score += 0.1

        return score

    def _is_suitable_for_content_type(self, candidate: DigitalHuman, content_type: str) -> bool:
        """检查是否适合内容类型"""
        suitability_map = {
            "educational": [Style.PROFESSIONAL, Style.AUTHORITATIVE],
            "commercial": [Style.PROFESSIONAL, Style.FRIENDLY],
            "entertainment": [Style.FRIENDLY, Style.CREATIVE, Style.CASUAL],
            "news": [Style.PROFESSIONAL, Style.AUTHORITATIVE]
        }

        suitable_styles = suitability_map.get(content_type, [])
        return candidate.style in suitable_styles

    def _matches_professionalism_level(self, candidate: DigitalHuman, level: str) -> bool:
        """检查专业程度匹配"""
        professionalism_map = {
            "high": [Style.PROFESSIONAL, Style.AUTHORITATIVE],
            "medium": [Style.PROFESSIONAL, Style.FRIENDLY],
            "low": [Style.FRIENDLY, Style.CASUAL, Style.CREATIVE]
        }

        suitable_styles = professionalism_map.get(level, [])
        return candidate.style in suitable_styles

    def _supports_emotion(self, candidate: DigitalHuman, emotion: str) -> bool:
        """检查是否支持特定情感"""
        if not candidate.supported_emotions:
            return True  # 假设支持所有情感

        emotion_mapping = {
            "serious": "serious",
            "happy": "happy",
            "neutral": "neutral",
            "excited": "excited"
        }

        mapped_emotion = emotion_mapping.get(emotion, "neutral")
        return mapped_emotion in candidate.supported_emotions

    def _select_best_candidate(self, scored_candidates: List[Tuple[DigitalHuman, float]]) -> DigitalHuman:
        """选择最佳候选"""
        if not scored_candidates:
            # 返回默认数字人
            return self.available_avatars[0]

        # 返回得分最高的
        return scored_candidates[0][0]

    async def generate_unified_talking_video(
        self,
        scripts: List[str],
        durations: List[float],
        background: str = "studio",
        emotion: str = "neutral"
    ) -> Dict[str, Any]:
        """
        生成统一的数字人视频
        """
        if not self.selected_avatar:
            raise ValueError("尚未选择数字人，请先调用 select_optimal_avatar")

        # Step 1: 合并脚本
        combined_script = self._merge_scripts(scripts, durations)

        # Step 2: 生成TTS音频
        audio_result = await self._generate_tts_audio(combined_script)

        # Step 3: 生成数字人视频
        video_result = await self._generate_avatar_video(
            audio_result["audio_url"],
            background,
            emotion,
            sum(durations)
        )

        # Step 4: 创建分段信息
        segments = self._create_segment_markers(scripts, durations)

        return {
            "video_url": video_result["video_url"],
            "audio_url": audio_result["audio_url"],
            "total_duration": sum(durations),
            "avatar_used": self.selected_avatar.name,
            "avatar_id": self.selected_avatar.avatar_id,
            "voice_id": self.selected_avatar.voice_id,
            "script": combined_script,
            "segments": segments,
            "background": background,
            "emotion": emotion
        }

    def _merge_scripts(self, scripts: List[str], durations: List[float]) -> str:
        """合并脚本，添加适当的停顿"""
        merged_parts = []

        for i, script in enumerate(scripts):
            cleaned_script = script.strip()
            if cleaned_script:
                merged_parts.append(cleaned_script)

                # 在片段间添加停顿标记（除了最后一个）
                if i < len(scripts) - 1:
                    # 根据下一段的开始添加停顿
                    pause_duration = min(durations[i] * 0.1, 1.0)  # 最多1秒停顿
                    if pause_duration > 0.3:
                        merged_parts.append(f"<break time='{pause_duration:.1f}s'/>")

        return "".join(merged_parts)

    async def _generate_tts_audio(self, script: str) -> Dict[str, Any]:
        """生成TTS音频"""
        tts_payload = {
            "text": script,
            "voice_id": self.selected_avatar.voice_id,
            "language": self.selected_avatar.language,
            "speaking_rate": self.selected_avatar.speaking_rate,
            "pitch": self.selected_avatar.pitch,
            "volume": self.selected_avatar.volume,
            "output_format": "mp3"
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.tts_api_url}/synthesize",
                    json=tts_payload,
                    timeout=60.0
                )

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "audio_url": result["audio_url"],
                        "duration": result.get("duration", 0),
                        "voice_used": self.selected_avatar.voice_id
                    }

        except Exception as e:
            print(f"[TTS生成] 失败: {e}")

        # 模拟返回结果
        return {
            "audio_url": f"https://tts-service.com/audio/{hash(script)}.mp3",
            "duration": len(script) * 0.1,  # 估算时长
            "voice_used": self.selected_avatar.voice_id
        }

    async def _generate_avatar_video(
        self,
        audio_url: str,
        background: str,
        emotion: str,
        duration: float
    ) -> Dict[str, Any]:
        """生成数字人视频"""
        avatar_payload = {
            "avatar_id": self.selected_avatar.avatar_id,
            "audio_url": audio_url,
            "background": background,
            "emotion": emotion,
            "duration": duration,
            "resolution": "1920x1080",
            "fps": 30
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.avatar_api_url}/generate",
                    json=avatar_payload,
                    timeout=300.0  # 5分钟超时
                )

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "video_url": result["video_url"],
                        "thumbnail_url": result.get("thumbnail_url"),
                        "processing_time": result.get("processing_time", 0)
                    }

        except Exception as e:
            print(f"[数字人视频生成] 失败: {e}")

        # 模拟返回结果
        return {
            "video_url": f"https://avatar-service.com/video/{hash(audio_url)}.mp4",
            "thumbnail_url": f"https://avatar-service.com/thumb/{hash(audio_url)}.jpg",
            "processing_time": duration * 2  # 估算处理时间
        }

    def _create_segment_markers(self, scripts: List[str], durations: List[float]) -> List[Dict[str, Any]]:
        """创建分段标记"""
        segments = []
        current_time = 0

        for i, (script, duration) in enumerate(zip(scripts, durations)):
            segment = {
                "segment_id": f"talking_seg_{i}",
                "start_time": current_time,
                "end_time": current_time + duration,
                "duration": duration,
                "script": script,
                "avatar_id": self.selected_avatar.avatar_id
            }
            segments.append(segment)
            current_time += duration

        return segments

    def get_avatar_info(self) -> Optional[Dict[str, Any]]:
        """获取当前选择的数字人信息"""
        if not self.selected_avatar:
            return None

        return {
            "avatar_id": self.selected_avatar.avatar_id,
            "name": self.selected_avatar.name,
            "gender": self.selected_avatar.gender.value,
            "age_group": self.selected_avatar.age_group.value,
            "style": self.selected_avatar.style.value,
            "voice_id": self.selected_avatar.voice_id,
            "language": self.selected_avatar.language,
            "description": self.selected_avatar.description,
            "supported_emotions": self.selected_avatar.supported_emotions,
            "supported_backgrounds": self.selected_avatar.supported_backgrounds
        }

    def reset_selection(self):
        """重置数字人选择"""
        self.selected_avatar = None
        self.avatar_selection_locked = False

    def lock_selection(self):
        """锁定当前选择"""
        self.avatar_selection_locked = True

    def unlock_selection(self):
        """解锁选择"""
        self.avatar_selection_locked = False

    def get_available_avatars(self, criteria: Optional[SelectionCriteria] = None) -> List[Dict[str, Any]]:
        """获取可用数字人列表"""
        avatars = self.available_avatars

        if criteria:
            avatars = self._filter_candidates_by_criteria(criteria)

        return [
            {
                "avatar_id": avatar.avatar_id,
                "name": avatar.name,
                "gender": avatar.gender.value,
                "age_group": avatar.age_group.value,
                "style": avatar.style.value,
                "description": avatar.description,
                "thumbnail_url": avatar.thumbnail_url
            }
            for avatar in avatars
        ]

    async def preview_avatar_voice(self, avatar_id: str, sample_text: str = "这是一段测试语音") -> Dict[str, Any]:
        """预览数字人声音"""
        avatar = next((a for a in self.available_avatars if a.avatar_id == avatar_id), None)
        if not avatar:
            return {"error": "数字人不存在"}

        # 生成预览音频
        preview_payload = {
            "text": sample_text,
            "voice_id": avatar.voice_id,
            "language": avatar.language,
            "duration_limit": 10  # 限制10秒
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    f"{self.tts_api_url}/preview",
                    json=preview_payload,
                    timeout=30.0
                )

                if response.status_code == 200:
                    result = response.json()
                    return {
                        "preview_url": result["audio_url"],
                        "avatar_name": avatar.name,
                        "voice_id": avatar.voice_id
                    }

        except Exception as e:
            print(f"[语音预览] 失败: {e}")

        # 模拟返回
        return {
            "preview_url": f"https://tts-service.com/preview/{avatar_id}.mp3",
            "avatar_name": avatar.name,
            "voice_id": avatar.voice_id
        }