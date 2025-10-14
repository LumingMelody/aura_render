"""
增强的情感分析节点 - 添加新的方法和功能
"""
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import numpy as np
from collections import defaultdict

from .emotion_analysis_node import EmotionAnalysisNode, EMOTION_CATEGORIES, EMOTION_DIMENSIONS, EMOTION_COMPATIBILITY
from .emotion_analysis_node import EmotionCurve, EmotionAnalysisResult


class EmotionAnalysisEnhanced(EmotionAnalysisNode):
    """增强的情感分析节点"""

    def _keyword_emotion_analysis(self, text: str) -> Dict[str, float]:
        """基于关键词的情感分析"""
        lower_text = text.lower()
        scores = {emotion: 0.1 for emotion in EMOTION_CATEGORIES}

        # 扩展的关键词规则
        keyword_rules = {
            "激昂": {
                "keywords": ["激动", "热血", "燃", "爆发", "澎湃", "激情", "狂欢", "震撼", "强烈", "火热"],
                "boost": 0.6,
                "context_boost": {"动作": 0.2, "运动": 0.3, "竞赛": 0.4}
            },
            "温馨": {
                "keywords": ["温暖", "家", "爱", "陪伴", "关怀", "呵护", "亲情", "温柔", "舒适", "安逸"],
                "boost": 0.5,
                "context_boost": {"家庭": 0.3, "生活": 0.2, "日常": 0.2}
            },
            "悬疑": {
                "keywords": ["神秘", "未知", "背后", "真相", "秘密", "隐藏", "谜团", "探索", "发现", "揭示"],
                "boost": 0.7,
                "context_boost": {"侦探": 0.4, "推理": 0.3, "调查": 0.3}
            },
            "幽默": {
                "keywords": ["搞笑", "笑死", "段子", "有趣", "好玩", "逗", "调侃", "诙谐", "滑稽", "欢乐"],
                "boost": 0.5,
                "context_boost": {"喜剧": 0.4, "娱乐": 0.2, "轻松": 0.2}
            },
            "悲伤": {
                "keywords": ["悲伤", "难过", "失落", "痛苦", "伤心", "哭泣", "眼泪", "忧郁", "沮丧", "绝望"],
                "boost": 0.6,
                "context_boost": {"分离": 0.3, "失去": 0.4, "离别": 0.3}
            },
            "励志": {
                "keywords": ["励志", "奋斗", "梦想", "坚持", "努力", "拼搏", "成功", "突破", "挑战", "成长"],
                "boost": 0.5,
                "context_boost": {"创业": 0.3, "学习": 0.2, "进步": 0.3}
            },
            "冷静": {
                "keywords": ["冷静", "理性", "分析", "客观", "专业", "严肃", "正式", "商务", "学术", "科学"],
                "boost": 0.5,
                "context_boost": {"教育": 0.3, "商业": 0.2, "技术": 0.3}
            },
            "浪漫": {
                "keywords": ["浪漫", "爱情", "甜蜜", "约会", "情侣", "美好", "梦幻", "诗意", "唯美", "温馨"],
                "boost": 0.5,
                "context_boost": {"婚礼": 0.4, "节日": 0.2, "庆祝": 0.2}
            },
            "恐惧": {
                "keywords": ["恐惧", "害怕", "惊悚", "可怕", "恐怖", "威胁", "危险", "紧张", "担心", "焦虑"],
                "boost": 0.7,
                "context_boost": {"危机": 0.4, "灾难": 0.3, "紧急": 0.3}
            },
            "感动": {
                "keywords": ["感动", "泪目", "触动", "震撼", "感人", "温情", "真挚", "深刻", "打动", "感慨"],
                "boost": 0.6,
                "context_boost": {"故事": 0.2, "经历": 0.3, "回忆": 0.2}
            }
        }

        for emotion, rule in keyword_rules.items():
            base_score = 0

            # 基础关键词匹配
            for keyword in rule["keywords"]:
                if keyword in lower_text:
                    base_score += rule["boost"]

            # 上下文增强
            for context, boost in rule["context_boost"].items():
                if context in lower_text:
                    base_score += boost

            scores[emotion] += base_score

        # 添加随机扰动
        for emotion in scores:
            scores[emotion] += np.random.uniform(-0.05, 0.1)
            scores[emotion] = max(0.0, scores[emotion])

        return scores

    def _syntax_emotion_analysis(self, text: str) -> Dict[str, float]:
        """基于句法结构的情感分析"""
        scores = {emotion: 0.1 for emotion in EMOTION_CATEGORIES}

        # 句法特征分析
        sentences = text.split('。')

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            # 感叹号和问号分析
            exclamation_count = sentence.count('!')
            question_count = sentence.count('?')

            if exclamation_count > 0:
                scores["激昂"] += 0.2 * exclamation_count
                scores["幽默"] += 0.1 * exclamation_count
                scores["感动"] += 0.15 * exclamation_count

            if question_count > 0:
                scores["悬疑"] += 0.25 * question_count
                scores["冷静"] += 0.1 * question_count

            # 句子长度分析
            sentence_length = len(sentence)
            if sentence_length > 50:  # 长句倾向于冷静、专业
                scores["冷静"] += 0.1
            elif sentence_length < 15:  # 短句倾向于激昂、紧张
                scores["激昂"] += 0.1
                scores["悬疑"] += 0.05

            # 语气词分析
            tone_words = {
                "啊": ("感动", 0.2),
                "呀": ("幽默", 0.15),
                "哦": ("温馨", 0.1),
                "嗯": ("冷静", 0.1),
                "哇": ("激昂", 0.25),
                "呜": ("悲伤", 0.3)
            }

            for tone, (emotion, boost) in tone_words.items():
                if tone in sentence:
                    scores[emotion] += boost

        return scores

    def _fuse_emotion_results(self, results: List[Tuple[Dict[str, float], float]]) -> Dict[str, float]:
        """融合多个情感分析结果"""
        fused_scores = defaultdict(float)
        total_weight = sum(weight for _, weight in results)

        for emotion_dict, weight in results:
            for emotion, score in emotion_dict.items():
                fused_scores[emotion] += score * (weight / total_weight)

        return dict(fused_scores)

    def _calculate_dimensional_scores(self, emotions: Dict[str, float]) -> Dict[str, float]:
        """计算情感维度分数"""
        dimensional_scores = {}

        for dimension, categories in EMOTION_DIMENSIONS.items():
            dimension_score = {}

            for level, emotion_list in categories.items():
                score = sum(emotions.get(emotion, 0) for emotion in emotion_list)
                dimension_score[level] = score

            dimensional_scores[dimension] = dimension_score

        return dimensional_scores

    def _apply_dimensional_constraints(self, emotions: Dict[str, float], dimensional_scores: Dict[str, float]) -> Dict[str, float]:
        """应用维度约束来调整情感分数"""
        adjusted_emotions = emotions.copy()

        # 能量维度约束 - 高能量和低能量情感不应同时强烈
        energy_scores = dimensional_scores.get("energy", {})
        high_energy = energy_scores.get("high", 0)
        low_energy = energy_scores.get("low", 0)

        if high_energy > 0.8 and low_energy > 0.6:
            # 降低低能量情感
            for emotion in EMOTION_DIMENSIONS["energy"]["low"]:
                if emotion in adjusted_emotions:
                    adjusted_emotions[emotion] *= 0.7

        # 情感价值约束 - 正面和负面情感的平衡
        valence_scores = dimensional_scores.get("valence", {})
        positive = valence_scores.get("positive", 0)
        negative = valence_scores.get("negative", 0)

        if positive > 0.8 and negative > 0.5:
            # 适度降低负面情感
            for emotion in EMOTION_DIMENSIONS["valence"]["negative"]:
                if emotion in adjusted_emotions:
                    adjusted_emotions[emotion] *= 0.8

        return adjusted_emotions

    async def _analyze_shot_emotions(self, shots_info: List[Dict], base_emotions: Dict[str, float]) -> List[Dict[str, float]]:
        """分析每个分镜的情感"""
        shot_emotions = []

        for shot in shots_info:
            shot_description = shot.get("description", "")

            if shot_description:
                # 对每个分镜进行情感分析
                shot_emotion = await self._llm_emotion_analysis(shot_description)

                # 与全局情感进行融合
                fused_emotion = self._fuse_emotion_results([
                    (base_emotions, 0.4),  # 全局情感权重40%
                    (shot_emotion, 0.6)    # 分镜情感权重60%
                ])

                shot_emotions.append(fused_emotion)
            else:
                # 没有描述的分镜使用全局情感
                shot_emotions.append(base_emotions.copy())

        return shot_emotions

    def _calculate_confidence_score(self, base_emotions: Dict[str, float], final_emotions: Dict[str, float], text: str) -> float:
        """计算置信度分数"""
        confidence = 0.5  # 基础置信度

        # 文本长度影响置信度
        text_length = len(text)
        if text_length > 100:
            confidence += 0.2
        elif text_length > 50:
            confidence += 0.1

        # 情感强度影响置信度
        max_emotion_score = max(final_emotions.values()) if final_emotions else 0
        confidence += max_emotion_score * 0.3

        # 情感一致性影响置信度
        consistency = self._calculate_emotion_consistency(base_emotions, final_emotions)
        confidence += consistency * 0.2

        return min(confidence, 1.0)

    def _calculate_emotion_consistency(self, base_emotions: Dict[str, float], final_emotions: Dict[str, float]) -> float:
        """计算情感一致性"""
        if not base_emotions or not final_emotions:
            return 0.0

        # 计算相关系数
        base_values = [base_emotions.get(emotion, 0) for emotion in EMOTION_CATEGORIES]
        final_values = [final_emotions.get(emotion, 0) for emotion in EMOTION_CATEGORIES]

        correlation = np.corrcoef(base_values, final_values)[0, 1]
        return max(0, correlation) if not np.isnan(correlation) else 0.0

    def _generate_emotion_tags(self, emotions: Dict[str, float]) -> List[str]:
        """生成情感标签"""
        tags = []

        # 主要情感标签
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        for emotion, score in sorted_emotions[:3]:
            if score > 20:  # 只有超过20%的情感才生成标签
                if score > 50:
                    tags.append(f"强{emotion}")
                elif score > 30:
                    tags.append(f"中{emotion}")
                else:
                    tags.append(f"轻{emotion}")

        # 组合情感标签
        if len(sorted_emotions) >= 2:
            first_emotion, first_score = sorted_emotions[0]
            second_emotion, second_score = sorted_emotions[1]

            if first_score > 40 and second_score > 25:
                # 检查情感兼容性
                compatibility = EMOTION_COMPATIBILITY.get(first_emotion, {}).get(second_emotion, 0)
                if compatibility > 0.5:
                    tags.append(f"{first_emotion}+{second_emotion}")

        return tags

    async def generate_emotion_curve(self, emotions: Dict[str, float], duration: float, shots_info: List[Dict] = None) -> EmotionCurve:
        """生成情感曲线"""
        # 时间点设置 - 每5秒一个点
        time_step = 5.0
        timeline = list(np.arange(0, duration + time_step, time_step))

        # 初始化情感值
        emotion_values = {emotion: [] for emotion in EMOTION_CATEGORIES}

        if shots_info and len(shots_info) > 1:
            # 基于分镜生成曲线
            await self._generate_shot_based_curve(emotions, timeline, shots_info, emotion_values)
        else:
            # 基于整体情感生成平滑曲线
            self._generate_smooth_curve(emotions, timeline, emotion_values)

        # 找出峰值时刻
        peak_moments = self._find_peak_moments(timeline, emotion_values)

        # 找出情感转换点
        transitions = self._find_emotion_transitions(timeline, emotion_values)

        return EmotionCurve(
            timeline=timeline,
            emotion_values=emotion_values,
            peak_moments=peak_moments,
            transitions=transitions
        )

    async def _generate_shot_based_curve(self, base_emotions: Dict[str, float], timeline: List[float], shots_info: List[Dict], emotion_values: Dict[str, List[float]]):
        """基于分镜生成情感曲线"""
        current_time = 0.0
        shot_index = 0

        for time_point in timeline:
            # 找到当前时间点对应的分镜
            while shot_index < len(shots_info) and current_time + shots_info[shot_index].get("duration", 5.0) < time_point:
                current_time += shots_info[shot_index].get("duration", 5.0)
                shot_index += 1

            if shot_index < len(shots_info):
                # 获取当前分镜的情感
                shot = shots_info[shot_index]
                shot_emotions = await self._llm_emotion_analysis(shot.get("description", ""))

                # 与基础情感融合
                current_emotions = self._fuse_emotion_results([
                    (base_emotions, 0.3),
                    (shot_emotions, 0.7)
                ])
            else:
                # 超出分镜范围，使用基础情感
                current_emotions = base_emotions

            # 添加随机波动
            for emotion in EMOTION_CATEGORIES:
                base_value = current_emotions.get(emotion, 0.1)
                noise = np.random.uniform(-0.1, 0.1)
                final_value = max(0, base_value + noise)
                emotion_values[emotion].append(final_value)

    def _generate_smooth_curve(self, emotions: Dict[str, float], timeline: List[float], emotion_values: Dict[str, List[float]]):
        """生成平滑的情感曲线"""
        for emotion in EMOTION_CATEGORIES:
            base_value = emotions.get(emotion, 0.1)

            # 生成平滑曲线，添加自然波动
            for i, time_point in enumerate(timeline):
                # 使用正弦波产生自然波动
                wave = np.sin(time_point / 10) * 0.1
                trend = np.sin(time_point / 30) * 0.05  # 长期趋势
                noise = np.random.uniform(-0.05, 0.05)  # 随机噪声

                final_value = max(0, base_value + wave + trend + noise)
                emotion_values[emotion].append(final_value)

    def _find_peak_moments(self, timeline: List[float], emotion_values: Dict[str, List[float]]) -> List[Tuple[float, str, float]]:
        """找出情感峰值时刻"""
        peak_moments = []

        for emotion, values in emotion_values.items():
            if len(values) < 3:
                continue

            # 找局部最大值
            for i in range(1, len(values) - 1):
                if values[i] > values[i-1] and values[i] > values[i+1] and values[i] > 0.6:
                    peak_moments.append((timeline[i], emotion, values[i]))

        # 按强度排序，取前5个
        peak_moments.sort(key=lambda x: x[2], reverse=True)
        return peak_moments[:5]

    def _find_emotion_transitions(self, timeline: List[float], emotion_values: Dict[str, List[float]]) -> List[Tuple[float, str, str]]:
        """找出情感转换点"""
        transitions = []

        # 计算每个时间点的主导情感
        dominant_emotions = []
        for i in range(len(timeline)):
            point_emotions = {emotion: values[i] for emotion, values in emotion_values.items()}
            dominant = max(point_emotions, key=point_emotions.get)
            dominant_emotions.append(dominant)

        # 找出转换点
        for i in range(1, len(dominant_emotions)):
            if dominant_emotions[i] != dominant_emotions[i-1]:
                transitions.append((timeline[i], dominant_emotions[i-1], dominant_emotions[i]))

        return transitions

    async def generate_music_recommendations(self, emotions: Dict[str, float]) -> Dict[str, Any]:
        """生成音乐匹配建议"""
        recommendations = {
            "bgm_style": [],
            "tempo_range": {"min": 60, "max": 120},
            "key_suggestions": [],
            "instrument_preferences": [],
            "genre_recommendations": []
        }

        # 主导情感
        dominant_emotion = max(emotions, key=emotions.get)
        dominant_score = emotions[dominant_emotion]

        # 基于主导情感推荐音乐风格
        music_mapping = {
            "激昂": {
                "tempo_range": {"min": 120, "max": 180},
                "genres": ["电子音乐", "摇滚", "古典交响"],
                "instruments": ["电吉他", "鼓组", "铜管乐器"],
                "keys": ["C大调", "D大调", "E大调"]
            },
            "温馨": {
                "tempo_range": {"min": 60, "max": 100},
                "genres": ["民谣", "轻音乐", "古典室内乐"],
                "instruments": ["木吉他", "钢琴", "弦乐"],
                "keys": ["F大调", "G大调", "降E大调"]
            },
            "悬疑": {
                "tempo_range": {"min": 80, "max": 120},
                "genres": ["氛围音乐", "电影配乐", "实验音乐"],
                "instruments": ["合成器", "弦乐颤音", "打击乐"],
                "keys": ["小调", "C小调", "A小调"]
            },
            "幽默": {
                "tempo_range": {"min": 100, "max": 140},
                "genres": ["爵士", "布鲁斯", "轻快流行"],
                "instruments": ["萨克斯", "钢琴", "铜管"],
                "keys": ["F大调", "降B大调", "C大调"]
            }
        }

        if dominant_emotion in music_mapping:
            mapping = music_mapping[dominant_emotion]
            recommendations.update({
                "tempo_range": mapping["tempo_range"],
                "genre_recommendations": mapping["genres"],
                "instrument_preferences": mapping["instruments"],
                "key_suggestions": mapping["keys"]
            })

        # 考虑次要情感的影响
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        if len(sorted_emotions) > 1:
            secondary_emotion, secondary_score = sorted_emotions[1]
            if secondary_score > 20:  # 次要情感有显著影响
                recommendations["bgm_style"].append(f"融合{dominant_emotion}与{secondary_emotion}特色")

        # 添加动态变化建议
        if len([e for e in emotions.values() if e > 25]) > 2:
            recommendations["bgm_style"].append("建议使用动态变化的音乐，体现情感层次")

        return recommendations