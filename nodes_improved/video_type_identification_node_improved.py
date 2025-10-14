#!/usr/bin/env python3
"""
Improved Video Type Identification Node

Enhanced version with better AI integration, caching, and error handling.
"""

import sys
from pathlib import Path
import asyncio
import json
from typing import Dict, Any, List

# Add project root for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from nodes_improved.base_node_improved import BaseNodeImproved, call_ai_service, NodeExecutionError
from config import settings

# Supported video types with detailed descriptions
VIDEO_TYPES_DATABASE = {
    "产品宣传片": {
        "description": "展示产品特性和优势的商业宣传视频",
        "typical_duration": [30, 60, 90],
        "key_elements": ["产品展示", "功能介绍", "用户场景"],
        "structure_template": "intro_problem_solution_cta"
    },
    "品牌形象片": {
        "description": "展示企业文化和品牌价值的形象宣传",
        "typical_duration": [60, 120, 180],
        "key_elements": ["企业理念", "团队风貌", "发展历程"],
        "structure_template": "story_driven_brand"
    },
    "教学视频": {
        "description": "知识传授和技能教学类视频",
        "typical_duration": [300, 600, 1200],
        "key_elements": ["知识点", "步骤演示", "练习案例"],
        "structure_template": "intro_teach_practice_summary"
    },
    "VLOG": {
        "description": "个人生活记录和分享类视频",
        "typical_duration": [180, 300, 600],
        "key_elements": ["个人视角", "日常记录", "情感表达"],
        "structure_template": "chronological_narrative"
    },
    "短视频故事": {
        "description": "简短的故事性内容，适合社交媒体",
        "typical_duration": [15, 30, 60],
        "key_elements": ["冲突设置", "转折点", "情感共鸣"],
        "structure_template": "hook_build_payoff"
    },
    "新闻播报": {
        "description": "新闻事件报道和信息传递",
        "typical_duration": [60, 120, 300],
        "key_elements": ["事实陈述", "背景介绍", "专家观点"],
        "structure_template": "lead_body_conclusion"
    },
    "访谈节目": {
        "description": "对话形式的深度交流节目",
        "typical_duration": [600, 1800, 3600],
        "key_elements": ["主持引导", "嘉宾观点", "互动交流"],
        "structure_template": "intro_questions_insights"
    }
}


class VideoTypeIdentificationNodeImproved(BaseNodeImproved):
    """
    Improved Video Type Identification Node
    
    Features:
    - Enhanced AI prompt engineering
    - Confidence scoring
    - Caching for similar requests
    - Multiple structure templates
    - Detailed reasoning output
    """
    
    node_name = "VideoTypeIdentificationNode"
    node_description = "智能识别视频类型并生成结构模板"
    node_version = "2.0.0"
    
    required_inputs = [
        {
            "name": "theme_id",
            "label": "视频主题",
            "type": str,
            "required": True,
            "description": "视频的主要话题或背景"
        },
        {
            "name": "keywords_id", 
            "label": "关键词",
            "type": list,
            "required": True,
            "description": "与视频内容相关的关键词列表"
        },
        {
            "name": "target_duration_id",
            "label": "目标时长",
            "type": int,
            "required": True,
            "description": "视频的目标长度（秒）"
        },
        {
            "name": "user_description_id",
            "label": "用户描述",
            "type": str,
            "required": True,
            "description": "用户对视频的详细描述"
        }
    ]
    
    output_schema = [
        {
            "name": "video_type_id",
            "type": str,
            "description": "识别的视频类型"
        },
        {
            "name": "structure_template_id",
            "type": str,
            "description": "推荐的视频结构模板"
        },
        {
            "name": "confidence_score",
            "type": float,
            "description": "识别置信度 (0-1)"
        },
        {
            "name": "reasoning",
            "type": str,
            "description": "识别理由和建议"
        },
        {
            "name": "alternative_types",
            "type": list,
            "description": "其他可能的视频类型"
        }
    ]
    
    def __init__(self, node_id: str):
        super().__init__(node_id)
        self._cache = {}  # Simple in-memory cache
    
    def _generate_cache_key(self, context: Dict[str, Any]) -> str:
        """Generate cache key for similar requests"""
        key_data = {
            "theme": context["theme_id"],
            "keywords": sorted(context["keywords_id"]),
            "duration_range": self._get_duration_range(context["target_duration_id"])
        }
        return str(hash(json.dumps(key_data, sort_keys=True)))
    
    def _get_duration_range(self, duration: int) -> str:
        """Categorize duration into ranges"""
        if duration <= 60:
            return "short"
        elif duration <= 300:
            return "medium"
        elif duration <= 900:
            return "long"
        else:
            return "extended"
    
    def _build_ai_prompt(self, context: Dict[str, Any]) -> str:
        """Build comprehensive AI prompt for video type identification"""
        
        # Build context information
        theme = context["theme_id"]
        keywords = ", ".join(context["keywords_id"])
        duration = context["target_duration_id"]
        description = context["user_description_id"]
        
        # Build video types information
        types_info = []
        for video_type, info in VIDEO_TYPES_DATABASE.items():
            types_info.append(f"- {video_type}: {info['description']}")
        
        prompt = f"""
# 视频类型识别任务

你是一个专业的视频内容分析师，需要根据用户提供的信息识别最适合的视频类型。

## 用户输入信息：
- 主题：{theme}
- 关键词：{keywords}
- 目标时长：{duration}秒
- 详细描述：{description}

## 可选视频类型：
{chr(10).join(types_info)}

## 分析要求：
1. 分析用户需求的核心特征
2. 考虑时长与内容类型的匹配度
3. 评估关键词与各类型的相关性
4. 给出主要推荐类型和置信度
5. 提供2-3个备选类型

## 输出格式（JSON）：
{{
    "primary_type": "主要推荐的视频类型",
    "confidence": 0.95,
    "reasoning": "详细的分析理由，包括为什么选择这个类型，考虑了哪些因素",
    "structure_template": "推荐的结构模板ID",
    "alternative_types": [
        {{
            "type": "备选类型1",
            "confidence": 0.75,
            "reason": "选择理由"
        }},
        {{
            "type": "备选类型2", 
            "confidence": 0.65,
            "reason": "选择理由"
        }}
    ],
    "duration_analysis": "时长与类型匹配度分析",
    "suggestions": "针对用户需求的具体建议"
}}

请进行专业分析并输出结果：
"""
        
        return prompt
    
    async def generate_async(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async implementation of video type identification
        """
        
        # Check cache first
        cache_key = self._generate_cache_key(context)
        if cache_key in self._cache:
            self.logger.info("📋 Using cached result")
            return self._cache[cache_key]
        
        # Build AI prompt
        prompt = self._build_ai_prompt(context)
        
        # Call AI service
        ai_response = await call_ai_service(
            prompt,
            parse_json=True,
            json_schema={
                "type": "object",
                "properties": {
                    "primary_type": {"type": "string"},
                    "confidence": {"type": "number"},
                    "reasoning": {"type": "string"},
                    "structure_template": {"type": "string"},
                    "alternative_types": {"type": "array"}
                }
            }
        )
        
        if not ai_response["success"]:
            # Fallback to rule-based analysis
            self.logger.warning("🤖 AI service failed, using rule-based fallback")
            return self._rule_based_analysis(context)
        
        # Parse AI response
        try:
            ai_result = ai_response["result"]
            
            # Validate the result
            primary_type = ai_result.get("primary_type", "产品宣传片")
            if primary_type not in VIDEO_TYPES_DATABASE:
                primary_type = self._find_closest_type(primary_type)
            
            # Build result
            result = {
                "video_type_id": primary_type,
                "structure_template_id": VIDEO_TYPES_DATABASE[primary_type]["structure_template"],
                "confidence_score": min(ai_result.get("confidence", 0.8), 1.0),
                "reasoning": ai_result.get("reasoning", "AI分析得出的结果"),
                "alternative_types": ai_result.get("alternative_types", []),
                "duration_analysis": ai_result.get("duration_analysis", ""),
                "suggestions": ai_result.get("suggestions", ""),
                "type_details": VIDEO_TYPES_DATABASE[primary_type]
            }
            
            # Cache the result
            self._cache[cache_key] = result
            
            return result
            
        except Exception as e:
            self.logger.error(f"Failed to parse AI response: {e}")
            return self._rule_based_analysis(context)
    
    def _rule_based_analysis(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Fallback rule-based analysis when AI service fails
        """
        theme = context["theme_id"].lower()
        keywords = [k.lower() for k in context["keywords_id"]]
        duration = context["target_duration_id"]
        description = context["user_description_id"].lower()
        
        # Simple keyword matching
        type_scores = {}
        
        for video_type, info in VIDEO_TYPES_DATABASE.items():
            score = 0
            
            # Check theme matching
            if any(word in theme for word in ["产品", "宣传", "推广"]):
                if video_type == "产品宣传片":
                    score += 0.4
            
            if any(word in theme for word in ["教学", "教育", "培训"]):
                if video_type == "教学视频":
                    score += 0.4
            
            if any(word in theme for word in ["vlog", "生活", "日常"]):
                if video_type == "VLOG":
                    score += 0.4
            
            # Check duration matching
            typical_durations = info["typical_duration"]
            duration_score = 1.0 - min(abs(duration - d) / d for d in typical_durations)
            score += duration_score * 0.3
            
            # Check keyword matching
            for keyword in keywords:
                if keyword in " ".join(info["key_elements"]).lower():
                    score += 0.1
            
            type_scores[video_type] = score
        
        # Find best match
        best_type = max(type_scores, key=type_scores.get)
        best_score = type_scores[best_type]
        
        # Build alternative types
        sorted_types = sorted(type_scores.items(), key=lambda x: x[1], reverse=True)
        alternatives = [
            {
                "type": t,
                "confidence": round(s, 2),
                "reason": f"Rule-based matching score: {s:.2f}"
            }
            for t, s in sorted_types[1:4]  # Top 3 alternatives
        ]
        
        return {
            "video_type_id": best_type,
            "structure_template_id": VIDEO_TYPES_DATABASE[best_type]["structure_template"],
            "confidence_score": round(best_score, 2),
            "reasoning": f"基于规则分析：主题匹配、时长适配、关键词相关性综合评分",
            "alternative_types": alternatives,
            "type_details": VIDEO_TYPES_DATABASE[best_type],
            "fallback_method": True
        }
    
    def _find_closest_type(self, unknown_type: str) -> str:
        """Find closest known video type"""
        # Simple string matching fallback
        for known_type in VIDEO_TYPES_DATABASE:
            if unknown_type.lower() in known_type.lower() or known_type.lower() in unknown_type.lower():
                return known_type
        
        # Default fallback
        return "产品宣传片"


# Factory function
def create_video_type_identification_node(node_id: str) -> VideoTypeIdentificationNodeImproved:
    """Create video type identification node"""
    return VideoTypeIdentificationNodeImproved(node_id)


if __name__ == "__main__":
    # Test the node
    print("🧪 Testing VideoTypeIdentificationNode...")
    
    node = create_video_type_identification_node("video_type_001")
    
    test_context = {
        "theme_id": "AI产品介绍",
        "keywords_id": ["人工智能", "创新", "技术"],
        "target_duration_id": 60,
        "user_description_id": "想要制作一个60秒的AI产品宣传视频，展示我们的技术优势"
    }
    
    result = node.execute(test_context)
    print(f"✅ Result: {result}")
    print(f"📊 Node info: {node.get_node_info()}")