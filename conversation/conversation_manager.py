"""
对话管理系统 - 实现智能对话修改功能
"""

import re
import json
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

logger = logging.getLogger(__name__)


class ModificationType(Enum):
    """修改类型枚举"""
    ADD = "add"              # 添加元素
    REMOVE = "remove"        # 删除元素
    REPLACE = "replace"      # 替换元素
    ADJUST = "adjust"        # 调整参数
    ENHANCE = "enhance"      # 增强效果
    COMPLETE = "complete"    # 完全重做


class ModificationScope(Enum):
    """修改范围枚举"""
    FULL = "full"            # 全局修改
    PARTIAL = "partial"      # 部分修改
    SPECIFIC = "specific"    # 特定节点修改


@dataclass
class ConversationState:
    """对话状态"""
    conversation_id: str
    messages: List[Dict[str, Any]] = field(default_factory=list)
    current_context: Dict[str, Any] = field(default_factory=dict)
    generation_history: List[Dict[str, Any]] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


class ConversationIntentAnalyzer:
    """对话意图分析器 - 理解用户的修改意图"""

    # 修改意图关键词映射
    INTENT_KEYWORDS = {
        ModificationType.ADD: ["添加", "加入", "增加", "补充", "加上", "add", "include", "append"],
        ModificationType.REMOVE: ["删除", "去掉", "移除", "删掉", "去除", "remove", "delete", "exclude"],
        ModificationType.REPLACE: ["替换", "换成", "改成", "更换", "修改为", "replace", "change to", "switch to"],
        ModificationType.ADJUST: ["调整", "改为", "设置", "修改", "调节", "adjust", "modify", "set to"],
        ModificationType.ENHANCE: ["增强", "加强", "提升", "优化", "改进", "enhance", "improve", "boost"],
        ModificationType.COMPLETE: ["重新", "重做", "全部", "完全", "redo", "remake", "completely"]
    }

    # 目标关键词映射
    TARGET_KEYWORDS = {
        "duration": ["时长", "时间", "秒", "分钟", "duration", "length", "seconds", "minutes"],
        "style": ["风格", "样式", "主题", "色调", "style", "theme", "tone", "mood"],
        "music": ["音乐", "配乐", "背景音", "BGM", "music", "audio", "soundtrack"],
        "effects": ["特效", "效果", "动画", "转场", "effects", "animation", "transition"],
        "text": ["文字", "字幕", "标题", "文案", "text", "subtitle", "title", "caption"],
        "speed": ["速度", "节奏", "快慢", "speed", "pace", "tempo", "rhythm"]
    }

    def __init__(self):
        self.logger = logger.getChild('IntentAnalyzer')

    def analyze_intent(self,
                       current_description: str,
                       previous_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        分析用户意图

        Args:
            current_description: 当前用户描述
            previous_context: 之前的上下文

        Returns:
            意图分析结果
        """
        self.logger.info(f"分析用户意图: {current_description[:100]}...")

        # 1. 识别修改类型
        modification_type = self._identify_modification_type(current_description)

        # 2. 识别修改目标
        targets = self._identify_targets(current_description)

        # 3. 提取具体修改内容
        modifications = self._extract_modifications(current_description, targets)

        # 4. 分析修改范围
        scope = self._analyze_scope(modifications, previous_context)

        # 5. 生成修改指令
        instructions = self._generate_instructions(
            modification_type, targets, modifications, previous_context
        )

        result = {
            "modification_type": modification_type.value,
            "targets": targets,
            "modifications": modifications,
            "scope": scope.value,
            "instructions": instructions,
            "confidence": self._calculate_confidence(current_description, targets)
        }

        self.logger.info(f"意图分析完成: {result}")
        return result

    def _identify_modification_type(self, description: str) -> ModificationType:
        """识别修改类型"""
        description_lower = description.lower()

        # 统计每种类型的关键词出现次数
        type_scores = {}
        for mod_type, keywords in self.INTENT_KEYWORDS.items():
            score = sum(1 for keyword in keywords if keyword in description_lower)
            if score > 0:
                type_scores[mod_type] = score

        # 返回得分最高的类型
        if type_scores:
            return max(type_scores.items(), key=lambda x: x[1])[0]

        # 默认为调整类型
        return ModificationType.ADJUST

    def _identify_targets(self, description: str) -> List[str]:
        """识别修改目标"""
        description_lower = description.lower()
        targets = []

        for target, keywords in self.TARGET_KEYWORDS.items():
            if any(keyword in description_lower for keyword in keywords):
                targets.append(target)

        return targets if targets else ["general"]

    def _extract_modifications(self, description: str, targets: List[str]) -> Dict[str, Any]:
        """提取具体修改内容"""
        modifications = {}

        # 提取时长修改
        if "duration" in targets:
            duration_match = re.search(r'(\d+)\s*[秒分]|(\d+)\s*s', description)
            if duration_match:
                seconds = int(duration_match.group(1) or duration_match.group(2))
                if '分' in description:
                    seconds *= 60
                modifications["duration"] = seconds

        # 提取颜色/风格修改
        if "style" in targets:
            style_keywords = ["现代", "复古", "科技", "自然", "简约", "华丽", "暗色", "明亮"]
            for keyword in style_keywords:
                if keyword in description:
                    modifications.setdefault("style", []).append(keyword)

        # 提取其他文本修改
        modifications["description"] = description

        return modifications

    def _analyze_scope(self, modifications: Dict[str, Any], previous_context: Optional[Dict]) -> ModificationScope:
        """分析修改范围"""
        if not previous_context:
            return ModificationScope.FULL

        # 如果只修改了少数属性，认为是部分修改
        if len(modifications) <= 2:
            return ModificationScope.PARTIAL

        # 如果修改了多个属性，认为是全局修改
        if len(modifications) > 3:
            return ModificationScope.FULL

        return ModificationScope.SPECIFIC

    def _generate_instructions(self,
                              mod_type: ModificationType,
                              targets: List[str],
                              modifications: Dict[str, Any],
                              previous_context: Optional[Dict]) -> List[Dict[str, Any]]:
        """生成具体的修改指令"""
        instructions = []

        # 根据修改类型生成指令
        if mod_type == ModificationType.ADJUST:
            for target in targets:
                if target in modifications:
                    instructions.append({
                        "action": "update",
                        "target": target,
                        "value": modifications[target],
                        "preserve_others": True
                    })

        elif mod_type == ModificationType.ADD:
            for target in targets:
                instructions.append({
                    "action": "add",
                    "target": target,
                    "value": modifications.get(target),
                    "merge_with_existing": True
                })

        elif mod_type == ModificationType.REMOVE:
            for target in targets:
                instructions.append({
                    "action": "remove",
                    "target": target,
                    "preserve_structure": True
                })

        elif mod_type == ModificationType.REPLACE:
            for target in targets:
                instructions.append({
                    "action": "replace",
                    "target": target,
                    "old_value": previous_context.get(target) if previous_context else None,
                    "new_value": modifications.get(target)
                })

        return instructions

    def _calculate_confidence(self, description: str, targets: List[str]) -> float:
        """计算意图识别的置信度"""
        # 基础置信度
        confidence = 0.5

        # 如果识别到明确的目标，增加置信度
        if targets and targets != ["general"]:
            confidence += 0.2 * len(targets)

        # 如果包含明确的数字或具体值，增加置信度
        if re.search(r'\d+', description):
            confidence += 0.1

        return min(confidence, 1.0)


class ConversationHistoryManager:
    """对话历史管理器 - 管理和追踪对话状态"""

    def __init__(self):
        self.conversations: Dict[str, ConversationState] = {}
        self.logger = logger.getChild('HistoryManager')

    def get_or_create_conversation(self, conversation_id: str) -> ConversationState:
        """获取或创建对话"""
        if conversation_id not in self.conversations:
            self.conversations[conversation_id] = ConversationState(
                conversation_id=conversation_id
            )
            self.logger.info(f"创建新对话: {conversation_id}")

        return self.conversations[conversation_id]

    def add_message(self,
                   conversation_id: str,
                   message_id: str,
                   request_data: Dict[str, Any],
                   intent_analysis: Optional[Dict[str, Any]] = None):
        """添加消息到对话历史"""
        conversation = self.get_or_create_conversation(conversation_id)

        message = {
            "message_id": message_id,
            "timestamp": datetime.now().isoformat(),
            "request": request_data,
            "intent_analysis": intent_analysis
        }

        conversation.messages.append(message)
        conversation.updated_at = datetime.now()

        self.logger.info(f"添加消息到对话 {conversation_id}: {message_id}")

    def add_generation_result(self,
                            conversation_id: str,
                            task_id: str,
                            result: Dict[str, Any]):
        """添加生成结果到历史"""
        conversation = self.get_or_create_conversation(conversation_id)

        generation = {
            "task_id": task_id,
            "timestamp": datetime.now().isoformat(),
            "result": result
        }

        conversation.generation_history.append(generation)
        conversation.updated_at = datetime.now()

        self.logger.info(f"添加生成结果到对话 {conversation_id}: {task_id}")

    def get_previous_generation(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """获取上一次的生成结果"""
        if conversation_id not in self.conversations:
            return None

        conversation = self.conversations[conversation_id]
        if conversation.generation_history:
            return conversation.generation_history[-1]

        return None

    def get_conversation_context(self, conversation_id: str) -> Dict[str, Any]:
        """获取对话上下文"""
        conversation = self.get_or_create_conversation(conversation_id)

        return {
            "conversation_id": conversation_id,
            "message_count": len(conversation.messages),
            "generation_count": len(conversation.generation_history),
            "current_context": conversation.current_context,
            "last_update": conversation.updated_at.isoformat()
        }

    def update_context(self, conversation_id: str, context_updates: Dict[str, Any]):
        """更新对话上下文"""
        conversation = self.get_or_create_conversation(conversation_id)
        conversation.current_context.update(context_updates)
        conversation.updated_at = datetime.now()

        self.logger.info(f"更新对话上下文 {conversation_id}: {context_updates}")


class IncrementalModificationEngine:
    """增量修改引擎 - 实现智能的增量修改"""

    def __init__(self):
        self.logger = logger.getChild('ModificationEngine')

    def apply_modifications(self,
                           previous_result: Dict[str, Any],
                           modifications: Dict[str, Any],
                           instructions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        应用修改到之前的结果

        Args:
            previous_result: 之前的生成结果
            modifications: 修改内容
            instructions: 修改指令

        Returns:
            修改后的结果
        """
        self.logger.info("开始应用增量修改")

        # 复制之前的结果
        modified_result = previous_result.copy()

        # 应用每个修改指令
        for instruction in instructions:
            action = instruction.get("action")
            target = instruction.get("target")

            if action == "update":
                self._apply_update(modified_result, target, instruction)
            elif action == "add":
                self._apply_addition(modified_result, target, instruction)
            elif action == "remove":
                self._apply_removal(modified_result, target, instruction)
            elif action == "replace":
                self._apply_replacement(modified_result, target, instruction)

        # 更新修改时间戳
        modified_result["last_modified"] = datetime.now().isoformat()
        modified_result["is_incremental"] = True

        return modified_result

    def _apply_update(self, result: Dict, target: str, instruction: Dict):
        """应用更新操作"""
        value = instruction.get("value")
        preserve_others = instruction.get("preserve_others", True)

        if target == "duration":
            result["target_duration"] = value
            # 标记需要调整的节点
            result.setdefault("nodes_to_update", []).append("bgm_composition")
            result.setdefault("nodes_to_update", []).append("subtitle_generation")

        elif target == "style":
            if preserve_others:
                result.setdefault("style_keywords", []).extend(value)
            else:
                result["style_keywords"] = value
            result.setdefault("nodes_to_update", []).append("filter_application")
            result.setdefault("nodes_to_update", []).append("dynamic_effects")

    def _apply_addition(self, result: Dict, target: str, instruction: Dict):
        """应用添加操作"""
        value = instruction.get("value")
        merge = instruction.get("merge_with_existing", True)

        if target in result and merge:
            if isinstance(result[target], list):
                result[target].extend(value if isinstance(value, list) else [value])
            elif isinstance(result[target], dict):
                result[target].update(value)
            else:
                result[target] = value
        else:
            result[target] = value

    def _apply_removal(self, result: Dict, target: str, instruction: Dict):
        """应用删除操作"""
        if target in result:
            del result[target]
            result.setdefault("removed_elements", []).append(target)

    def _apply_replacement(self, result: Dict, target: str, instruction: Dict):
        """应用替换操作"""
        new_value = instruction.get("new_value")
        result[target] = new_value
        result.setdefault("replaced_elements", []).append({
            "target": target,
            "old": instruction.get("old_value"),
            "new": new_value
        })


class ConversationManager:
    """对话管理器 - 统一管理对话功能"""

    def __init__(self):
        self.intent_analyzer = ConversationIntentAnalyzer()
        self.history_manager = ConversationHistoryManager()
        self.modification_engine = IncrementalModificationEngine()
        self.logger = logger.getChild('ConversationManager')

    async def process_conversation_request(self,
                                          request_data: Dict[str, Any],
                                          conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        处理对话请求

        Args:
            request_data: 请求数据
            conversation_context: 对话上下文

        Returns:
            处理结果，包含修改指令和上下文
        """
        conversation_id = conversation_context.get("conversation_id")
        message_id = conversation_context.get("message_id")
        is_regeneration = conversation_context.get("is_regeneration", False)

        self.logger.info(f"处理对话请求: {conversation_id}/{message_id}, 重新生成: {is_regeneration}")

        # 获取对话历史
        conversation = self.history_manager.get_or_create_conversation(conversation_id)

        # 如果是重新生成，进行意图分析
        intent_analysis = None
        modifications_to_apply = None

        if is_regeneration:
            # 获取之前的生成结果
            previous_generation = self.history_manager.get_previous_generation(conversation_id)

            if previous_generation:
                # 分析用户意图
                user_description = request_data.get("user_description_id", "")
                intent_analysis = self.intent_analyzer.analyze_intent(
                    user_description,
                    previous_generation.get("result")
                )

                # 如果识别到明确的修改意图，准备增量修改
                if intent_analysis["confidence"] > 0.6:
                    modifications_to_apply = self.modification_engine.apply_modifications(
                        previous_generation["result"],
                        intent_analysis["modifications"],
                        intent_analysis["instructions"]
                    )

        # 添加消息到历史
        self.history_manager.add_message(
            conversation_id,
            message_id,
            request_data,
            intent_analysis
        )

        # 构建处理结果
        result = {
            "conversation_id": conversation_id,
            "message_id": message_id,
            "is_regeneration": is_regeneration,
            "intent_analysis": intent_analysis,
            "modifications": modifications_to_apply,
            "conversation_context": self.history_manager.get_conversation_context(conversation_id)
        }

        # 如果有增量修改，标记相关节点
        if modifications_to_apply:
            result["incremental_mode"] = True
            result["nodes_to_update"] = modifications_to_apply.get("nodes_to_update", [])
            result["skip_nodes"] = self._determine_skip_nodes(modifications_to_apply)

        return result

    def _determine_skip_nodes(self, modifications: Dict[str, Any]) -> List[str]:
        """确定可以跳过的节点"""
        skip_nodes = []

        # 如果只修改了时长，可以跳过风格相关节点
        if "duration" in modifications and "style" not in modifications:
            skip_nodes.extend([
                "emotion_analysis",
                "filter_application",
                "dynamic_effects"
            ])

        # 如果只修改了风格，可以跳过结构相关节点
        if "style" in modifications and "duration" not in modifications:
            skip_nodes.extend([
                "shot_block_generation",
                "bgm_anchor_planning"
            ])

        return skip_nodes

    def save_generation_result(self, conversation_id: str, task_id: str, result: Dict[str, Any]):
        """保存生成结果"""
        self.history_manager.add_generation_result(conversation_id, task_id, result)
        self.logger.info(f"保存生成结果: {conversation_id}/{task_id}")


# 创建全局实例
conversation_manager = ConversationManager()