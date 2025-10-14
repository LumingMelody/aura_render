"""
对话管理模块 - 智能对话修改功能

提供以下功能：
- 对话意图解析
- 对话历史管理
- 增量修改引擎
- 智能节点优化
"""

from .conversation_manager import (
    ConversationManager,
    ConversationIntentAnalyzer,
    ConversationHistoryManager,
    IncrementalModificationEngine,
    conversation_manager
)

__all__ = [
    'ConversationManager',
    'ConversationIntentAnalyzer',
    'ConversationHistoryManager',
    'IncrementalModificationEngine',
    'conversation_manager'
]