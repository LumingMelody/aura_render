"""
增强对话管理系统 - 支持mem0记忆管理和数据库存储
智能视频剪辑对话系统的核心对话管理组件
"""

import json
import logging
import sqlite3
import hashlib
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio

logger = logging.getLogger(__name__)


class ConversationStatus(Enum):
    """对话状态"""
    ACTIVE = "active"
    PAUSED = "paused"
    COMPLETED = "completed"
    EXPIRED = "expired"


@dataclass
class ConversationContext:
    """对话上下文数据结构"""
    conversation_id: str
    user_id: str
    project_name: str = ""
    created_at: datetime = None
    updated_at: datetime = None
    status: ConversationStatus = ConversationStatus.ACTIVE
    message_count: int = 0
    last_node_execution: Dict[str, Any] = None  # 最后执行的节点状态
    global_style: Dict[str, Any] = None  # 全局风格设置
    user_assets: List[Dict[str, Any]] = None  # 用户上传的素材
    generation_history: List[Dict[str, Any]] = None  # 生成历史

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
        if self.updated_at is None:
            self.updated_at = datetime.now()
        if self.last_node_execution is None:
            self.last_node_execution = {}
        if self.global_style is None:
            self.global_style = {}
        if self.user_assets is None:
            self.user_assets = []
        if self.generation_history is None:
            self.generation_history = []


class ConversationMemoryManager:
    """对话记忆管理器 - 集成mem0或本地实现"""

    def __init__(self, use_mem0: bool = False):
        self.use_mem0 = use_mem0
        self.logger = logger.getChild('MemoryManager')

        # 本地记忆存储
        self.memory_store = {}

        # 如果使用mem0，在这里初始化
        if use_mem0:
            self._init_mem0()

    def _init_mem0(self):
        """初始化mem0（如果可用）"""
        try:
            # 这里可以集成实际的mem0
            # import mem0
            # self.mem0_client = mem0.Client()
            self.logger.info("Mem0 initialized (placeholder)")
        except ImportError:
            self.logger.warning("Mem0 not available, using local memory")
            self.use_mem0 = False

    async def store_memory(self, conversation_id: str, key: str, value: Any, expire_hours: int = 24 * 7):
        """存储记忆"""
        memory_key = f"{conversation_id}:{key}"
        expire_time = datetime.now() + timedelta(hours=expire_hours)

        memory_item = {
            "value": value,
            "created_at": datetime.now().isoformat(),
            "expire_at": expire_time.isoformat()
        }

        if self.use_mem0:
            # 使用mem0存储
            # await self.mem0_client.store(memory_key, memory_item)
            pass
        else:
            # 本地存储
            self.memory_store[memory_key] = memory_item

        self.logger.debug(f"Stored memory: {memory_key}")

    async def retrieve_memory(self, conversation_id: str, key: str) -> Optional[Any]:
        """检索记忆"""
        memory_key = f"{conversation_id}:{key}"

        if self.use_mem0:
            # 从mem0检索
            # return await self.mem0_client.retrieve(memory_key)
            return None
        else:
            # 本地检索
            memory_item = self.memory_store.get(memory_key)
            if memory_item:
                # 检查是否过期
                expire_time = datetime.fromisoformat(memory_item["expire_at"])
                if datetime.now() > expire_time:
                    del self.memory_store[memory_key]
                    return None
                return memory_item["value"]
            return None

    async def update_memory(self, conversation_id: str, key: str, value: Any):
        """更新记忆"""
        await self.store_memory(conversation_id, key, value)

    async def clear_conversation_memory(self, conversation_id: str):
        """清除对话记忆"""
        if self.use_mem0:
            # 清除mem0中的记忆
            pass
        else:
            # 清除本地记忆
            keys_to_remove = [k for k in self.memory_store.keys() if k.startswith(f"{conversation_id}:")]
            for key in keys_to_remove:
                del self.memory_store[key]

        self.logger.info(f"Cleared memory for conversation: {conversation_id}")


class ConversationDatabase:
    """对话数据库管理器"""

    def __init__(self, db_path: str = "conversations.db"):
        self.db_path = db_path
        self.logger = logger.getChild('Database')
        self._init_database()

    def _init_database(self):
        """初始化数据库"""
        with sqlite3.connect(self.db_path) as conn:
            # 对话表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    project_name TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT DEFAULT 'active',
                    message_count INTEGER DEFAULT 0,
                    context_data TEXT  -- JSON格式存储上下文
                )
            """)

            # 消息表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    message_index INTEGER,
                    user_input TEXT,
                    system_response TEXT,
                    node_results TEXT,  -- JSON格式存储节点执行结果
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            """)

            # 素材表
            conn.execute("""
                CREATE TABLE IF NOT EXISTS user_assets (
                    asset_id TEXT PRIMARY KEY,
                    conversation_id TEXT,
                    asset_type TEXT,  -- 'video', 'image', 'audio'
                    file_path TEXT,
                    metadata TEXT,  -- JSON格式存储元数据
                    uploaded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (conversation_id) REFERENCES conversations (conversation_id)
                )
            """)

            conn.commit()

        self.logger.info(f"Database initialized: {self.db_path}")

    def create_conversation(self, conversation_id: str, user_id: str, project_name: str = "") -> bool:
        """创建新对话"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO conversations (conversation_id, user_id, project_name, context_data)
                    VALUES (?, ?, ?, ?)
                """, (conversation_id, user_id, project_name, "{}"))
                conn.commit()

            self.logger.info(f"Created conversation: {conversation_id}")
            return True
        except sqlite3.IntegrityError:
            self.logger.warning(f"Conversation already exists: {conversation_id}")
            return False

    def get_conversation(self, conversation_id: str) -> Optional[ConversationContext]:
        """获取对话上下文"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM conversations WHERE conversation_id = ?
                """, (conversation_id,))

                row = cursor.fetchone()
                if row:
                    context_data = json.loads(row['context_data'])

                    context = ConversationContext(
                        conversation_id=row['conversation_id'],
                        user_id=row['user_id'],
                        project_name=row['project_name'] or "",
                        created_at=datetime.fromisoformat(row['created_at']),
                        updated_at=datetime.fromisoformat(row['updated_at']),
                        status=ConversationStatus(row['status']),
                        message_count=row['message_count']
                    )

                    # 加载上下文数据
                    context.last_node_execution = context_data.get('last_node_execution', {})
                    context.global_style = context_data.get('global_style', {})
                    context.user_assets = context_data.get('user_assets', [])
                    context.generation_history = context_data.get('generation_history', [])

                    return context

                return None
        except Exception as e:
            self.logger.error(f"Error retrieving conversation {conversation_id}: {e}")
            return None

    def update_conversation(self, context: ConversationContext):
        """更新对话上下文"""
        try:
            context_data = {
                'last_node_execution': context.last_node_execution,
                'global_style': context.global_style,
                'user_assets': context.user_assets,
                'generation_history': context.generation_history
            }

            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    UPDATE conversations
                    SET updated_at = CURRENT_TIMESTAMP,
                        message_count = ?,
                        status = ?,
                        context_data = ?
                    WHERE conversation_id = ?
                """, (context.message_count, context.status.value, json.dumps(context_data), context.conversation_id))
                conn.commit()

            self.logger.debug(f"Updated conversation: {context.conversation_id}")
        except Exception as e:
            self.logger.error(f"Error updating conversation {context.conversation_id}: {e}")

    def add_message(self, conversation_id: str, message_id: str, message_index: int,
                    user_input: str, system_response: str = "", node_results: Dict = None):
        """添加消息记录"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO messages (message_id, conversation_id, message_index,
                                        user_input, system_response, node_results)
                    VALUES (?, ?, ?, ?, ?, ?)
                """, (message_id, conversation_id, message_index, user_input,
                     system_response, json.dumps(node_results or {})))
                conn.commit()

            self.logger.debug(f"Added message: {message_id}")
        except Exception as e:
            self.logger.error(f"Error adding message {message_id}: {e}")

    def get_conversation_messages(self, conversation_id: str, limit: int = 10) -> List[Dict]:
        """获取对话消息历史"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute("""
                    SELECT * FROM messages
                    WHERE conversation_id = ?
                    ORDER BY message_index DESC
                    LIMIT ?
                """, (conversation_id, limit))

                messages = []
                for row in cursor.fetchall():
                    messages.append({
                        'message_id': row['message_id'],
                        'message_index': row['message_index'],
                        'user_input': row['user_input'],
                        'system_response': row['system_response'],
                        'node_results': json.loads(row['node_results']),
                        'created_at': row['created_at']
                    })

                return list(reversed(messages))  # 按时间顺序返回
        except Exception as e:
            self.logger.error(f"Error retrieving messages for {conversation_id}: {e}")
            return []


class ConversationFileManager:
    """对话上下文文件管理器"""

    def __init__(self, base_dir: str = "conversation_contexts"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.logger = logger.getChild('FileManager')

    def _get_context_file_path(self, conversation_id: str) -> Path:
        """获取对话上下文文件路径"""
        # 使用conversation_id的hash来避免文件名过长
        file_hash = hashlib.md5(conversation_id.encode()).hexdigest()
        return self.base_dir / f"context_{file_hash}.json"

    def save_context_file(self, conversation_id: str, context_data: Dict[str, Any]):
        """保存对话上下文到文件"""
        try:
            file_path = self._get_context_file_path(conversation_id)

            context_file_data = {
                'conversation_id': conversation_id,
                'saved_at': datetime.now().isoformat(),
                'context': context_data
            }

            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(context_file_data, f, indent=2, ensure_ascii=False)

            self.logger.debug(f"Saved context file: {file_path}")
        except Exception as e:
            self.logger.error(f"Error saving context file for {conversation_id}: {e}")

    def load_context_file(self, conversation_id: str) -> Optional[Dict[str, Any]]:
        """从文件加载对话上下文"""
        try:
            file_path = self._get_context_file_path(conversation_id)

            if not file_path.exists():
                return None

            with open(file_path, 'r', encoding='utf-8') as f:
                context_file_data = json.load(f)

            return context_file_data['context']
        except Exception as e:
            self.logger.error(f"Error loading context file for {conversation_id}: {e}")
            return None

    def cleanup_old_files(self, days_old: int = 30):
        """清理旧的上下文文件"""
        cutoff_time = datetime.now() - timedelta(days=days_old)

        for file_path in self.base_dir.glob("context_*.json"):
            try:
                if file_path.stat().st_mtime < cutoff_time.timestamp():
                    file_path.unlink()
                    self.logger.info(f"Cleaned up old context file: {file_path}")
            except Exception as e:
                self.logger.warning(f"Error cleaning up file {file_path}: {e}")


class EnhancedConversationManager:
    """增强对话管理器 - 完整对话生命周期管理"""

    def __init__(self, db_path: str = "conversations.db", use_mem0: bool = False):
        self.memory_manager = ConversationMemoryManager(use_mem0=use_mem0)
        self.database = ConversationDatabase(db_path=db_path)
        self.file_manager = ConversationFileManager()
        self.logger = logger.getChild('EnhancedManager')

        # 活跃对话缓存
        self.active_conversations: Dict[str, ConversationContext] = {}

    async def create_or_get_conversation(self, conversation_id: str, user_id: str,
                                       project_name: str = "") -> ConversationContext:
        """创建或获取对话上下文"""
        # 先从缓存查找
        if conversation_id in self.active_conversations:
            return self.active_conversations[conversation_id]

        # 从数据库查找
        context = self.database.get_conversation(conversation_id)

        if context is None:
            # 创建新对话
            self.database.create_conversation(conversation_id, user_id, project_name)
            context = ConversationContext(
                conversation_id=conversation_id,
                user_id=user_id,
                project_name=project_name
            )

            self.logger.info(f"Created new conversation: {conversation_id}")
        else:
            self.logger.info(f"Retrieved existing conversation: {conversation_id}")

        # 加载上下文文件数据
        file_context = self.file_manager.load_context_file(conversation_id)
        if file_context:
            # 合并文件中的上下文数据
            context.last_node_execution.update(file_context.get('last_node_execution', {}))
            context.global_style.update(file_context.get('global_style', {}))

        # 缓存活跃对话
        self.active_conversations[conversation_id] = context

        return context

    async def add_message(self, conversation_id: str, message_id: str, user_input: str,
                         node_results: Dict[str, Any] = None) -> ConversationContext:
        """添加消息到对话"""
        context = await self.create_or_get_conversation(conversation_id, "user")

        # 更新消息计数
        context.message_count += 1
        context.updated_at = datetime.now()

        # 存储消息到数据库
        self.database.add_message(
            conversation_id, message_id, context.message_count,
            user_input, node_results=node_results
        )

        # 存储到记忆管理器
        await self.memory_manager.store_memory(
            conversation_id, f"message_{context.message_count}", {
                'user_input': user_input,
                'node_results': node_results,
                'timestamp': datetime.now().isoformat()
            }
        )

        # 更新数据库
        self.database.update_conversation(context)

        # 保存上下文文件
        context_data = {
            'last_node_execution': context.last_node_execution,
            'global_style': context.global_style,
            'user_assets': context.user_assets,
            'generation_history': context.generation_history
        }
        self.file_manager.save_context_file(conversation_id, context_data)

        return context

    async def update_node_execution(self, conversation_id: str, node_id: str,
                                   node_result: Dict[str, Any]):
        """更新节点执行结果"""
        context = self.active_conversations.get(conversation_id)
        if context:
            context.last_node_execution[node_id] = {
                'result': node_result,
                'executed_at': datetime.now().isoformat()
            }

            # 存储到记忆
            await self.memory_manager.store_memory(
                conversation_id, f"node_{node_id}", node_result
            )

            self.logger.debug(f"Updated node execution: {conversation_id}/{node_id}")

    async def set_global_style(self, conversation_id: str, style_data: Dict[str, Any]):
        """设置全局风格"""
        context = self.active_conversations.get(conversation_id)
        if context:
            context.global_style.update(style_data)
            await self.memory_manager.store_memory(
                conversation_id, "global_style", style_data
            )

            self.logger.info(f"Updated global style for: {conversation_id}")

    async def add_user_asset(self, conversation_id: str, asset_data: Dict[str, Any]):
        """添加用户素材"""
        context = self.active_conversations.get(conversation_id)
        if context:
            context.user_assets.append(asset_data)
            await self.memory_manager.store_memory(
                conversation_id, f"asset_{len(context.user_assets)}", asset_data
            )

            self.logger.info(f"Added user asset to: {conversation_id}")

    async def get_conversation_history(self, conversation_id: str, limit: int = 10) -> List[Dict]:
        """获取对话历史"""
        return self.database.get_conversation_messages(conversation_id, limit)

    async def close_conversation(self, conversation_id: str):
        """关闭对话"""
        if conversation_id in self.active_conversations:
            context = self.active_conversations[conversation_id]
            context.status = ConversationStatus.COMPLETED
            context.updated_at = datetime.now()

            # 更新数据库
            self.database.update_conversation(context)

            # 保存最终上下文文件
            context_data = {
                'last_node_execution': context.last_node_execution,
                'global_style': context.global_style,
                'user_assets': context.user_assets,
                'generation_history': context.generation_history
            }
            self.file_manager.save_context_file(conversation_id, context_data)

            # 从缓存移除
            del self.active_conversations[conversation_id]

            self.logger.info(f"Closed conversation: {conversation_id}")


# 创建全局实例
enhanced_conversation_manager = EnhancedConversationManager()