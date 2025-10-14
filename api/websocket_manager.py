"""
WebSocket管理器 - 实时通信和消息推送
"""
from typing import Dict, List, Any, Optional, Set
import asyncio
import json
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
from fastapi import WebSocket


class MessageType(Enum):
    """消息类型"""
    SYSTEM = "system"
    PROGRESS = "progress"
    NOTIFICATION = "notification"
    ERROR = "error"


@dataclass
class WebSocketClient:
    """WebSocket客户端"""
    client_id: str
    websocket: WebSocket
    connected_at: datetime
    subscriptions: Set[str] = field(default_factory=set)


class WebSocketManager:
    """WebSocket连接管理器"""

    def __init__(self):
        self.connections: Dict[str, WebSocketClient] = {}
        self.topic_subscribers: Dict[str, Set[str]] = {}

    async def start(self):
        """启动WebSocket管理器"""
        print("🌐 WebSocket manager started")

    async def stop(self):
        """停止WebSocket管理器"""
        # 关闭所有连接
        for client in list(self.connections.values()):
            await self.disconnect(client.client_id)
        print("🛑 WebSocket manager stopped")

    async def connect(self, websocket: WebSocket, client_id: str):
        """建立WebSocket连接"""
        await websocket.accept()

        client = WebSocketClient(
            client_id=client_id,
            websocket=websocket,
            connected_at=datetime.now()
        )

        self.connections[client_id] = client

        # 发送欢迎消息
        await self.send_to_client(client_id, MessageType.SYSTEM, {
            "message": "Connected successfully",
            "client_id": client_id
        })

        print(f"🔗 WebSocket client connected: {client_id}")

    async def disconnect(self, client_id: str):
        """断开WebSocket连接"""
        if client_id in self.connections:
            client = self.connections[client_id]

            # 取消所有订阅
            for topic in list(client.subscriptions):
                await self.unsubscribe(client_id, topic)

            # 关闭连接
            try:
                await client.websocket.close()
            except Exception:
                pass

            del self.connections[client_id]
            print(f"❌ WebSocket client disconnected: {client_id}")

    async def send_to_client(self, client_id: str, message_type: MessageType, data: Dict[str, Any]):
        """发送消息给指定客户端"""
        if client_id not in self.connections:
            return False

        client = self.connections[client_id]

        try:
            await client.websocket.send_text(json.dumps({
                "type": message_type.value,
                "data": data,
                "timestamp": datetime.now().isoformat()
            }))
            return True

        except Exception as e:
            print(f"❌ Failed to send message to {client_id}: {e}")
            await self.disconnect(client_id)
            return False

    async def subscribe(self, client_id: str, topic: str):
        """订阅主题"""
        if client_id not in self.connections:
            return False

        client = self.connections[client_id]
        client.subscriptions.add(topic)

        if topic not in self.topic_subscribers:
            self.topic_subscribers[topic] = set()
        self.topic_subscribers[topic].add(client_id)

        return True

    async def unsubscribe(self, client_id: str, topic: str):
        """取消订阅主题"""
        if client_id not in self.connections:
            return False

        client = self.connections[client_id]
        client.subscriptions.discard(topic)

        if topic in self.topic_subscribers:
            self.topic_subscribers[topic].discard(client_id)
            if not self.topic_subscribers[topic]:
                del self.topic_subscribers[topic]

        return True

    async def publish_to_topic(self, topic: str, message_type: MessageType, data: Dict[str, Any]):
        """发布消息到主题"""
        if topic not in self.topic_subscribers:
            return 0

        sent_count = 0
        subscribers = list(self.topic_subscribers[topic])

        for client_id in subscribers:
            success = await self.send_to_client(client_id, message_type, data)
            if success:
                sent_count += 1

        return sent_count

    async def handle_message(self, client_id: str, message: str):
        """处理客户端消息"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            payload = data.get("data", {})

            if message_type == "subscribe":
                topic = payload.get("topic")
                if topic:
                    await self.subscribe(client_id, topic)

            elif message_type == "unsubscribe":
                topic = payload.get("topic")
                if topic:
                    await self.unsubscribe(client_id, topic)

        except Exception as e:
            print(f"❌ Error handling message from {client_id}: {e}")

    async def send_progress_update(self, task_id: str, progress: float, message: str = ""):
        """发送进度更新"""
        topic = f"task_progress_{task_id}"
        await self.publish_to_topic(topic, MessageType.PROGRESS, {
            "task_id": task_id,
            "progress": progress,
            "message": message
        })