"""
WebSocket API Endpoints

Real-time WebSocket endpoints for live notifications, progress updates,
and system monitoring.
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Depends, HTTPException
from pydantic import BaseModel

from analytics import get_metrics_collector, get_system_monitor
from task_queue.batch_processor import get_batch_processor
from auth import get_user_manager

router = APIRouter()
logger = logging.getLogger(__name__)


class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        # Active connections by connection ID
        self.connections: Dict[str, WebSocket] = {}
        
        # User-specific connections
        self.user_connections: Dict[str, List[str]] = {}
        
        # Topic subscriptions
        self.subscriptions: Dict[str, List[str]] = {}
        
        self.logger = logging.getLogger(__name__)
    
    async def connect(self, websocket: WebSocket, connection_id: str, user_id: Optional[str] = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        self.connections[connection_id] = websocket
        
        if user_id:
            if user_id not in self.user_connections:
                self.user_connections[user_id] = []
            self.user_connections[user_id].append(connection_id)
        
        self.logger.info(f"WebSocket connected: {connection_id} (user: {user_id})")
        
        # Send welcome message
        await self.send_to_connection(connection_id, {
            "type": "connection",
            "status": "connected",
            "connection_id": connection_id,
            "timestamp": datetime.now().isoformat()
        })
    
    async def disconnect(self, connection_id: str):
        """Remove a WebSocket connection"""
        if connection_id in self.connections:
            del self.connections[connection_id]
        
        # Remove from user connections
        for user_id, conn_list in self.user_connections.items():
            if connection_id in conn_list:
                conn_list.remove(connection_id)
                if not conn_list:
                    del self.user_connections[user_id]
                break
        
        # Remove from subscriptions
        for topic, conn_list in list(self.subscriptions.items()):
            if connection_id in conn_list:
                conn_list.remove(connection_id)
                if not conn_list:
                    del self.subscriptions[topic]
        
        self.logger.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_to_connection(self, connection_id: str, message: Dict[str, Any]):
        """Send message to a specific connection"""
        if connection_id in self.connections:
            try:
                await self.connections[connection_id].send_text(json.dumps(message))
                return True
            except Exception as e:
                self.logger.error(f"Failed to send to {connection_id}: {e}")
                await self.disconnect(connection_id)
                return False
        return False
    
    async def send_to_user(self, user_id: str, message: Dict[str, Any]):
        """Send message to all connections for a user"""
        if user_id in self.user_connections:
            sent_count = 0
            for connection_id in self.user_connections[user_id].copy():
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
            return sent_count
        return 0
    
    async def subscribe_to_topic(self, connection_id: str, topic: str):
        """Subscribe a connection to a topic"""
        if topic not in self.subscriptions:
            self.subscriptions[topic] = []
        
        if connection_id not in self.subscriptions[topic]:
            self.subscriptions[topic].append(connection_id)
            self.logger.info(f"Connection {connection_id} subscribed to {topic}")
    
    async def unsubscribe_from_topic(self, connection_id: str, topic: str):
        """Unsubscribe a connection from a topic"""
        if topic in self.subscriptions and connection_id in self.subscriptions[topic]:
            self.subscriptions[topic].remove(connection_id)
            if not self.subscriptions[topic]:
                del self.subscriptions[topic]
            self.logger.info(f"Connection {connection_id} unsubscribed from {topic}")
    
    async def broadcast_to_topic(self, topic: str, message: Dict[str, Any]):
        """Broadcast message to all subscribers of a topic"""
        if topic in self.subscriptions:
            sent_count = 0
            for connection_id in self.subscriptions[topic].copy():
                if await self.send_to_connection(connection_id, message):
                    sent_count += 1
            return sent_count
        return 0
    
    async def broadcast_to_all(self, message: Dict[str, Any]):
        """Broadcast message to all connections"""
        sent_count = 0
        for connection_id in list(self.connections.keys()):
            if await self.send_to_connection(connection_id, message):
                sent_count += 1
        return sent_count


# Global connection manager
connection_manager = ConnectionManager()


class NotificationService:
    """Service for sending various types of notifications"""
    
    def __init__(self, connection_manager: ConnectionManager):
        self.connection_manager = connection_manager
        self.logger = logging.getLogger(__name__)
    
    async def notify_video_generation_progress(self, user_id: str, task_id: str, progress: float, message: str):
        """Notify about video generation progress"""
        notification = {
            "type": "video_generation",
            "event": "progress",
            "task_id": task_id,
            "progress": progress,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.connection_manager.send_to_user(user_id, notification)
        await self.connection_manager.broadcast_to_topic("video_progress", notification)
    
    async def notify_video_generation_complete(self, user_id: str, task_id: str, result: Dict[str, Any]):
        """Notify about video generation completion"""
        notification = {
            "type": "video_generation",
            "event": "complete",
            "task_id": task_id,
            "result": result,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.connection_manager.send_to_user(user_id, notification)
        await self.connection_manager.broadcast_to_topic("video_complete", notification)
    
    async def notify_batch_job_progress(self, user_id: str, job_id: str, progress: Dict[str, Any]):
        """Notify about batch job progress"""
        notification = {
            "type": "batch_job",
            "event": "progress",
            "job_id": job_id,
            "progress": progress,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.connection_manager.send_to_user(user_id, notification)
        await self.connection_manager.broadcast_to_topic("batch_progress", notification)
    
    async def notify_export_progress(self, user_id: str, export_id: str, progress: float, status: str):
        """Notify about export progress"""
        notification = {
            "type": "export",
            "event": "progress",
            "export_id": export_id,
            "progress": progress,
            "status": status,
            "timestamp": datetime.now().isoformat()
        }
        
        await self.connection_manager.send_to_user(user_id, notification)
        await self.connection_manager.broadcast_to_topic("export_progress", notification)
    
    async def notify_system_alert(self, level: str, message: str, details: Optional[Dict[str, Any]] = None):
        """Send system alert notification"""
        notification = {
            "type": "system_alert",
            "level": level,
            "message": message,
            "details": details or {},
            "timestamp": datetime.now().isoformat()
        }
        
        await self.connection_manager.broadcast_to_topic("system_alerts", notification)
        
        # Also send to admin users if available
        await self.connection_manager.broadcast_to_topic("admin", notification)
    
    async def send_metric_update(self, metric_name: str, value: float, tags: Optional[Dict[str, str]] = None):
        """Send real-time metric update"""
        notification = {
            "type": "metric_update",
            "metric_name": metric_name,
            "value": value,
            "tags": tags or {},
            "timestamp": datetime.now().isoformat()
        }
        
        await self.connection_manager.broadcast_to_topic("metrics", notification)


# Global notification service
notification_service = NotificationService(connection_manager)


@router.websocket("/ws/{connection_id}")
async def websocket_endpoint(websocket: WebSocket, connection_id: str):
    """Main WebSocket endpoint"""
    await connection_manager.connect(websocket, connection_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_websocket_message(connection_id, message)
            except json.JSONDecodeError:
                await connection_manager.send_to_connection(connection_id, {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                await connection_manager.send_to_connection(connection_id, {
                    "type": "error", 
                    "message": f"Error processing message: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        await connection_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"WebSocket error for {connection_id}: {e}")
        await connection_manager.disconnect(connection_id)


@router.websocket("/ws/user/{user_id}/{connection_id}")
async def authenticated_websocket_endpoint(websocket: WebSocket, user_id: str, connection_id: str):
    """Authenticated WebSocket endpoint for specific users"""
    await connection_manager.connect(websocket, connection_id, user_id)
    
    try:
        while True:
            data = await websocket.receive_text()
            
            try:
                message = json.loads(data)
                await handle_authenticated_message(connection_id, user_id, message)
            except json.JSONDecodeError:
                await connection_manager.send_to_connection(connection_id, {
                    "type": "error",
                    "message": "Invalid JSON format",
                    "timestamp": datetime.now().isoformat()
                })
            except Exception as e:
                await connection_manager.send_to_connection(connection_id, {
                    "type": "error",
                    "message": f"Error processing message: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
                
    except WebSocketDisconnect:
        await connection_manager.disconnect(connection_id)
    except Exception as e:
        logger.error(f"Authenticated WebSocket error for {connection_id}: {e}")
        await connection_manager.disconnect(connection_id)


async def handle_websocket_message(connection_id: str, message: Dict[str, Any]):
    """Handle incoming WebSocket messages"""
    message_type = message.get("type")
    
    if message_type == "subscribe":
        topic = message.get("topic")
        if topic:
            await connection_manager.subscribe_to_topic(connection_id, topic)
            await connection_manager.send_to_connection(connection_id, {
                "type": "subscription",
                "status": "subscribed",
                "topic": topic,
                "timestamp": datetime.now().isoformat()
            })
    
    elif message_type == "unsubscribe":
        topic = message.get("topic")
        if topic:
            await connection_manager.unsubscribe_from_topic(connection_id, topic)
            await connection_manager.send_to_connection(connection_id, {
                "type": "subscription",
                "status": "unsubscribed",
                "topic": topic,
                "timestamp": datetime.now().isoformat()
            })
    
    elif message_type == "ping":
        await connection_manager.send_to_connection(connection_id, {
            "type": "pong",
            "timestamp": datetime.now().isoformat()
        })
    
    elif message_type == "get_stats":
        stats = {
            "type": "stats",
            "connections": len(connection_manager.connections),
            "user_connections": len(connection_manager.user_connections),
            "subscriptions": {topic: len(connections) for topic, connections in connection_manager.subscriptions.items()},
            "timestamp": datetime.now().isoformat()
        }
        await connection_manager.send_to_connection(connection_id, stats)
    
    else:
        await connection_manager.send_to_connection(connection_id, {
            "type": "error",
            "message": f"Unknown message type: {message_type}",
            "timestamp": datetime.now().isoformat()
        })


async def handle_authenticated_message(connection_id: str, user_id: str, message: Dict[str, Any]):
    """Handle authenticated WebSocket messages"""
    message_type = message.get("type")
    
    # Handle all regular messages
    await handle_websocket_message(connection_id, message)
    
    # Handle user-specific messages
    if message_type == "get_user_progress":
        # Get user's active tasks/jobs
        batch_processor = get_batch_processor()
        jobs = await batch_processor.get_jobs_by_user(user_id)
        
        active_jobs = [
            {
                "id": job.id,
                "type": "batch_job",
                "status": job.status.value,
                "progress": job.completed_items / job.total_items * 100 if job.total_items > 0 else 0,
                "created_at": job.created_at.isoformat()
            }
            for job in jobs
            if job.status.value in ["pending", "processing"]
        ]
        
        await connection_manager.send_to_connection(connection_id, {
            "type": "user_progress",
            "user_id": user_id,
            "active_jobs": active_jobs,
            "timestamp": datetime.now().isoformat()
        })


# Background task for periodic updates
async def periodic_system_updates():
    """Send periodic system updates to subscribers"""
    while True:
        try:
            # System performance metrics
            system_monitor = get_system_monitor()
            performance_metrics = await system_monitor.get_performance_metrics()
            
            await notification_service.connection_manager.broadcast_to_topic("system_metrics", {
                "type": "system_metrics",
                "metrics": performance_metrics.dict(),
                "timestamp": datetime.now().isoformat()
            })
            
            # Sleep for 30 seconds
            await asyncio.sleep(30)
            
        except Exception as e:
            logger.error(f"Error in periodic system updates: {e}")
            await asyncio.sleep(30)


# Background task will be started when the app starts
# Not when the module is imported
# asyncio.create_task(periodic_system_updates())


# HTTP endpoints for WebSocket management

@router.get("/ws/stats")
async def get_websocket_stats():
    """Get WebSocket connection statistics"""
    return {
        "total_connections": len(connection_manager.connections),
        "user_connections": len(connection_manager.user_connections),
        "active_topics": list(connection_manager.subscriptions.keys()),
        "topic_subscribers": {
            topic: len(connections) 
            for topic, connections in connection_manager.subscriptions.items()
        },
        "timestamp": datetime.now().isoformat()
    }


@router.post("/ws/broadcast")
async def broadcast_message(
    topic: str,
    message: str,
    message_type: str = "broadcast"
):
    """Broadcast a message to all subscribers of a topic"""
    try:
        notification = {
            "type": message_type,
            "message": message,
            "topic": topic,
            "timestamp": datetime.now().isoformat()
        }
        
        sent_count = await connection_manager.broadcast_to_topic(topic, notification)
        
        return {
            "success": True,
            "message": f"Broadcast sent to {sent_count} connections",
            "topic": topic,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to broadcast message: {str(e)}"
        )


@router.post("/ws/notify/user/{user_id}")
async def notify_user(
    user_id: str,
    message: str,
    message_type: str = "notification"
):
    """Send a notification to a specific user"""
    try:
        notification = {
            "type": message_type,
            "message": message,
            "timestamp": datetime.now().isoformat()
        }
        
        sent_count = await connection_manager.send_to_user(user_id, notification)
        
        return {
            "success": True,
            "message": f"Notification sent to {sent_count} connections",
            "user_id": user_id,
            "timestamp": datetime.now()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to send notification: {str(e)}"
        )


# Export the notification service for use in other modules
__all__ = ["router", "notification_service", "connection_manager"]