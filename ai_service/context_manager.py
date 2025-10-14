from datetime import datetime
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum

class ContextType(Enum):
    USER_PREFERENCE = "user_preference"
    SESSION_STATE = "session_state"
    CONVERSATION = "conversation"

class ContextPriority(Enum):
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"

@dataclass
class ConversationContext:
    session_id: str
    user_id: str
    context_data: Dict[str, Any] = field(default_factory=dict)

class ContextManager:
    def __init__(self):
        self.contexts = {}
        self.conversations = {}
    
    def create_session(self, session_id: str, user_id: str, metadata: Optional[Dict] = None) -> bool:
        self.conversations[session_id] = ConversationContext(session_id, user_id, metadata or {})
        return True
    
    def add_context(self, context_id: str, context_type: ContextType, content: Dict[str, Any], priority: ContextPriority, ttl: Optional[int] = None) -> bool:
        self.contexts[context_id] = {"type": context_type, "content": content}
        return True
    
    def update_conversation(self, session_id: str, role: str, content: str, metadata: Optional[Dict] = None):
        if session_id in self.conversations:
            if "history" not in self.conversations[session_id].context_data:
                self.conversations[session_id].context_data["history"] = []
            self.conversations[session_id].context_data["history"].append({"role": role, "content": content})
    
    def get_conversation_history(self, session_id: str, limit: int = 20) -> List[Dict]:
        if session_id in self.conversations:
            return self.conversations[session_id].context_data.get("history", [])[-limit:]
        return []
    
    def get_relevant_contexts(self, query_context: Dict, max_results: int = 5) -> List[Dict]:
        return []
    
    def get_stats(self) -> Dict[str, Any]:
        return {"total_contexts": len(self.contexts), "total_sessions": len(self.conversations)}
    
    def reset(self):
        self.contexts.clear()
        self.conversations.clear()

_context_manager = None

def get_context_manager() -> ContextManager:
    global _context_manager
    if _context_manager is None:
        _context_manager = ContextManager()
    return _context_manager