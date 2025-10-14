"""
速率限制器 - API请求频率控制和配额管理
"""
from typing import Dict, List, Any, Optional, Tuple
import time
from datetime import datetime, timedelta
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum


class LimitType(Enum):
    """限制类型"""
    PER_MINUTE = "per_minute"
    PER_HOUR = "per_hour"
    PER_DAY = "per_day"


@dataclass
class RateLimit:
    """速率限制配置"""
    limit_type: LimitType
    max_requests: int
    window_seconds: int


class RateLimiter:
    """速率限制器"""

    def __init__(self):
        # 内存存储
        self.request_records: Dict[str, deque] = defaultdict(lambda: deque(maxlen=1000))

        # 统计信息
        self.stats = {
            'total_requests': 0,
            'blocked_requests': 0
        }

        # 默认限制规则
        self.limits = {
            'guest': [
                RateLimit(LimitType.PER_MINUTE, 10, 60),
                RateLimit(LimitType.PER_HOUR, 100, 3600)
            ],
            'user': [
                RateLimit(LimitType.PER_MINUTE, 60, 60),
                RateLimit(LimitType.PER_HOUR, 1000, 3600)
            ],
            'admin': [
                RateLimit(LimitType.PER_MINUTE, 1000, 60),
                RateLimit(LimitType.PER_HOUR, 10000, 3600)
            ]
        }

    async def check_rate_limit(self, client_id: str, user_role: str = "guest") -> Tuple[bool, Dict[str, Any]]:
        """检查速率限制"""
        current_time = time.time()
        self.stats['total_requests'] += 1

        # 获取用户角色对应的限制
        limits = self.limits.get(user_role, self.limits['guest'])

        for limit in limits:
            key = f"{client_id}:{limit.limit_type.value}"
            window_start = current_time - limit.window_seconds

            # 清理过期记录
            while self.request_records[key] and self.request_records[key][0] < window_start:
                self.request_records[key].popleft()

            # 检查是否超过限制
            if len(self.request_records[key]) >= limit.max_requests:
                self.stats['blocked_requests'] += 1
                return False, {
                    'allowed': False,
                    'limit_type': limit.limit_type.value,
                    'max_requests': limit.max_requests,
                    'retry_after': 60
                }

        # 记录请求
        for limit in limits:
            key = f"{client_id}:{limit.limit_type.value}"
            self.request_records[key].append(current_time)

        return True, {'allowed': True}

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return self.stats.copy()