"""
用户行为跟踪器 - 分析用户使用模式和偏好
"""
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from enum import Enum
import json
import hashlib

from database.database_manager import DatabaseManager
from cache.cache_manager import CacheManager


class EventType(Enum):
    """事件类型"""
    PAGE_VIEW = "page_view"
    FEATURE_USE = "feature_use"
    USER_ACTION = "user_action"
    ERROR_ENCOUNTER = "error_encounter"
    SESSION_START = "session_start"
    SESSION_END = "session_end"
    CONVERSION = "conversion"


class UserSegment(Enum):
    """用户细分"""
    NEW_USER = "new_user"
    ACTIVE_USER = "active_user"
    POWER_USER = "power_user"
    INACTIVE_USER = "inactive_user"
    CHURNED_USER = "churned_user"


@dataclass
class UserEvent:
    """用户事件"""
    event_id: str
    user_id: Optional[str]
    session_id: str
    event_type: EventType
    event_name: str
    properties: Dict[str, Any]
    timestamp: datetime
    page_url: Optional[str] = None
    referrer: Optional[str] = None
    user_agent: Optional[str] = None


@dataclass
class UserSession:
    """用户会话"""
    session_id: str
    user_id: Optional[str]
    start_time: datetime
    end_time: Optional[datetime]
    page_views: int = 0
    events: List[UserEvent] = field(default_factory=list)
    duration_seconds: float = 0
    bounce: bool = False
    conversion: bool = False


@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    segment: UserSegment
    first_seen: datetime
    last_seen: datetime
    total_sessions: int
    total_page_views: int
    total_events: int
    favorite_features: List[str]
    usage_patterns: Dict[str, Any]
    preferences: Dict[str, Any]
    satisfaction_score: float


@dataclass
class BehaviorPattern:
    """行为模式"""
    pattern_id: str
    pattern_name: str
    description: str
    frequency: int
    user_count: int
    conversion_rate: float
    steps: List[str]
    metrics: Dict[str, Any]


class UserBehaviorTracker:
    """用户行为跟踪器"""

    def __init__(self, database_manager: Optional[DatabaseManager] = None,
                 cache_manager: Optional[CacheManager] = None):
        self.database_manager = database_manager
        self.cache_manager = cache_manager

        # 内存存储（生产环境应使用数据库）
        self.events: deque = deque(maxlen=10000)
        self.sessions: Dict[str, UserSession] = {}
        self.user_profiles: Dict[str, UserProfile] = {}

        # 实时统计
        self.real_time_stats = {
            'active_sessions': 0,
            'events_per_minute': deque(maxlen=60),
            'popular_pages': defaultdict(int),
            'popular_features': defaultdict(int),
            'error_counts': defaultdict(int)
        }

        # 分析任务
        self.analysis_task: Optional[asyncio.Task] = None
        self.running = False

    async def start_tracking(self):
        """开始行为跟踪"""
        if self.running:
            return

        self.running = True
        self.analysis_task = asyncio.create_task(self._analysis_loop())
        print("🔍 User behavior tracking started")

    async def stop_tracking(self):
        """停止行为跟踪"""
        self.running = False
        if self.analysis_task:
            self.analysis_task.cancel()
            try:
                await self.analysis_task
            except asyncio.CancelledError:
                pass
        print("🛑 User behavior tracking stopped")

    async def track_event(self, user_id: Optional[str], session_id: str,
                         event_type: EventType, event_name: str,
                         properties: Dict[str, Any] = None,
                         page_url: Optional[str] = None,
                         referrer: Optional[str] = None,
                         user_agent: Optional[str] = None):
        """跟踪用户事件"""
        event_id = f"evt_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

        event = UserEvent(
            event_id=event_id,
            user_id=user_id,
            session_id=session_id,
            event_type=event_type,
            event_name=event_name,
            properties=properties or {},
            timestamp=datetime.now(),
            page_url=page_url,
            referrer=referrer,
            user_agent=user_agent
        )

        # 存储事件
        self.events.append(event)

        # 更新会话
        await self._update_session(event)

        # 更新用户画像
        if user_id:
            await self._update_user_profile(user_id, event)

        # 更新实时统计
        await self._update_real_time_stats(event)

        # 存储到数据库（如果可用）
        if self.database_manager:
            await self._store_event_to_db(event)

    async def _update_session(self, event: UserEvent):
        """更新会话信息"""
        session_id = event.session_id

        if session_id not in self.sessions:
            # 创建新会话
            self.sessions[session_id] = UserSession(
                session_id=session_id,
                user_id=event.user_id,
                start_time=event.timestamp
            )
            self.real_time_stats['active_sessions'] += 1

            # 记录会话开始事件
            if event.event_type != EventType.SESSION_START:
                await self.track_event(
                    event.user_id, session_id,
                    EventType.SESSION_START, "session_start"
                )

        session = self.sessions[session_id]
        session.events.append(event)

        # 更新会话统计
        if event.event_type == EventType.PAGE_VIEW:
            session.page_views += 1

        if event.event_type == EventType.CONVERSION:
            session.conversion = True

        # 更新会话时长
        session.duration_seconds = (event.timestamp - session.start_time).total_seconds()

        # 检查是否为跳出
        if session.page_views == 1 and session.duration_seconds < 10:
            session.bounce = True

    async def _update_user_profile(self, user_id: str, event: UserEvent):
        """更新用户画像"""
        if user_id not in self.user_profiles:
            # 创建新用户画像
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                segment=UserSegment.NEW_USER,
                first_seen=event.timestamp,
                last_seen=event.timestamp,
                total_sessions=0,
                total_page_views=0,
                total_events=0,
                favorite_features=[],
                usage_patterns={},
                preferences={},
                satisfaction_score=0.5
            )

        profile = self.user_profiles[user_id]
        profile.last_seen = event.timestamp
        profile.total_events += 1

        if event.event_type == EventType.PAGE_VIEW:
            profile.total_page_views += 1

        if event.event_type == EventType.FEATURE_USE:
            feature_name = event.properties.get('feature_name')
            if feature_name:
                if feature_name not in profile.usage_patterns:
                    profile.usage_patterns[feature_name] = 0
                profile.usage_patterns[feature_name] += 1

        # 重新计算用户细分
        profile.segment = self._calculate_user_segment(profile)

    def _calculate_user_segment(self, profile: UserProfile) -> UserSegment:
        """计算用户细分"""
        days_since_first = (datetime.now() - profile.first_seen).days
        days_since_last = (datetime.now() - profile.last_seen).days

        if days_since_first <= 7:
            return UserSegment.NEW_USER
        elif days_since_last > 30:
            return UserSegment.CHURNED_USER
        elif days_since_last > 7:
            return UserSegment.INACTIVE_USER
        elif profile.total_events > 100:
            return UserSegment.POWER_USER
        else:
            return UserSegment.ACTIVE_USER

    async def _update_real_time_stats(self, event: UserEvent):
        """更新实时统计"""
        # 事件计数
        current_minute = datetime.now().strftime('%Y-%m-%d %H:%M')
        self.real_time_stats['events_per_minute'].append((current_minute, 1))

        # 热门页面
        if event.page_url:
            self.real_time_stats['popular_pages'][event.page_url] += 1

        # 热门功能
        if event.event_type == EventType.FEATURE_USE:
            feature_name = event.properties.get('feature_name')
            if feature_name:
                self.real_time_stats['popular_features'][feature_name] += 1

        # 错误统计
        if event.event_type == EventType.ERROR_ENCOUNTER:
            error_type = event.properties.get('error_type', 'unknown')
            self.real_time_stats['error_counts'][error_type] += 1

    async def _store_event_to_db(self, event: UserEvent):
        """存储事件到数据库"""
        try:
            query = """
            INSERT INTO user_events (event_id, user_id, session_id, event_type,
                                   event_name, properties, timestamp, page_url)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = [
                event.event_id, event.user_id, event.session_id,
                event.event_type.value, event.event_name,
                json.dumps(event.properties), event.timestamp, event.page_url
            ]

            await self.database_manager.execute_query(query, params)
        except Exception as e:
            print(f"❌ Failed to store event to database: {e}")

    async def _analysis_loop(self):
        """分析循环"""
        while self.running:
            try:
                await self._perform_analysis()
                await asyncio.sleep(300)  # 5分钟分析一次
            except asyncio.CancelledError:
                break
            except Exception as e:
                print(f"❌ Analysis error: {e}")
                await asyncio.sleep(300)

    async def _perform_analysis(self):
        """执行分析"""
        # 清理过期会话
        await self._cleanup_expired_sessions()

        # 计算满意度评分
        await self._calculate_satisfaction_scores()

        # 识别行为模式
        await self._identify_behavior_patterns()

        print("📊 User behavior analysis completed")

    async def _cleanup_expired_sessions(self):
        """清理过期会话"""
        cutoff_time = datetime.now() - timedelta(hours=1)
        expired_sessions = []

        for session_id, session in self.sessions.items():
            if session.start_time < cutoff_time and not session.end_time:
                # 标记会话结束
                session.end_time = session.start_time + timedelta(seconds=session.duration_seconds)
                expired_sessions.append(session_id)

                # 更新用户画像中的会话数
                if session.user_id and session.user_id in self.user_profiles:
                    self.user_profiles[session.user_id].total_sessions += 1

        # 从活跃会话中移除
        for session_id in expired_sessions:
            if session_id in self.sessions:
                del self.sessions[session_id]
                self.real_time_stats['active_sessions'] -= 1

    async def _calculate_satisfaction_scores(self):
        """计算满意度评分"""
        for profile in self.user_profiles.values():
            score_factors = []

            # 使用频率因子
            days_active = (profile.last_seen - profile.first_seen).days + 1
            usage_frequency = profile.total_events / days_active
            frequency_score = min(1.0, usage_frequency / 10)  # 每天10个事件为满分
            score_factors.append(frequency_score)

            # 功能使用多样性
            feature_diversity = len(profile.usage_patterns)
            diversity_score = min(1.0, feature_diversity / 5)  # 使用5个功能为满分
            score_factors.append(diversity_score)

            # 会话质量（假设基于用户细分）
            segment_scores = {
                UserSegment.POWER_USER: 1.0,
                UserSegment.ACTIVE_USER: 0.8,
                UserSegment.NEW_USER: 0.6,
                UserSegment.INACTIVE_USER: 0.3,
                UserSegment.CHURNED_USER: 0.1
            }
            segment_score = segment_scores.get(profile.segment, 0.5)
            score_factors.append(segment_score)

            # 计算综合满意度
            profile.satisfaction_score = sum(score_factors) / len(score_factors)

    async def _identify_behavior_patterns(self):
        """识别行为模式"""
        # 这里可以实现复杂的行为模式识别算法
        # 暂时返回一些示例模式
        pass

    def get_user_analytics(self, time_range_hours: int = 24) -> Dict[str, Any]:
        """获取用户分析数据"""
        cutoff_time = datetime.now() - timedelta(hours=time_range_hours)

        # 过滤时间范围内的事件
        recent_events = [
            event for event in self.events
            if event.timestamp >= cutoff_time
        ]

        # 统计数据
        unique_users = len(set(event.user_id for event in recent_events if event.user_id))
        total_events = len(recent_events)
        page_views = len([e for e in recent_events if e.event_type == EventType.PAGE_VIEW])

        # 按事件类型分组
        events_by_type = defaultdict(int)
        for event in recent_events:
            events_by_type[event.event_type.value] += 1

        # 用户细分统计
        segment_counts = defaultdict(int)
        for profile in self.user_profiles.values():
            if profile.last_seen >= cutoff_time:
                segment_counts[profile.segment.value] += 1

        return {
            'time_range_hours': time_range_hours,
            'overview': {
                'unique_users': unique_users,
                'total_events': total_events,
                'page_views': page_views,
                'active_sessions': self.real_time_stats['active_sessions']
            },
            'events_by_type': dict(events_by_type),
            'user_segments': dict(segment_counts),
            'popular_pages': dict(list(self.real_time_stats['popular_pages'].items())[:10]),
            'popular_features': dict(list(self.real_time_stats['popular_features'].items())[:10]),
            'error_summary': dict(self.real_time_stats['error_counts']),
            'timestamp': datetime.now().isoformat()
        }

    def get_user_journey(self, user_id: str, session_limit: int = 5) -> Dict[str, Any]:
        """获取用户旅程"""
        if user_id not in self.user_profiles:
            return {'error': 'User not found'}

        profile = self.user_profiles[user_id]

        # 获取用户的最近会话
        user_sessions = [
            session for session in self.sessions.values()
            if session.user_id == user_id
        ]

        # 按时间排序
        user_sessions.sort(key=lambda s: s.start_time, reverse=True)
        recent_sessions = user_sessions[:session_limit]

        # 构建旅程数据
        journey_steps = []
        for session in recent_sessions:
            session_steps = []
            for event in session.events:
                step = {
                    'timestamp': event.timestamp.isoformat(),
                    'event_type': event.event_type.value,
                    'event_name': event.event_name,
                    'page_url': event.page_url,
                    'properties': event.properties
                }
                session_steps.append(step)

            journey_steps.append({
                'session_id': session.session_id,
                'start_time': session.start_time.isoformat(),
                'duration_seconds': session.duration_seconds,
                'page_views': session.page_views,
                'events': session_steps,
                'conversion': session.conversion,
                'bounce': session.bounce
            })

        return {
            'user_id': user_id,
            'profile': {
                'segment': profile.segment.value,
                'first_seen': profile.first_seen.isoformat(),
                'last_seen': profile.last_seen.isoformat(),
                'total_sessions': profile.total_sessions,
                'satisfaction_score': profile.satisfaction_score,
                'favorite_features': profile.favorite_features
            },
            'recent_sessions': journey_steps,
            'usage_patterns': profile.usage_patterns
        }

    def get_conversion_funnel(self, funnel_steps: List[str]) -> Dict[str, Any]:
        """获取转化漏斗分析"""
        # 统计每个步骤的用户数
        step_users = {}
        for step in funnel_steps:
            step_users[step] = set()

        # 分析用户事件
        for event in self.events:
            if event.user_id:
                for step in funnel_steps:
                    if step.lower() in event.event_name.lower():
                        step_users[step].add(event.user_id)

        # 计算转化率
        funnel_data = []
        total_users = len(step_users.get(funnel_steps[0], set())) if funnel_steps else 0

        for i, step in enumerate(funnel_steps):
            step_user_count = len(step_users.get(step, set()))
            conversion_rate = (step_user_count / total_users * 100) if total_users > 0 else 0

            funnel_data.append({
                'step': step,
                'users': step_user_count,
                'conversion_rate': round(conversion_rate, 2)
            })

            # 更新total_users为当前步骤的用户数（用于下一步计算）
            if i == 0:
                continue

        return {
            'funnel_steps': funnel_data,
            'overall_conversion_rate': round(
                (len(step_users.get(funnel_steps[-1], set())) / total_users * 100)
                if total_users > 0 and funnel_steps else 0, 2
            ),
            'total_users': total_users
        }

    def get_cohort_analysis(self, period_days: int = 7) -> Dict[str, Any]:
        """获取队列分析"""
        # 按注册时间分组用户
        cohorts = defaultdict(list)

        for profile in self.user_profiles.values():
            cohort_key = profile.first_seen.strftime('%Y-%m-%d')
            cohorts[cohort_key].append(profile)

        # 计算保留率
        cohort_data = []
        for cohort_date, users in cohorts.items():
            if len(users) < 5:  # 跳过用户数太少的队列
                continue

            cohort_start = datetime.strptime(cohort_date, '%Y-%m-%d')
            retention_data = {'cohort_date': cohort_date, 'initial_users': len(users)}

            # 计算各个时期的保留率
            for period in range(1, 13):  # 12个周期
                period_start = cohort_start + timedelta(days=period * period_days)
                period_end = period_start + timedelta(days=period_days)

                retained_users = len([
                    user for user in users
                    if user.last_seen >= period_start and user.last_seen < period_end
                ])

                retention_rate = (retained_users / len(users) * 100) if users else 0
                retention_data[f'period_{period}'] = {
                    'retained_users': retained_users,
                    'retention_rate': round(retention_rate, 2)
                }

            cohort_data.append(retention_data)

        return {
            'period_days': period_days,
            'cohorts': cohort_data,
            'analysis_date': datetime.now().isoformat()
        }

    def export_user_data(self, user_id: str) -> Dict[str, Any]:
        """导出用户数据（用于GDPR合规）"""
        if user_id not in self.user_profiles:
            return {'error': 'User not found'}

        profile = self.user_profiles[user_id]

        # 收集用户相关的所有事件
        user_events = [
            {
                'event_id': event.event_id,
                'timestamp': event.timestamp.isoformat(),
                'event_type': event.event_type.value,
                'event_name': event.event_name,
                'properties': event.properties,
                'page_url': event.page_url
            }
            for event in self.events if event.user_id == user_id
        ]

        # 收集用户会话
        user_sessions = [
            {
                'session_id': session.session_id,
                'start_time': session.start_time.isoformat(),
                'end_time': session.end_time.isoformat() if session.end_time else None,
                'duration_seconds': session.duration_seconds,
                'page_views': session.page_views,
                'conversion': session.conversion,
                'bounce': session.bounce
            }
            for session in self.sessions.values() if session.user_id == user_id
        ]

        return {
            'user_id': user_id,
            'profile': {
                'segment': profile.segment.value,
                'first_seen': profile.first_seen.isoformat(),
                'last_seen': profile.last_seen.isoformat(),
                'total_sessions': profile.total_sessions,
                'total_page_views': profile.total_page_views,
                'total_events': profile.total_events,
                'satisfaction_score': profile.satisfaction_score,
                'usage_patterns': profile.usage_patterns,
                'preferences': profile.preferences
            },
            'events': user_events,
            'sessions': user_sessions,
            'exported_at': datetime.now().isoformat()
        }

    async def delete_user_data(self, user_id: str) -> bool:
        """删除用户数据（用于GDPR合规）"""
        try:
            # 删除用户画像
            if user_id in self.user_profiles:
                del self.user_profiles[user_id]

            # 删除用户会话
            sessions_to_delete = [
                session_id for session_id, session in self.sessions.items()
                if session.user_id == user_id
            ]
            for session_id in sessions_to_delete:
                del self.sessions[session_id]

            # 删除用户事件（将user_id设为None以保持统计数据）
            for event in self.events:
                if event.user_id == user_id:
                    event.user_id = None

            # 从数据库删除（如果可用）
            if self.database_manager:
                await self.database_manager.execute_query(
                    "DELETE FROM user_events WHERE user_id = %s",
                    [user_id]
                )

            print(f"✅ User data deleted: {user_id}")
            return True

        except Exception as e:
            print(f"❌ Failed to delete user data: {e}")
            return False