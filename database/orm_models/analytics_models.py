"""
Analytics Models

Models for tracking user activity, system metrics, usage statistics,
error logging, performance monitoring, and audit trails.
"""

from sqlalchemy import Column, String, Boolean, Integer, DateTime, Text, Float, ForeignKey, Enum, JSON, Index, BigInteger
from sqlalchemy.orm import relationship
from sqlalchemy.dialects.postgresql import UUID
from datetime import datetime, timezone
from enum import Enum as PyEnum
from .base import EnhancedBaseModel

class ActivityType(PyEnum):
    LOGIN = "login"
    LOGOUT = "logout"
    CREATE_PROJECT = "create_project"
    CREATE_COMPOSITION = "create_composition"
    START_RENDER = "start_render"
    COMPLETE_RENDER = "complete_render"
    UPLOAD_FILE = "upload_file"
    DOWNLOAD_FILE = "download_file"

class ErrorSeverity(PyEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class UserActivity(EnhancedBaseModel):
    """User activity tracking"""
    __tablename__ = "user_activities"
    
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=False)
    
    # Activity details
    activity_type = Column(Enum(ActivityType), nullable=False)
    description = Column(Text, nullable=True)
    
    # Context
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    session_id = Column(String(255), nullable=True)
    
    # Related objects
    project_id = Column(UUID(as_uuid=True), nullable=True)
    composition_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Additional data
    activity_metadata = Column(JSON, default=dict, nullable=False)
    
    # Relationships
    user = relationship("User", back_populates="activities")
    
    __table_args__ = (Index('idx_activity_user_type', 'user_id', 'activity_type'),)

class SystemMetrics(EnhancedBaseModel):
    """System performance metrics"""
    __tablename__ = "system_metrics"
    
    # Metric identification
    metric_name = Column(String(100), nullable=False)
    metric_type = Column(String(50), nullable=False)  # counter, gauge, histogram
    
    # Values
    value = Column(Float, nullable=False)
    count = Column(Integer, default=1, nullable=False)
    
    # Context
    hostname = Column(String(255), nullable=True)
    service_name = Column(String(100), nullable=True)
    
    # Tags for grouping
    tags = Column(JSON, default=dict, nullable=False)
    
    __table_args__ = (Index('idx_metrics_name_time', 'metric_name', 'created_at'),)

class UsageStatistics(EnhancedBaseModel):
    """Usage statistics aggregation"""
    __tablename__ = "usage_statistics"
    
    # Time period
    date = Column(DateTime(timezone=True), nullable=False)
    period_type = Column(String(20), nullable=False)  # hour, day, week, month
    
    # Metrics
    active_users = Column(Integer, default=0, nullable=False)
    new_users = Column(Integer, default=0, nullable=False)
    projects_created = Column(Integer, default=0, nullable=False)
    videos_rendered = Column(Integer, default=0, nullable=False)
    total_render_time = Column(Float, default=0.0, nullable=False)
    
    # Storage metrics
    total_storage_used = Column(BigInteger, default=0, nullable=False)
    files_uploaded = Column(Integer, default=0, nullable=False)
    
    __table_args__ = (Index('idx_usage_date_period', 'date', 'period_type'),)

class ErrorLog(EnhancedBaseModel):
    """Error logging and tracking"""
    __tablename__ = "error_logs"
    
    # Error details
    error_type = Column(String(100), nullable=False)
    error_message = Column(Text, nullable=False)
    stack_trace = Column(Text, nullable=True)
    severity = Column(Enum(ErrorSeverity), default=ErrorSeverity.MEDIUM, nullable=False)
    
    # Context
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    request_id = Column(String(255), nullable=True)
    endpoint = Column(String(255), nullable=True)
    method = Column(String(10), nullable=True)
    
    # Additional context
    user_agent = Column(Text, nullable=True)
    ip_address = Column(String(45), nullable=True)
    
    # Error metadata
    context_data = Column(JSON, default=dict, nullable=False)
    
    # Resolution
    is_resolved = Column(Boolean, default=False, nullable=False)
    resolved_at = Column(DateTime(timezone=True), nullable=True)
    resolution_notes = Column(Text, nullable=True)
    
    __table_args__ = (Index('idx_error_type_severity', 'error_type', 'severity'),)

class PerformanceMetrics(EnhancedBaseModel):
    """Performance monitoring"""
    __tablename__ = "performance_metrics"
    
    # Request identification
    request_id = Column(String(255), nullable=False)
    endpoint = Column(String(255), nullable=False)
    method = Column(String(10), nullable=False)
    
    # Timing metrics
    response_time = Column(Float, nullable=False)  # milliseconds
    db_query_time = Column(Float, default=0.0, nullable=False)
    external_api_time = Column(Float, default=0.0, nullable=False)
    
    # Resource usage
    memory_usage = Column(Integer, nullable=True)  # MB
    cpu_usage = Column(Float, nullable=True)  # percentage
    
    # Request details
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    status_code = Column(Integer, nullable=False)
    
    __table_args__ = (Index('idx_performance_endpoint_time', 'endpoint', 'created_at'),)

class AuditLog(EnhancedBaseModel):
    """Audit trail for important actions"""
    __tablename__ = "audit_logs"
    
    # Action details
    action = Column(String(100), nullable=False)
    resource_type = Column(String(50), nullable=False)
    resource_id = Column(UUID(as_uuid=True), nullable=True)
    
    # Actor
    user_id = Column(UUID(as_uuid=True), ForeignKey('users.id'), nullable=True)
    user_email = Column(String(255), nullable=True)  # For deleted users
    
    # Changes
    old_values = Column(JSON, nullable=True)
    new_values = Column(JSON, nullable=True)
    
    # Context
    ip_address = Column(String(45), nullable=True)
    user_agent = Column(Text, nullable=True)
    
    __table_args__ = (
        Index('idx_audit_resource', 'resource_type', 'resource_id'),
        Index('idx_audit_user_action', 'user_id', 'action'),
    )