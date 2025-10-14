"""
Error Handler

Centralized error handling with logging, alerts, and recovery strategies.
"""

import traceback
import uuid
from datetime import datetime
from typing import Optional, Dict, Any, Callable, List
from enum import Enum
import logging
import asyncio
from config import Settings


class ErrorSeverity(str, Enum):
    """Error severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ErrorCategory(str, Enum):
    """Error categories"""
    SYSTEM = "system"
    API = "api"
    DATABASE = "database"
    NETWORK = "network"
    AI_SERVICE = "ai_service"
    RENDERING = "rendering"
    MATERIAL_PROVIDER = "material_provider"
    VALIDATION = "validation"
    CONFIGURATION = "configuration"
    USER_INPUT = "user_input"


class ErrorRecord:
    """Error record structure"""
    
    def __init__(
        self,
        error_id: str,
        exception: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity,
        context: Dict[str, Any] = None,
        user_message: str = None,
        recovery_action: str = None
    ):
        self.error_id = error_id
        self.exception = exception
        self.category = category
        self.severity = severity
        self.context = context or {}
        self.user_message = user_message
        self.recovery_action = recovery_action
        self.timestamp = datetime.utcnow()
        self.traceback = traceback.format_exc()
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "error_id": self.error_id,
            "exception_type": type(self.exception).__name__,
            "exception_message": str(self.exception),
            "category": self.category.value,
            "severity": self.severity.value,
            "context": self.context,
            "user_message": self.user_message,
            "recovery_action": self.recovery_action,
            "timestamp": self.timestamp.isoformat(),
            "traceback": self.traceback
        }


class ErrorHandler:
    """Centralized error handler"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.logger = logging.getLogger(__name__)
        self.error_callbacks: Dict[ErrorCategory, List[Callable]] = {}
        self.error_history: List[ErrorRecord] = []
        self.max_history_size = 1000
        
    def register_error_callback(
        self, 
        category: ErrorCategory, 
        callback: Callable[[ErrorRecord], None]
    ):
        """Register callback for specific error category"""
        if category not in self.error_callbacks:
            self.error_callbacks[category] = []
        self.error_callbacks[category].append(callback)
        
    async def handle_error(
        self,
        exception: Exception,
        category: ErrorCategory,
        severity: ErrorSeverity = ErrorSeverity.MEDIUM,
        context: Dict[str, Any] = None,
        user_message: str = None,
        recovery_action: str = None,
        task_id: str = None
    ) -> str:
        """Handle error with logging and recovery"""
        
        error_id = str(uuid.uuid4())
        error_record = ErrorRecord(
            error_id=error_id,
            exception=exception,
            category=category,
            severity=severity,
            context=context,
            user_message=user_message,
            recovery_action=recovery_action
        )
        
        # Add to history
        self._add_to_history(error_record)
        
        # Log error
        await self._log_error(error_record)
        
        # Execute callbacks
        await self._execute_callbacks(error_record)
        
        # Send alerts for critical errors
        if severity == ErrorSeverity.CRITICAL:
            await self._send_alert(error_record)
            
        # Update task status if task_id provided
        if task_id:
            await self._update_task_error(task_id, error_record)
            
        return error_id
        
    def _add_to_history(self, error_record: ErrorRecord):
        """Add error to history with size management"""
        self.error_history.append(error_record)
        if len(self.error_history) > self.max_history_size:
            self.error_history = self.error_history[-self.max_history_size:]
            
    async def _log_error(self, error_record: ErrorRecord):
        """Log error with appropriate level"""
        
        log_data = error_record.to_dict()
        log_message = f"[{error_record.error_id}] {error_record.exception}"
        
        if error_record.severity == ErrorSeverity.CRITICAL:
            self.logger.critical(log_message, extra=log_data)
        elif error_record.severity == ErrorSeverity.HIGH:
            self.logger.error(log_message, extra=log_data)
        elif error_record.severity == ErrorSeverity.MEDIUM:
            self.logger.warning(log_message, extra=log_data)
        else:
            self.logger.info(log_message, extra=log_data)
            
    async def _execute_callbacks(self, error_record: ErrorRecord):
        """Execute registered callbacks for error category"""
        
        callbacks = self.error_callbacks.get(error_record.category, [])
        for callback in callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(error_record)
                else:
                    callback(error_record)
            except Exception as e:
                self.logger.error(f"Error in callback execution: {e}")
                
    async def _send_alert(self, error_record: ErrorRecord):
        """Send alert for critical errors"""
        
        # In production, this would integrate with alerting systems
        # like Slack, email, PagerDuty, etc.
        alert_message = f"""
        🚨 CRITICAL ERROR ALERT
        
        Error ID: {error_record.error_id}
        Category: {error_record.category.value}
        Exception: {error_record.exception}
        Time: {error_record.timestamp}
        
        Context: {error_record.context}
        Recovery Action: {error_record.recovery_action}
        """
        
        self.logger.critical(f"ALERT: {alert_message}")
        
        # TODO: Integrate with external alerting services
        
    async def _update_task_error(self, task_id: str, error_record: ErrorRecord):
        """Update task with error information"""
        
        try:
            from database.service_manager import DatabaseServiceManager
            from database.models import TaskStatus
            from database import get_session
            
            # This would need proper session management in production
            # For now, just log the error
            self.logger.info(f"Task {task_id} error: {error_record.error_id}")
            
        except Exception as e:
            self.logger.error(f"Failed to update task error: {e}")
            
    # Error recovery strategies
    async def retry_with_backoff(
        self,
        func: Callable,
        max_retries: int = 3,
        backoff_factor: float = 2.0,
        exception_types: tuple = (Exception,)
    ):
        """Retry function with exponential backoff"""
        
        for attempt in range(max_retries + 1):
            try:
                if asyncio.iscoroutinefunction(func):
                    return await func()
                else:
                    return func()
            except exception_types as e:
                if attempt == max_retries:
                    await self.handle_error(
                        e,
                        ErrorCategory.SYSTEM,
                        ErrorSeverity.HIGH,
                        context={"max_retries_exceeded": True, "attempts": attempt + 1}
                    )
                    raise
                    
                wait_time = backoff_factor ** attempt
                self.logger.warning(f"Retry attempt {attempt + 1} failed, waiting {wait_time}s: {e}")
                await asyncio.sleep(wait_time)
                
    def circuit_breaker(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        expected_exception: type = Exception
    ):
        """Circuit breaker decorator"""
        
        def decorator(func):
            failure_count = 0
            last_failure_time = None
            
            async def wrapper(*args, **kwargs):
                nonlocal failure_count, last_failure_time
                
                # Check if circuit is open
                if failure_count >= failure_threshold:
                    if last_failure_time and (datetime.utcnow() - last_failure_time).seconds < recovery_timeout:
                        raise Exception("Circuit breaker is open")
                    else:
                        # Reset circuit breaker
                        failure_count = 0
                        last_failure_time = None
                        
                try:
                    if asyncio.iscoroutinefunction(func):
                        result = await func(*args, **kwargs)
                    else:
                        result = func(*args, **kwargs)
                    
                    # Reset on success
                    failure_count = 0
                    return result
                    
                except expected_exception as e:
                    failure_count += 1
                    last_failure_time = datetime.utcnow()
                    
                    await self.handle_error(
                        e,
                        ErrorCategory.SYSTEM,
                        ErrorSeverity.MEDIUM,
                        context={
                            "circuit_breaker": True,
                            "failure_count": failure_count,
                            "threshold": failure_threshold
                        }
                    )
                    raise
                    
            return wrapper
        return decorator
        
    # Error analysis and reporting
    def get_error_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get error statistics for the last N hours"""
        
        from datetime import timedelta
        cutoff_time = datetime.utcnow() - timedelta(hours=hours)
        
        recent_errors = [
            err for err in self.error_history 
            if err.timestamp > cutoff_time
        ]
        
        stats = {
            "total_errors": len(recent_errors),
            "by_category": {},
            "by_severity": {},
            "error_rate": len(recent_errors) / hours if hours > 0 else 0,
            "most_common_errors": {},
            "critical_errors": 0
        }
        
        for error in recent_errors:
            # By category
            category = error.category.value
            stats["by_category"][category] = stats["by_category"].get(category, 0) + 1
            
            # By severity
            severity = error.severity.value
            stats["by_severity"][severity] = stats["by_severity"].get(severity, 0) + 1
            
            if error.severity == ErrorSeverity.CRITICAL:
                stats["critical_errors"] += 1
                
            # Most common errors
            error_type = type(error.exception).__name__
            stats["most_common_errors"][error_type] = stats["most_common_errors"].get(error_type, 0) + 1
            
        return stats
        
    def get_recent_errors(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recent errors"""
        
        recent = sorted(self.error_history, key=lambda x: x.timestamp, reverse=True)[:limit]
        return [err.to_dict() for err in recent]
        
    def clear_error_history(self):
        """Clear error history"""
        self.error_history.clear()
        self.logger.info("Error history cleared")


# Global error handler instance
_error_handler: Optional[ErrorHandler] = None


def get_error_handler(settings: Optional[Settings] = None) -> ErrorHandler:
    """Get global error handler instance"""
    global _error_handler
    if _error_handler is None:
        _error_handler = ErrorHandler(settings)
    return _error_handler