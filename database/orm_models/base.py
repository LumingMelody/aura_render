"""
Base Database Models

Foundation classes and utilities for all database models including
base model class, mixins for timestamps and soft deletes, and database session management.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Any, Dict
from sqlalchemy import (
    create_engine, Column, String, DateTime, Boolean, 
    Integer, Text, JSON, Index, event
)
from sqlalchemy.ext.declarative import declarative_base, declared_attr
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.engine import Engine
import logging

logger = logging.getLogger(__name__)

# Base declarative model
Base = declarative_base()

class BaseModel(Base):
    """Abstract base model with common fields and functionality"""
    __abstract__ = True
    
    @declared_attr
    def __tablename__(cls):
        """Generate table name from class name"""
        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()
    
    # Primary key as UUID
    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    
    def to_dict(self, exclude_fields: Optional[set] = None) -> Dict[str, Any]:
        """Convert model instance to dictionary"""
        exclude_fields = exclude_fields or set()
        
        result = {}
        for column in self.__table__.columns:
            if column.name in exclude_fields:
                continue
            
            value = getattr(self, column.name)
            
            # Handle datetime objects
            if isinstance(value, datetime):
                result[column.name] = value.isoformat()
            # Handle UUID objects
            elif hasattr(value, 'hex'):
                result[column.name] = str(value)
            # Handle JSON fields
            elif column.type.python_type in (dict, list):
                result[column.name] = value
            else:
                result[column.name] = value
        
        return result
    
    def update_from_dict(self, data: Dict[str, Any], exclude_fields: Optional[set] = None):
        """Update model instance from dictionary"""
        exclude_fields = exclude_fields or {'id', 'created_at'}
        
        for key, value in data.items():
            if key in exclude_fields:
                continue
            
            if hasattr(self, key):
                setattr(self, key, value)
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(id={self.id})>"

class TimestampMixin:
    """Mixin for created_at and updated_at timestamps"""
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), nullable=False)
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc), nullable=False)

class SoftDeleteMixin:
    """Mixin for soft delete functionality"""
    is_deleted = Column(Boolean, default=False, nullable=False)
    deleted_at = Column(DateTime(timezone=True), nullable=True)
    
    def soft_delete(self):
        """Mark record as deleted"""
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)
    
    def restore(self):
        """Restore soft deleted record"""
        self.is_deleted = False
        self.deleted_at = None

class UserTrackingMixin:
    """Mixin for tracking user who created/modified records"""
    created_by_id = Column(UUID(as_uuid=True), nullable=True)
    updated_by_id = Column(UUID(as_uuid=True), nullable=True)

class VersionMixin:
    """Mixin for record versioning"""
    version = Column(Integer, default=1, nullable=False)
    
    def increment_version(self):
        """Increment record version"""
        self.version = (self.version or 0) + 1

class MetadataMixin:
    """Mixin for storing additional metadata"""
    extra_metadata = Column(JSON, default=dict, nullable=False)
    
    def set_metadata(self, key: str, value: Any):
        """Set metadata value"""
        if self.extra_metadata is None:
            self.extra_metadata = {}
        self.extra_metadata[key] = value
    
    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Get metadata value"""
        if self.extra_metadata is None:
            return default
        return self.extra_metadata.get(key, default)

# Database connection management
class DatabaseManager:
    """Database connection and session management"""
    
    def __init__(self):
        self.engine: Optional[Engine] = None
        self.SessionLocal: Optional[sessionmaker] = None
    
    def initialize(self, database_url: str, **kwargs):
        """Initialize database connection"""
        try:
            # Default engine options for production
            engine_kwargs = {
                'echo': kwargs.get('echo', False),
                'pool_size': kwargs.get('pool_size', 20),
                'max_overflow': kwargs.get('max_overflow', 30),
                'pool_timeout': kwargs.get('pool_timeout', 30),
                'pool_recycle': kwargs.get('pool_recycle', 3600),
                'pool_pre_ping': kwargs.get('pool_pre_ping', True),
            }
            
            # PostgreSQL specific settings
            if database_url.startswith('postgresql'):
                engine_kwargs.update({
                    'connect_args': {
                        'connect_timeout': 10,
                        'application_name': 'aura_render',
                        'options': '-c timezone=utc'
                    }
                })
            
            # SQLite specific settings
            elif database_url.startswith('sqlite'):
                engine_kwargs = {
                    'echo': kwargs.get('echo', False),
                    'connect_args': {
                        'check_same_thread': False,
                        'timeout': 30
                    }
                }
            
            self.engine = create_engine(database_url, **engine_kwargs)
            self.SessionLocal = sessionmaker(
                autocommit=False, 
                autoflush=False, 
                bind=self.engine
            )
            
            # Set up event listeners
            self._setup_event_listeners()
            
            logger.info(f"Database initialized: {database_url.split('://')[0]}://...")
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            raise
    
    def create_all_tables(self):
        """Create all tables"""
        if not self.engine:
            raise RuntimeError("Database not initialized")
        
        try:
            Base.metadata.create_all(bind=self.engine)
            logger.info("All database tables created successfully")
        except Exception as e:
            logger.error(f"Table creation failed: {e}")
            raise
    
    def drop_all_tables(self):
        """Drop all tables (use with caution!)"""
        if not self.engine:
            raise RuntimeError("Database not initialized")
        
        try:
            Base.metadata.drop_all(bind=self.engine)
            logger.warning("All database tables dropped")
        except Exception as e:
            logger.error(f"Table drop failed: {e}")
            raise
    
    def get_session(self) -> Session:
        """Get database session"""
        if not self.SessionLocal:
            raise RuntimeError("Database not initialized")
        
        return self.SessionLocal()
    
    def _setup_event_listeners(self):
        """Set up SQLAlchemy event listeners"""
        
        @event.listens_for(self.engine, "connect")
        def set_sqlite_pragma(dbapi_connection, connection_record):
            """Set SQLite pragmas for better performance"""
            if 'sqlite' in self.engine.url.drivername:
                cursor = dbapi_connection.cursor()
                # Enable foreign key constraints
                cursor.execute("PRAGMA foreign_keys=ON")
                # Enable WAL mode for better concurrency
                cursor.execute("PRAGMA journal_mode=WAL")
                # Set synchronous to NORMAL for better performance
                cursor.execute("PRAGMA synchronous=NORMAL")
                cursor.close()
        
        @event.listens_for(Session, "before_flush")
        def receive_before_flush(session, flush_context, instances):
            """Update timestamps before flush"""
            for instance in session.new | session.dirty:
                if hasattr(instance, 'updated_at'):
                    instance.updated_at = datetime.now(timezone.utc)
    
    def health_check(self) -> bool:
        """Check database connectivity"""
        try:
            with self.get_session() as session:
                session.execute("SELECT 1")
                return True
        except Exception as e:
            logger.error(f"Database health check failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get database connection statistics"""
        if not self.engine:
            return {"status": "not_initialized"}
        
        pool = self.engine.pool
        return {
            "status": "connected",
            "pool_size": pool.size(),
            "checked_in": pool.checkedin(),
            "checked_out": pool.checkedout(),
            "overflow": pool.overflow(),
            "invalid": pool.invalid()
        }

# Global database manager instance
db_manager = DatabaseManager()

def get_db_engine() -> Engine:
    """Get database engine"""
    if not db_manager.engine:
        raise RuntimeError("Database not initialized")
    return db_manager.engine

def get_db_session() -> Session:
    """Get database session"""
    return db_manager.get_session()

def init_database(database_url: str, **kwargs):
    """Initialize database"""
    db_manager.initialize(database_url, **kwargs)

def create_tables():
    """Create all database tables"""
    db_manager.create_all_tables()

def drop_tables():
    """Drop all database tables"""
    db_manager.drop_all_tables()

def get_db_health() -> bool:
    """Check database health"""
    return db_manager.health_check()

def get_db_stats() -> Dict[str, Any]:
    """Get database statistics"""
    return db_manager.get_stats()

# Database session dependency for FastAPI
def get_db():
    """Database session dependency"""
    db = get_db_session()
    try:
        yield db
    finally:
        db.close()

# Enhanced base model with all mixins
class EnhancedBaseModel(BaseModel, TimestampMixin, SoftDeleteMixin, 
                       UserTrackingMixin, VersionMixin, MetadataMixin):
    """Enhanced base model with all common functionality"""
    __abstract__ = True