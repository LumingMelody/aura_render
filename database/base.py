"""
Database Base Configuration

Handles database connection, session management, and base model definitions.
"""

import os
from typing import Generator
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.pool import NullPool
import logging

logger = logging.getLogger(__name__)

# Database URL configuration
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./aura_render.db"  # Default to SQLite for development
)

# For production, use PostgreSQL:
# DATABASE_URL = "postgresql://user:password@localhost/aura_render"

# 延迟导入 settings 以避免循环导入
def get_echo_setting() -> bool:
    """获取数据库echo设置"""
    try:
        from config import settings
        return settings.is_development if settings else False
    except ImportError:
        return False

# Create engine with appropriate settings
if DATABASE_URL.startswith("sqlite"):
    # SQLite specific settings
    engine = create_engine(
        DATABASE_URL,
        connect_args={"check_same_thread": False},  # Needed for SQLite
        echo=get_echo_setting(),  # Log SQL in development
    )
else:
    # PostgreSQL/MySQL settings
    engine = create_engine(
        DATABASE_URL,
        echo=get_echo_setting(),
        pool_pre_ping=True,  # Verify connections before using
        pool_size=10,  # Connection pool size
        max_overflow=20,  # Maximum overflow connections
    )

# Create session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

# Create base class for models
metadata = MetaData()
Base = declarative_base(metadata=metadata)


def get_db() -> Generator[Session, None, None]:
    """
    Dependency to get database session.
    
    Usage in FastAPI:
    ```python
    @app.get("/items")
    def read_items(db: Session = Depends(get_db)):
        return db.query(Item).all()
    ```
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db():
    """
    Initialize database by creating all tables.
    
    Call this on application startup.
    """
    try:
        # Import all models to ensure they're registered
        from . import models  # noqa
        
        # Create all tables
        Base.metadata.create_all(bind=engine)
        logger.info("✅ Database tables created successfully")
        
        # Optional: Create initial data
        _create_initial_data()
        
    except Exception as e:
        logger.error(f"❌ Database initialization failed: {e}")
        raise


def _create_initial_data():
    """Create initial/default data if needed"""
    db = SessionLocal()
    try:
        # Example: Create default project or settings
        from .models import Project
        
        default_project = db.query(Project).filter_by(name="Default").first()
        if not default_project:
            default_project = Project(
                name="Default",
                description="Default project for testing"
            )
            db.add(default_project)
            db.commit()
            logger.info("✅ Created default project")
            
    except Exception as e:
        logger.warning(f"⚠️ Could not create initial data: {e}")
    finally:
        db.close()


def drop_all_tables():
    """
    Drop all tables (use with caution!)
    
    Only for development/testing.
    """
    if settings.is_development:
        Base.metadata.drop_all(bind=engine)
        logger.warning("⚠️ All database tables dropped!")
    else:
        logger.error("❌ Cannot drop tables in production mode")
        raise RuntimeError("Attempted to drop tables in production")


def get_db_stats() -> dict:
    """Get database connection statistics"""
    if hasattr(engine.pool, 'size'):
        return {
            "pool_size": engine.pool.size(),
            "checked_in_connections": engine.pool.checkedin(),
            "overflow": engine.pool.overflow(),
            "total": engine.pool.size() + engine.pool.overflow()
        }
    return {"message": "Pool statistics not available"}


if __name__ == "__main__":
    # Test database connection
    print("🔧 Testing database connection...")
    print(f"📍 Database URL: {DATABASE_URL}")
    
    try:
        from sqlalchemy import text
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ Database connection successful!")
            
        # Initialize tables
        init_db()
        print("✅ Database initialized!")
        
    except Exception as e:
        print(f"❌ Database connection failed: {e}")