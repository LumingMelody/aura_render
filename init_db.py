#!/usr/bin/env python3
"""
Database Initialization Script

Run this script to initialize the database and create all tables.
"""

import sys
from pathlib import Path
import logging

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from database import init_db, Base, get_db
from database.models import Project, Task, TaskStatus
from config import settings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_sample_data():
    """Create some sample data for testing"""
    from database.base import SessionLocal
    
    db = SessionLocal()
    try:
        # Check if we already have data
        existing_projects = db.query(Project).count()
        if existing_projects > 0:
            logger.info(f"Database already has {existing_projects} projects, skipping sample data")
            return
        
        # Create a sample project
        sample_project = Project(
            name="示例项目",
            description="这是一个示例项目，用于测试视频生成功能"
        )
        db.add(sample_project)
        db.commit()
        db.refresh(sample_project)
        
        # Create a sample task
        sample_task = Task(
            project_id=sample_project.id,
            theme="AI技术展示",
            keywords=["人工智能", "创新", "未来"],
            target_duration=60,
            user_description="创建一个展示AI技术的60秒宣传视频",
            status=TaskStatus.PENDING
        )
        db.add(sample_task)
        db.commit()
        
        logger.info("✅ Created sample project and task")
        
    except Exception as e:
        logger.error(f"❌ Failed to create sample data: {e}")
        db.rollback()
    finally:
        db.close()


def check_database_status():
    """Check database connection and table status"""
    from database.base import engine
    from sqlalchemy import inspect
    
    try:
        # Check connection
        from sqlalchemy import text
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection successful")
        
        # Check tables
        inspector = inspect(engine)
        tables = inspector.get_table_names()
        
        if tables:
            logger.info(f"📊 Found {len(tables)} tables: {', '.join(tables)}")
        else:
            logger.info("📋 No tables found, will create them")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Database check failed: {e}")
        return False


def main():
    """Main initialization function"""
    print("🔧 Aura Render Database Initialization")
    print("=" * 50)
    
    # Check database status
    if not check_database_status():
        print("❌ Cannot connect to database")
        print("💡 Please check your DATABASE_URL environment variable")
        print("   Default: sqlite:///./aura_render.db")
        print("   PostgreSQL: postgresql://user:password@localhost/aura_render")
        return 1
    
    # Initialize database
    try:
        print("\n📊 Creating database tables...")
        init_db()
        print("✅ Database initialized successfully!")
        
        # Optionally create sample data
        if settings.is_development:
            print("\n📝 Creating sample data...")
            create_sample_data()
            print("✅ Sample data created!")
        
        # Show final status
        print("\n📊 Database Status:")
        from database.base import SessionLocal
        db = SessionLocal()
        try:
            project_count = db.query(Project).count()
            task_count = db.query(Task).count()
            print(f"   Projects: {project_count}")
            print(f"   Tasks: {task_count}")
        finally:
            db.close()
        
        print("\n🎉 Database initialization complete!")
        return 0
        
    except Exception as e:
        print(f"\n❌ Database initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())