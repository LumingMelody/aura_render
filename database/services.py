"""
Database Services

Service layer for database operations with clean business logic separation.
"""

from typing import List, Optional, Dict, Any
from datetime import datetime
from sqlalchemy.orm import Session

from .models import Task, Project, TaskStatus, NodeExecution, NodeStatus, Material, MaterialType
import uuid


class TaskService:
    """Service for task-related database operations"""
    
    @staticmethod
    def create_task(
        db: Session,
        theme: str,
        keywords: List[str],
        target_duration: int,
        user_description: str,
        project_id: Optional[int] = None,
        config: Optional[Dict[str, Any]] = None
    ) -> Task:
        """Create a new task"""
        
        # Get or create default project if none specified
        if not project_id:
            default_project = ProjectService.get_or_create_default_project(db)
            project_id = default_project.id
        
        task = Task(
            theme=theme,
            keywords=keywords,
            target_duration=target_duration,
            user_description=user_description,
            project_id=project_id,
            config=config or {},
            status=TaskStatus.PENDING
        )
        
        db.add(task)
        db.commit()
        db.refresh(task)
        return task
    
    @staticmethod
    def get_task_by_id(db: Session, task_id: str) -> Optional[Task]:
        """Get task by task_id"""
        return db.query(Task).filter(Task.task_id == task_id).first()
    
    @staticmethod
    def update_task_status(
        db: Session,
        task_id: str,
        status: TaskStatus,
        progress: Optional[float] = None,
        message: Optional[str] = None,
        error_message: Optional[str] = None,
        output_url: Optional[str] = None,
        result: Optional[Dict[str, Any]] = None
    ) -> Optional[Task]:
        """Update task status and related fields"""
        task = TaskService.get_task_by_id(db, task_id)
        if not task:
            return None
        
        task.status = status
        
        if progress is not None:
            task.progress = progress
        
        if message is not None:
            task.message = message
        
        if error_message is not None:
            task.error_message = error_message
        
        if output_url is not None:
            task.output_url = output_url
            
        if result is not None:
            task.result = result
        
        db.commit()
        db.refresh(task)
        return task
    
    @staticmethod
    def get_tasks_by_status(db: Session, status: TaskStatus, limit: int = 10) -> List[Task]:
        """Get tasks by status"""
        return db.query(Task).filter(Task.status == status).order_by(Task.created_at.desc()).limit(limit).all()
    
    @staticmethod
    def get_recent_tasks(db: Session, limit: int = 20) -> List[Task]:
        """Get recent tasks"""
        return db.query(Task).order_by(Task.created_at.desc()).limit(limit).all()


class ProjectService:
    """Service for project-related database operations"""
    
    @staticmethod
    def get_or_create_default_project(db: Session) -> Project:
        """Get or create the default project"""
        default_project = db.query(Project).filter(Project.name == "Default").first()
        
        if not default_project:
            default_project = Project(
                name="Default",
                description="Default project for video generation tasks"
            )
            db.add(default_project)
            db.commit()
            db.refresh(default_project)
        
        return default_project
    
    @staticmethod
    def create_project(db: Session, name: str, description: str = "") -> Project:
        """Create a new project"""
        project = Project(
            name=name,
            description=description
        )
        db.add(project)
        db.commit()
        db.refresh(project)
        return project
    
    @staticmethod
    def get_project_by_id(db: Session, project_id: int) -> Optional[Project]:
        """Get project by ID"""
        return db.query(Project).filter(Project.id == project_id).first()


class NodeExecutionService:
    """Service for node execution tracking"""
    
    @staticmethod
    def create_node_execution(
        db: Session,
        task_id: int,
        node_name: str,
        node_type: str = None,
        node_version: str = None
    ) -> NodeExecution:
        """Create a new node execution record"""
        execution = NodeExecution(
            task_id=task_id,
            node_name=node_name,
            node_type=node_type,
            node_version=node_version,
            status=NodeStatus.WAITING
        )
        db.add(execution)
        db.commit()
        db.refresh(execution)
        return execution
    
    @staticmethod
    def update_node_execution(
        db: Session,
        execution_id: int,
        status: NodeStatus,
        input_data: Optional[Dict[str, Any]] = None,
        output_data: Optional[Dict[str, Any]] = None,
        error_message: Optional[str] = None
    ) -> Optional[NodeExecution]:
        """Update node execution status"""
        execution = db.query(NodeExecution).filter(NodeExecution.id == execution_id).first()
        if not execution:
            return None
        
        execution.status = status
        
        if status == NodeStatus.RUNNING and not execution.started_at:
            execution.started_at = datetime.utcnow()
        elif status in [NodeStatus.SUCCESS, NodeStatus.ERROR] and not execution.completed_at:
            execution.completed_at = datetime.utcnow()
            if execution.started_at:
                delta = execution.completed_at - execution.started_at
                execution.execution_time = delta.total_seconds()
        
        if input_data is not None:
            execution.input_data = input_data
        
        if output_data is not None:
            execution.output_data = output_data
        
        if error_message is not None:
            execution.error_message = error_message
        
        db.commit()
        db.refresh(execution)
        return execution
    
    @staticmethod
    def get_task_executions(db: Session, task_id: int) -> List[NodeExecution]:
        """Get all node executions for a task"""
        return db.query(NodeExecution).filter(NodeExecution.task_id == task_id).order_by(NodeExecution.started_at).all()


class MaterialService:
    """Service for material/asset management"""
    
    @staticmethod
    def create_material(
        db: Session,
        name: str,
        type: MaterialType,
        url: str,
        description: str = "",
        tags: List[str] = None,
        meta_info: Dict[str, Any] = None,
        quality_score: float = 0.0,
        source: str = "",
        license: str = ""
    ) -> Material:
        """Create a new material"""
        material = Material(
            name=name,
            type=type,
            url=url,
            description=description,
            tags=tags or [],
            meta_info=meta_info or {},
            quality_score=quality_score,
            source=source,
            license=license
        )
        db.add(material)
        db.commit()
        db.refresh(material)
        return material
    
    @staticmethod
    def search_materials(
        db: Session,
        type: Optional[MaterialType] = None,
        tags: Optional[List[str]] = None,
        min_quality: float = 0.0,
        limit: int = 20
    ) -> List[Material]:
        """Search materials by criteria"""
        query = db.query(Material)
        
        if type:
            query = query.filter(Material.type == type)
        
        if min_quality > 0:
            query = query.filter(Material.quality_score >= min_quality)
        
        # For tags, we'd need JSON operators - simplified for now
        if tags:
            # This is a simplified search - in production you'd use JSON operators
            for tag in tags:
                query = query.filter(Material.tags.contains([tag]))
        
        return query.order_by(Material.quality_score.desc()).limit(limit).all()
    
    @staticmethod
    def increment_usage_count(db: Session, material_id: int):
        """Increment the usage count for a material"""
        material = db.query(Material).filter(Material.id == material_id).first()
        if material:
            material.usage_count += 1
            db.commit()