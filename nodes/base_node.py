"""
Base Node Class for Video Generation Pipeline

Abstract base class that defines the interface and common functionality
for all processing nodes in the video generation pipeline.
"""

import logging
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json


class NodeStatus(Enum):
    """Node execution status"""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingPriority(Enum):
    """Processing priority levels"""
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class NodeConfig:
    """Base configuration for all nodes"""
    node_id: str
    name: str
    description: str
    priority: ProcessingPriority = ProcessingPriority.NORMAL
    timeout: float = 300.0  # 5 minutes default
    retry_count: int = 3
    retry_delay: float = 1.0
    parallel_processing: bool = False
    resource_requirements: Dict[str, Any] = field(default_factory=dict)
    custom_params: Dict[str, Any] = field(default_factory=dict)

    def __hash__(self):
        """Make NodeConfig hashable by using node_id"""
        return hash(self.node_id)


@dataclass
class ProcessingContext:
    """Context passed between nodes during processing"""
    task_id: str
    session_id: str
    user_id: Optional[str] = None
    project_data: Dict[str, Any] = field(default_factory=dict)
    intermediate_results: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass 
class NodeResult:
    """Result returned by node processing"""
    status: NodeStatus
    data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Dict[str, Any] = field(default_factory=dict)
    next_nodes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def is_success(self) -> bool:
        """Check if node execution was successful"""
        return self.status == NodeStatus.COMPLETED
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary"""
        return {
            'status': self.status.value,
            'data': self.data,
            'error_message': self.error_message,
            'execution_time': self.execution_time,
            'resource_usage': self.resource_usage,
            'next_nodes': self.next_nodes,
            'metadata': self.metadata
        }


class BaseNode(ABC):
    """Abstract base class for all processing nodes"""
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.logger = logging.getLogger(f"nodes.{config.node_id}")
        self.status = NodeStatus.IDLE
        self.current_context: Optional[ProcessingContext] = None
        self.execution_start_time: Optional[datetime] = None
        
    @property
    def node_id(self) -> str:
        """Get node ID"""
        return self.config.node_id
    
    @property
    def node_name(self) -> str:
        """Get node name"""
        return self.config.name
    
    @abstractmethod
    async def process(self, context: ProcessingContext) -> NodeResult:
        """
        Main processing method that must be implemented by all nodes.
        
        Args:
            context: Processing context containing task data and intermediate results
            
        Returns:
            NodeResult containing the processing outcome
        """
        pass
    
    @abstractmethod
    def validate_input(self, context: ProcessingContext) -> bool:
        """
        Validate input data before processing.
        
        Args:
            context: Processing context to validate
            
        Returns:
            True if input is valid, False otherwise
        """
        pass
    
    def get_required_inputs(self) -> List[str]:
        """
        Get list of required input keys for this node.
        
        Returns:
            List of required input keys
        """
        return []
    
    def get_output_schema(self) -> Dict[str, Any]:
        """
        Get the output schema for this node.
        
        Returns:
            Dictionary describing the output structure
        """
        return {}
    
    async def execute(self, context: ProcessingContext) -> NodeResult:
        """
        Execute the node with error handling and status management.
        
        Args:
            context: Processing context
            
        Returns:
            NodeResult with execution outcome
        """
        self.execution_start_time = datetime.now()
        self.current_context = context
        self.status = NodeStatus.RUNNING
        
        try:
            self.logger.info(f"Starting execution of node {self.node_id}")
            
            # Validate input
            if not self.validate_input(context):
                return self._create_error_result("Input validation failed")
            
            # Check for required inputs
            missing_inputs = []
            for required_input in self.get_required_inputs():
                if required_input not in context.intermediate_results:
                    missing_inputs.append(required_input)
            
            if missing_inputs:
                return self._create_error_result(
                    f"Missing required inputs: {', '.join(missing_inputs)}"
                )
            
            # Execute with retry logic
            result = await self._execute_with_retry(context)
            
            # Update status based on result
            self.status = result.status
            
            # Calculate execution time
            if self.execution_start_time:
                execution_time = (datetime.now() - self.execution_start_time).total_seconds()
                result.execution_time = execution_time
            
            self.logger.info(
                f"Node {self.node_id} completed with status {result.status.value} "
                f"in {result.execution_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Node {self.node_id} failed with exception: {e}")
            self.status = NodeStatus.FAILED
            return self._create_error_result(str(e))
        
        finally:
            self.current_context = None
    
    async def _execute_with_retry(self, context: ProcessingContext) -> NodeResult:
        """Execute node with retry logic"""
        last_error = None
        
        for attempt in range(self.config.retry_count + 1):
            try:
                # Set timeout for processing
                result = await asyncio.wait_for(
                    self.process(context),
                    timeout=self.config.timeout
                )
                
                if result.is_success():
                    return result
                else:
                    last_error = result.error_message or "Processing failed"
                    
            except asyncio.TimeoutError:
                last_error = f"Node execution timed out after {self.config.timeout}s"
                self.logger.warning(f"Attempt {attempt + 1} timed out for node {self.node_id}")
                
            except Exception as e:
                last_error = str(e)
                self.logger.warning(f"Attempt {attempt + 1} failed for node {self.node_id}: {e}")
            
            # Wait before retry (except on last attempt)
            if attempt < self.config.retry_count:
                await asyncio.sleep(self.config.retry_delay * (attempt + 1))
        
        return self._create_error_result(
            f"Node failed after {self.config.retry_count + 1} attempts. Last error: {last_error}"
        )
    
    def _create_error_result(self, error_message: str) -> NodeResult:
        """Create an error result"""
        execution_time = 0.0
        if self.execution_start_time:
            execution_time = (datetime.now() - self.execution_start_time).total_seconds()
        
        return NodeResult(
            status=NodeStatus.FAILED,
            error_message=error_message,
            execution_time=execution_time,
            metadata={'node_id': self.node_id, 'node_name': self.node_name}
        )
    
    def get_status(self) -> Dict[str, Any]:
        """Get current node status"""
        return {
            'node_id': self.node_id,
            'node_name': self.node_name,
            'status': self.status.value,
            'execution_start_time': self.execution_start_time.isoformat() if self.execution_start_time else None,
            'current_task': self.current_context.task_id if self.current_context else None
        }
    
    def reset(self):
        """Reset node to idle state"""
        self.status = NodeStatus.IDLE
        self.current_context = None
        self.execution_start_time = None
        self.logger.info(f"Node {self.node_id} reset to idle state")