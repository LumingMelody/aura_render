#!/usr/bin/env python3
"""
Improved Base Node Implementation

Enhanced base node with better error handling, logging, and configuration integration.
"""

import sys
from pathlib import Path
import logging
import json
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Union
from datetime import datetime

# Add project root for imports  
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import settings

logger = logging.getLogger(__name__)


class NodeExecutionError(Exception):
    """Custom exception for node execution errors"""
    pass


class BaseNodeImproved(ABC):
    """
    Improved Base Node with better architecture
    
    Features:
    - Better error handling and logging
    - Configuration integration
    - Async support
    - Input/output validation
    - Performance monitoring
    """
    
    # Node metadata (to be overridden by subclasses)
    node_name: str = "BaseNode"
    node_description: str = "Base node for video generation pipeline"
    node_version: str = "1.0.0"
    
    # Input/Output schema
    required_inputs: List[Dict[str, Any]] = []
    output_schema: List[Dict[str, Any]] = []
    
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.execution_history: List[Dict[str, Any]] = []
        self.last_execution_time: Optional[datetime] = None
        self.logger = logging.getLogger(f"{__class__.__module__}.{node_id}")
        
        self.logger.info(f"🔧 Initialized {self.node_name} (ID: {node_id})")
    
    def validate_inputs(self, context: Dict[str, Any]) -> bool:
        """
        Validate input context against required inputs schema
        """
        missing_inputs = []
        type_errors = []
        
        for input_spec in self.required_inputs:
            name = input_spec["name"]
            required = input_spec.get("required", True)
            expected_type = input_spec.get("type", str)
            
            if name not in context:
                if required:
                    missing_inputs.append(name)
            else:
                value = context[name]
                # Basic type checking
                if expected_type == str and not isinstance(value, str):
                    type_errors.append(f"{name}: expected string, got {type(value)}")
                elif expected_type == int and not isinstance(value, int):
                    type_errors.append(f"{name}: expected int, got {type(value)}")
                elif expected_type == list and not isinstance(value, list):
                    type_errors.append(f"{name}: expected list, got {type(value)}")
        
        if missing_inputs:
            raise NodeExecutionError(f"Missing required inputs: {missing_inputs}")
        
        if type_errors:
            raise NodeExecutionError(f"Type validation errors: {type_errors}")
        
        return True
    
    def log_execution(self, context: Dict[str, Any], result: Dict[str, Any], 
                     execution_time: float, success: bool = True, error: Optional[str] = None):
        """Log execution details"""
        execution_record = {
            "timestamp": datetime.now().isoformat(),
            "node_id": self.node_id,
            "node_name": self.node_name,
            "execution_time": execution_time,
            "success": success,
            "input_keys": list(context.keys()),
            "output_keys": list(result.keys()) if result else [],
            "error": error
        }
        
        self.execution_history.append(execution_record)
        self.last_execution_time = datetime.now()
        
        if settings.development.save_intermediate_results:
            log_file = settings.storage.cache_dir / f"{self.node_id}_execution.jsonl"
            with open(log_file, "a", encoding='utf-8') as f:
                f.write(json.dumps(execution_record, ensure_ascii=False) + "\n")
    
    async def execute_async(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async wrapper for node execution with full error handling and monitoring
        """
        start_time = datetime.now()
        
        try:
            self.logger.info(f"🚀 Executing {self.node_name}")
            
            # Validate inputs
            self.validate_inputs(context)
            
            # Execute the actual node logic
            result = await self.generate_async(context)
            
            # Calculate execution time
            execution_time = (datetime.now() - start_time).total_seconds()
            
            # Log successful execution
            self.log_execution(context, result, execution_time, success=True)
            
            self.logger.info(f"✅ {self.node_name} completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            error_msg = str(e)
            
            # Log failed execution
            self.log_execution(context, {}, execution_time, success=False, error=error_msg)
            
            self.logger.error(f"❌ {self.node_name} failed: {error_msg}")
            raise NodeExecutionError(f"{self.node_name} execution failed: {error_msg}") from e
    
    def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous wrapper for backward compatibility
        """
        return asyncio.run(self.execute_async(context))
    
    @abstractmethod
    async def generate_async(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Async implementation of node logic (to be implemented by subclasses)
        """
        pass
    
    def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Synchronous generate method for backward compatibility
        """
        return asyncio.run(self.generate_async(context))
    
    async def regenerate_async(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Regenerate with user feedback (async)
        """
        # Default implementation: just re-run generate with modified context
        modified_context = {**context, **user_intent}
        return await self.generate_async(modified_context)
    
    def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        """
        Regenerate with user feedback (sync)
        """
        return asyncio.run(self.regenerate_async(context, user_intent))
    
    def get_node_info(self) -> Dict[str, Any]:
        """Get node metadata and status"""
        return {
            "node_id": self.node_id,
            "node_name": self.node_name,
            "description": self.node_description,
            "version": self.node_version,
            "required_inputs": self.required_inputs,
            "output_schema": self.output_schema,
            "last_execution": self.last_execution_time.isoformat() if self.last_execution_time else None,
            "execution_count": len(self.execution_history),
            "success_rate": self._calculate_success_rate()
        }
    
    def _calculate_success_rate(self) -> float:
        """Calculate success rate from execution history"""
        if not self.execution_history:
            return 0.0
        
        successful = sum(1 for record in self.execution_history if record["success"])
        return successful / len(self.execution_history)
    
    def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution statistics"""
        if not self.execution_history:
            return {"message": "No execution history"}
        
        execution_times = [record["execution_time"] for record in self.execution_history if record["success"]]
        
        return {
            "total_executions": len(self.execution_history),
            "successful_executions": len(execution_times),
            "success_rate": self._calculate_success_rate(),
            "avg_execution_time": sum(execution_times) / len(execution_times) if execution_times else 0,
            "min_execution_time": min(execution_times) if execution_times else 0,
            "max_execution_time": max(execution_times) if execution_times else 0,
            "last_execution": self.last_execution_time.isoformat() if self.last_execution_time else None
        }


# Utility function for creating AI-powered nodes
async def call_ai_service(prompt: str, **kwargs) -> Dict[str, Any]:
    """
    Utility function to call AI service (Qwen) with proper error handling
    """
    try:
        from llm.qwen import QwenLLM
        
        # Initialize LLM with settings
        llm = QwenLLM(
            model_name=settings.ai.qwen_model_name,
            api_key=settings.ai.dashscope_api_key
        )
        
        # Call LLM with retry logic
        for attempt in range(settings.ai.max_retries):
            try:
                result = llm.generate(prompt, parse_json=kwargs.get('parse_json', False))
                return {"result": result, "success": True}
            except Exception as e:
                if attempt < settings.ai.max_retries - 1:
                    await asyncio.sleep(settings.ai.retry_delay * (attempt + 1))
                    continue
                raise e
                
    except Exception as e:
        logger.error(f"AI service call failed: {e}")
        return {"result": None, "success": False, "error": str(e)}


if __name__ == "__main__":
    # Simple test
    print("🧪 Testing BaseNodeImproved...")
    
    class TestNode(BaseNodeImproved):
        node_name = "TestNode"
        required_inputs = [
            {"name": "test_input", "type": str, "required": True}
        ]
        
        async def generate_async(self, context: Dict[str, Any]) -> Dict[str, Any]:
            await asyncio.sleep(0.1)  # Simulate work
            return {"test_output": f"Processed: {context['test_input']}"}
    
    # Test the node
    test_node = TestNode("test_001")
    result = test_node.execute({"test_input": "Hello World"})
    print(f"✅ Result: {result}")
    print(f"📊 Stats: {test_node.get_execution_stats()}")