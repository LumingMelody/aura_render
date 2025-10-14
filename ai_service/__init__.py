"""
Aura Render AI Service Module

Enhanced AI service integration with advanced features including:
- Multi-model support (Qwen, OpenAI, Claude)
- Intelligent prompt management
- Context-aware conversation handling
- Quality assessment and content optimization
- Vision analysis capabilities
- Performance monitoring and caching
"""

from .enhanced_qwen_service import EnhancedQwenService, get_enhanced_qwen_service
from .prompt_manager import (
    PromptManager, PromptTemplate, PromptType, DifficultyLevel,
    get_prompt_manager
)
from .context_manager import (
    ContextManager, ContextType, ContextPriority, ConversationContext,
    get_context_manager
)
from .quality_assessor import (
    QualityAssessor, ContentType, QualityDimension, AssessmentResult,
    get_quality_assessor
)
from .vision_analyzer import (
    VisionAnalyzer, AnalysisType, VisionAnalysisResult,
    get_vision_analyzer
)
from .ai_coordinator import AICoordinator, get_ai_coordinator
from .performance_monitor import AIPerformanceMonitor, get_performance_monitor

__all__ = [
    # Enhanced Services
    'EnhancedQwenService', 'get_enhanced_qwen_service',
    
    # Prompt Management
    'PromptManager', 'PromptTemplate', 'PromptType', 'DifficultyLevel',
    'get_prompt_manager',
    
    # Context Management
    'ContextManager', 'ContextType', 'ContextPriority', 'ConversationContext',
    'get_context_manager',
    
    # Quality Assessment
    'QualityAssessor', 'ContentType', 'QualityDimension', 'AssessmentResult',
    'get_quality_assessor',
    
    # Vision Analysis
    'VisionAnalyzer', 'AnalysisType', 'VisionAnalysisResult',
    'get_vision_analyzer',
    
    # Coordination and Monitoring
    'AICoordinator', 'get_ai_coordinator',
    'AIPerformanceMonitor', 'get_performance_monitor'
]