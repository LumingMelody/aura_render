#!/usr/bin/env python3
"""
Intelligent Prompt Management System for Aura Render

Advanced prompt template management with features:
- Template composition and inheritance
- Variable interpolation with validation
- Context-aware prompt generation
- Performance tracking and optimization
- Multi-language support
- Dynamic prompt adaptation
"""

import json
import re
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Set, Tuple, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import yaml
import jinja2
from pydantic import BaseModel, Field, validator

from utils.logger import get_logger


class PromptType(Enum):
    """Prompt template types"""
    VIDEO_SCRIPT = "video_script"
    SCENE_ANALYSIS = "scene_analysis"
    CHARACTER_DEVELOPMENT = "character_development"
    DIALOGUE_GENERATION = "dialogue_generation"
    VISUAL_DESCRIPTION = "visual_description"
    EMOTION_ANALYSIS = "emotion_analysis"
    CONTENT_OPTIMIZATION = "content_optimization"
    QUALITY_ASSESSMENT = "quality_assessment"
    STYLE_TRANSFER = "style_transfer"
    TRANSLATION = "translation"
    SUMMARIZATION = "summarization"
    SYSTEM = "system"
    CONTEXTUAL = "contextual"


class DifficultyLevel(Enum):
    """Template difficulty levels"""
    SIMPLE = "simple"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


@dataclass
class PromptTemplate:
    """Enhanced prompt template with metadata"""
    id: str
    name: str
    type: PromptType
    content: str
    variables: List[str] = field(default_factory=list)
    description: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    difficulty: DifficultyLevel = DifficultyLevel.SIMPLE
    version: str = "1.0.0"
    author: Optional[str] = None
    language: str = "en"
    parent_template: Optional[str] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)
    validation_rules: Dict[str, Any] = field(default_factory=dict)
    performance_stats: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[datetime] = None
    updated_at: Optional[datetime] = None
    usage_count: int = 0
    success_rate: float = 0.0
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.created_at is None:
            self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # Extract variables from content if not provided
        if not self.variables:
            self.variables = self._extract_variables()
    
    def _extract_variables(self) -> List[str]:
        """Extract variables from template content"""
        # Extract Jinja2 variables
        pattern = r'\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}'
        variables = set(re.findall(pattern, self.content))
        
        # Also look for simple placeholder format
        pattern2 = r'\{([a-zA-Z_][a-zA-Z0-9_]*)\}'
        variables.update(re.findall(pattern2, self.content))
        
        return sorted(list(variables))
    
    def validate_variables(self, variables: Dict[str, Any]) -> List[str]:
        """Validate provided variables against template requirements"""
        errors = []
        
        # Check required variables
        missing_vars = set(self.variables) - set(variables.keys())
        if missing_vars:
            errors.append(f"Missing required variables: {missing_vars}")
        
        # Apply validation rules
        for var_name, rules in self.validation_rules.items():
            if var_name not in variables:
                continue
            
            value = variables[var_name]
            
            # Type validation
            if "type" in rules:
                expected_type = rules["type"]
                if expected_type == "string" and not isinstance(value, str):
                    errors.append(f"Variable '{var_name}' must be a string")
                elif expected_type == "number" and not isinstance(value, (int, float)):
                    errors.append(f"Variable '{var_name}' must be a number")
                elif expected_type == "boolean" and not isinstance(value, bool):
                    errors.append(f"Variable '{var_name}' must be a boolean")
            
            # Length validation for strings
            if isinstance(value, str):
                if "min_length" in rules and len(value) < rules["min_length"]:
                    errors.append(f"Variable '{var_name}' must be at least {rules['min_length']} characters")
                if "max_length" in rules and len(value) > rules["max_length"]:
                    errors.append(f"Variable '{var_name}' must not exceed {rules['max_length']} characters")
            
            # Range validation for numbers
            if isinstance(value, (int, float)):
                if "min_value" in rules and value < rules["min_value"]:
                    errors.append(f"Variable '{var_name}' must be at least {rules['min_value']}")
                if "max_value" in rules and value > rules["max_value"]:
                    errors.append(f"Variable '{var_name}' must not exceed {rules['max_value']}")
            
            # Enum validation
            if "allowed_values" in rules and value not in rules["allowed_values"]:
                errors.append(f"Variable '{var_name}' must be one of: {rules['allowed_values']}")
        
        return errors
    
    def render(self, variables: Dict[str, Any]) -> str:
        """Render template with variables"""
        # Validate variables
        validation_errors = self.validate_variables(variables)
        if validation_errors:
            raise ValueError(f"Variable validation failed: {validation_errors}")
        
        # Use Jinja2 for advanced templating
        try:
            template = jinja2.Template(self.content)
            rendered = template.render(**variables)
            
            # Update usage statistics
            self.usage_count += 1
            self.updated_at = datetime.now()
            
            return rendered.strip()
            
        except jinja2.TemplateError as e:
            raise ValueError(f"Template rendering failed: {e}")
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        # Convert enums and datetime objects
        data["type"] = self.type.value
        data["difficulty"] = self.difficulty.value
        if self.created_at:
            data["created_at"] = self.created_at.isoformat()
        if self.updated_at:
            data["updated_at"] = self.updated_at.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PromptTemplate':
        """Create from dictionary"""
        # Convert string enums back
        data["type"] = PromptType(data["type"])
        data["difficulty"] = DifficultyLevel(data["difficulty"])
        
        # Convert datetime strings
        if data.get("created_at"):
            data["created_at"] = datetime.fromisoformat(data["created_at"])
        if data.get("updated_at"):
            data["updated_at"] = datetime.fromisoformat(data["updated_at"])
        
        return cls(**data)


class PromptManager:
    """Intelligent prompt management system"""
    
    def __init__(self, templates_dir: Optional[Path] = None):
        """Initialize prompt manager"""
        self.logger = get_logger(__name__)
        self.templates: Dict[str, PromptTemplate] = {}
        self.templates_dir = templates_dir or Path("prompts")
        
        # Performance tracking
        self.usage_stats = {
            "total_renders": 0,
            "successful_renders": 0,
            "failed_renders": 0,
            "average_render_time": 0.0,
            "template_usage": {},
            "variable_usage": {},
            "error_patterns": {}
        }
        
        # Load built-in templates
        self._load_builtin_templates()
        
        # Load custom templates if directory exists
        if self.templates_dir.exists():
            self._load_templates_from_directory()
        
        self.logger.info(f"Prompt manager initialized with {len(self.templates)} templates")
    
    def _load_builtin_templates(self):
        """Load built-in prompt templates"""
        builtin_templates = [
            PromptTemplate(
                id="video_script_generation",
                name="Video Script Generation",
                type=PromptType.VIDEO_SCRIPT,
                content="""Create a compelling video script for {{ topic }}.

Target Audience: {{ target_audience }}
Video Duration: {{ duration }} seconds
Video Style: {{ style }}

Requirements:
- Hook viewers in the first 5 seconds
- Include clear call-to-action
- Match the specified style and tone
- Optimize for {{ platform }} platform

Generate a structured script with:
1. Opening hook
2. Main content sections
3. Transitions between sections
4. Closing with call-to-action

Output in JSON format with timing and visual cues.""",
                variables=["topic", "target_audience", "duration", "style", "platform"],
                description="Generate comprehensive video scripts with timing and structure",
                tags=["video", "script", "content", "generation"],
                difficulty=DifficultyLevel.INTERMEDIATE,
                validation_rules={
                    "duration": {"type": "number", "min_value": 15, "max_value": 600},
                    "target_audience": {"type": "string", "min_length": 3},
                    "style": {"type": "string", "allowed_values": ["educational", "entertainment", "marketing", "documentary", "tutorial"]}
                }
            ),
            
            PromptTemplate(
                id="scene_analysis_comprehensive",
                name="Comprehensive Scene Analysis",
                type=PromptType.SCENE_ANALYSIS,
                content="""Analyze the following scene description in detail:

Scene: {{ scene_description }}

Context:
- Video Genre: {{ genre }}
- Target Mood: {{ target_mood }}
- Visual Style: {{ visual_style }}

Provide comprehensive analysis covering:

1. **Visual Elements**:
   - Composition and framing
   - Color palette and lighting
   - Key visual focal points
   
2. **Emotional Impact**:
   - Mood and atmosphere
   - Emotional arc
   - Viewer engagement factors

3. **Technical Considerations**:
   - Camera angles and movements
   - Transition opportunities
   - Special effects requirements

4. **Content Optimization**:
   - Pacing recommendations
   - Enhancement suggestions
   - Platform-specific adaptations

Return analysis in structured JSON format.""",
                variables=["scene_description", "genre", "target_mood", "visual_style"],
                description="Comprehensive scene analysis for video production",
                tags=["scene", "analysis", "visual", "technical"],
                difficulty=DifficultyLevel.ADVANCED
            )
        ]
        
        for template in builtin_templates:
            self.templates[template.id] = template
            self.logger.debug(f"Loaded built-in template: {template.id}")
    
    def _load_templates_from_directory(self):
        """Load templates from directory"""
        template_files = list(self.templates_dir.glob("*.json")) + list(self.templates_dir.glob("*.yaml"))
        
        for file_path in template_files:
            try:
                if file_path.suffix == ".json":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                elif file_path.suffix == ".yaml":
                    with open(file_path, 'r', encoding='utf-8') as f:
                        data = yaml.safe_load(f)
                
                template = PromptTemplate.from_dict(data)
                self.templates[template.id] = template
                self.logger.debug(f"Loaded template from file: {template.id}")
                
            except Exception as e:
                self.logger.error(f"Failed to load template from {file_path}: {e}")
    
    def get_template(self, template_id: str) -> Optional[PromptTemplate]:
        """Get template by ID"""
        return self.templates.get(template_id)
    
    def list_templates(self, 
                      type_filter: Optional[PromptType] = None,
                      tag_filter: Optional[str] = None,
                      difficulty_filter: Optional[DifficultyLevel] = None) -> List[PromptTemplate]:
        """List templates with optional filtering"""
        templates = list(self.templates.values())
        
        if type_filter:
            templates = [t for t in templates if t.type == type_filter]
        
        if tag_filter:
            templates = [t for t in templates if tag_filter in t.tags]
        
        if difficulty_filter:
            templates = [t for t in templates if t.difficulty == difficulty_filter]
        
        # Sort by usage and success rate
        templates.sort(key=lambda t: (t.usage_count, t.success_rate), reverse=True)
        
        return templates
    
    def render_prompt(self, template_id: str, variables: Dict[str, Any]) -> Optional[str]:
        """Render a single prompt template"""
        template = self.get_template(template_id)
        if not template:
            self.logger.error(f"Template not found: {template_id}")
            return None
        
        try:
            rendered = template.render(variables)
            return rendered
            
        except Exception as e:
            self.logger.error(f"Failed to render template {template_id}: {e}")
            return None
    
    def save_template(self, template: PromptTemplate) -> bool:
        """Save template to manager and optionally to file"""
        try:
            # Add to memory
            self.templates[template.id] = template
            self.logger.info(f"Template saved: {template.id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save template {template.id}: {e}")
            return False
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get usage statistics"""
        return self.usage_stats.copy()


# Global instance
_prompt_manager: Optional[PromptManager] = None


def get_prompt_manager() -> PromptManager:
    """Get or create prompt manager instance"""
    global _prompt_manager
    
    if _prompt_manager is None:
        _prompt_manager = PromptManager()
    
    return _prompt_manager