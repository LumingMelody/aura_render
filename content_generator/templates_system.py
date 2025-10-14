"""
Advanced Video Templates and Presets System

Provides professional video templates, presets, and customizable templates for
different use cases like marketing, education, social media, etc.
"""

import json
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
from datetime import datetime

from pydantic import BaseModel, Field


class TemplateCategory(str, Enum):
    """Template categories for different use cases"""
    MARKETING = "marketing"
    EDUCATION = "education"
    SOCIAL_MEDIA = "social_media"
    CORPORATE = "corporate"
    ENTERTAINMENT = "entertainment"
    NEWS = "news"
    TUTORIAL = "tutorial"
    PRESENTATION = "presentation"


class TemplateStyle(str, Enum):
    """Visual styles for templates"""
    MODERN = "modern"
    CLASSIC = "classic"
    MINIMAL = "minimal"
    DYNAMIC = "dynamic"
    PROFESSIONAL = "professional"
    CREATIVE = "creative"
    CINEMATIC = "cinematic"
    CASUAL = "casual"


@dataclass
class SceneConfiguration:
    """Configuration for individual scenes in templates"""
    duration: float = 5.0
    transition_type: str = "fade"
    background_type: str = "solid"
    background_value: str = "#000000"
    text_overlay: Optional[Dict[str, Any]] = None
    music_enabled: bool = True
    effects: List[str] = None
    
    def __post_init__(self):
        if self.effects is None:
            self.effects = []


@dataclass
class TemplateAssets:
    """Assets required for a template"""
    background_images: List[str] = None
    background_videos: List[str] = None
    overlay_graphics: List[str] = None
    audio_tracks: List[str] = None
    fonts: List[str] = None
    
    def __post_init__(self):
        if self.background_images is None:
            self.background_images = []
        if self.background_videos is None:
            self.background_videos = []
        if self.overlay_graphics is None:
            self.overlay_graphics = []
        if self.audio_tracks is None:
            self.audio_tracks = []
        if self.fonts is None:
            self.fonts = ["Arial", "Helvetica"]


class VideoTemplate(BaseModel):
    """Complete video template definition"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    description: str
    category: TemplateCategory
    style: TemplateStyle
    duration: float = Field(default=30.0, description="Default duration in seconds")
    aspect_ratio: str = Field(default="16:9", description="Video aspect ratio")
    resolution: str = Field(default="1920x1080", description="Video resolution")
    fps: int = Field(default=30, description="Frames per second")
    
    # Scene configuration
    scenes: List[Dict[str, Any]] = Field(default_factory=list)
    
    # Assets
    assets: Dict[str, Any] = Field(default_factory=dict)
    
    # Customization options
    customizable_elements: List[str] = Field(default_factory=list)
    color_scheme: Dict[str, str] = Field(default_factory=dict)
    
    # Metadata
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    created_by: Optional[str] = None
    tags: List[str] = Field(default_factory=list)
    is_premium: bool = False
    usage_count: int = 0


class TemplatePreset(BaseModel):
    """Predefined settings for quick template application"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    template_id: str
    settings: Dict[str, Any] = Field(default_factory=dict)
    description: str = ""
    thumbnail_url: Optional[str] = None


class TemplatesSystem:
    """Advanced templates and presets management system"""
    
    def __init__(self, templates_dir: str = "templates"):
        self.templates_dir = Path(templates_dir)
        self.templates_dir.mkdir(exist_ok=True)
        
        self.templates: Dict[str, VideoTemplate] = {}
        self.presets: Dict[str, TemplatePreset] = {}
        
        # Initialize with built-in templates
        self._initialize_builtin_templates()
    
    def _initialize_builtin_templates(self):
        """Initialize system with built-in professional templates"""
        builtin_templates = [
            self._create_marketing_template(),
            self._create_education_template(),
            self._create_social_media_template(),
            self._create_corporate_template(),
            self._create_tutorial_template()
        ]
        
        for template in builtin_templates:
            self.templates[template.id] = template
    
    def _create_marketing_template(self) -> VideoTemplate:
        """Create a professional marketing video template"""
        return VideoTemplate(
            name="Professional Marketing",
            description="High-impact marketing video with animated text and transitions",
            category=TemplateCategory.MARKETING,
            style=TemplateStyle.DYNAMIC,
            duration=60.0,
            scenes=[
                {
                    "type": "intro",
                    "duration": 3.0,
                    "background": {"type": "gradient", "colors": ["#667eea", "#764ba2"]},
                    "text": {"content": "{{title}}", "animation": "slide_up", "font_size": 48},
                    "music": True
                },
                {
                    "type": "content",
                    "duration": 45.0,
                    "background": {"type": "image", "source": "{{background_image}}"},
                    "text": {"content": "{{description}}", "animation": "fade_in", "font_size": 24},
                    "overlay": {"type": "logo", "position": "bottom_right"}
                },
                {
                    "type": "cta",
                    "duration": 12.0,
                    "background": {"type": "solid", "color": "#6366f1"},
                    "text": {"content": "{{call_to_action}}", "animation": "zoom_in", "font_size": 36},
                    "button": {"text": "Learn More", "style": "primary"}
                }
            ],
            customizable_elements=["title", "description", "call_to_action", "background_image", "logo"],
            color_scheme={
                "primary": "#6366f1",
                "secondary": "#8b5cf6",
                "accent": "#10b981",
                "text": "#ffffff",
                "background": "#1f2937"
            },
            tags=["marketing", "business", "promotional", "corporate"]
        )
    
    def _create_education_template(self) -> VideoTemplate:
        """Create an educational video template"""
        return VideoTemplate(
            name="Educational Course",
            description="Clean educational template with clear typography and structured layout",
            category=TemplateCategory.EDUCATION,
            style=TemplateStyle.PROFESSIONAL,
            duration=180.0,
            scenes=[
                {
                    "type": "title",
                    "duration": 5.0,
                    "background": {"type": "solid", "color": "#f8fafc"},
                    "text": {"content": "{{course_title}}", "animation": "fade_in", "font_size": 42},
                    "subtitle": {"content": "{{instructor}}", "font_size": 18}
                },
                {
                    "type": "chapter",
                    "duration": 150.0,
                    "background": {"type": "split", "left_color": "#ffffff", "right_image": "{{diagram}}"},
                    "text": {"content": "{{chapter_content}}", "position": "left", "font_size": 20},
                    "diagram": {"position": "right", "animation": "draw"}
                },
                {
                    "type": "summary",
                    "duration": 25.0,
                    "background": {"type": "gradient", "colors": ["#e0f2fe", "#b3e5fc"]},
                    "text": {"content": "{{key_points}}", "animation": "list_appear", "font_size": 22}
                }
            ],
            customizable_elements=["course_title", "instructor", "chapter_content", "key_points", "diagram"],
            color_scheme={
                "primary": "#2563eb",
                "secondary": "#0ea5e9",
                "accent": "#059669",
                "text": "#1f2937",
                "background": "#ffffff"
            },
            tags=["education", "learning", "course", "tutorial"]
        )
    
    def _create_social_media_template(self) -> VideoTemplate:
        """Create a social media optimized template"""
        return VideoTemplate(
            name="Social Media Vertical",
            description="Vertical video template optimized for social platforms",
            category=TemplateCategory.SOCIAL_MEDIA,
            style=TemplateStyle.CREATIVE,
            duration=15.0,
            aspect_ratio="9:16",
            resolution="1080x1920",
            scenes=[
                {
                    "type": "hook",
                    "duration": 3.0,
                    "background": {"type": "video", "source": "{{hook_video}}"},
                    "text": {"content": "{{hook_text}}", "animation": "bounce", "font_size": 32},
                    "effects": ["zoom", "glow"]
                },
                {
                    "type": "content",
                    "duration": 10.0,
                    "background": {"type": "image", "source": "{{content_image}}", "filter": "vibrant"},
                    "text": {"content": "{{main_text}}", "animation": "typewriter", "font_size": 24},
                    "stickers": [{"type": "emoji", "content": "{{emoji}}", "animation": "pulse"}]
                },
                {
                    "type": "cta",
                    "duration": 2.0,
                    "background": {"type": "solid", "color": "#ff6b6b"},
                    "text": {"content": "Follow for more!", "animation": "shake", "font_size": 28}
                }
            ],
            customizable_elements=["hook_text", "main_text", "hook_video", "content_image", "emoji"],
            color_scheme={
                "primary": "#ff6b6b",
                "secondary": "#4ecdc4",
                "accent": "#ffe66d",
                "text": "#ffffff",
                "background": "#2c3e50"
            },
            tags=["social", "vertical", "instagram", "tiktok", "mobile"]
        )
    
    def _create_corporate_template(self) -> VideoTemplate:
        """Create a corporate presentation template"""
        return VideoTemplate(
            name="Corporate Presentation",
            description="Professional corporate template with clean design and data visualization",
            category=TemplateCategory.CORPORATE,
            style=TemplateStyle.PROFESSIONAL,
            duration=120.0,
            scenes=[
                {
                    "type": "title_slide",
                    "duration": 8.0,
                    "background": {"type": "gradient", "colors": ["#1e3a8a", "#1e40af"]},
                    "text": {"content": "{{presentation_title}}", "animation": "slide_left", "font_size": 40},
                    "logo": {"position": "top_right", "size": "medium"}
                },
                {
                    "type": "content_slide",
                    "duration": 90.0,
                    "background": {"type": "solid", "color": "#ffffff"},
                    "title": {"content": "{{slide_title}}", "font_size": 32, "color": "#1f2937"},
                    "content": {"content": "{{slide_content}}", "font_size": 18, "color": "#4b5563"},
                    "chart": {"type": "{{chart_type}}", "data": "{{chart_data}}", "position": "right"}
                },
                {
                    "type": "closing",
                    "duration": 22.0,
                    "background": {"type": "image", "source": "{{company_image}}"},
                    "text": {"content": "Thank You", "animation": "fade_in", "font_size": 48},
                    "contact": {"content": "{{contact_info}}", "position": "bottom", "font_size": 16}
                }
            ],
            customizable_elements=["presentation_title", "slide_title", "slide_content", "chart_type", "chart_data", "company_image", "contact_info"],
            color_scheme={
                "primary": "#1e40af",
                "secondary": "#3b82f6",
                "accent": "#059669",
                "text": "#1f2937",
                "background": "#ffffff"
            },
            tags=["corporate", "business", "presentation", "professional"]
        )
    
    def _create_tutorial_template(self) -> VideoTemplate:
        """Create a step-by-step tutorial template"""
        return VideoTemplate(
            name="Step-by-Step Tutorial",
            description="Clear tutorial template with numbered steps and visual guides",
            category=TemplateCategory.TUTORIAL,
            style=TemplateStyle.MODERN,
            duration=300.0,
            scenes=[
                {
                    "type": "intro",
                    "duration": 10.0,
                    "background": {"type": "video", "source": "{{intro_video}}"},
                    "text": {"content": "{{tutorial_title}}", "animation": "fade_in", "font_size": 36},
                    "overlay": {"type": "progress_bar", "total_steps": "{{total_steps}}"}
                },
                {
                    "type": "step",
                    "duration": 280.0,
                    "background": {"type": "screen_record", "source": "{{screen_recording}}"},
                    "step_number": {"content": "Step {{step_num}}", "position": "top_left", "font_size": 24},
                    "instruction": {"content": "{{step_instruction}}", "position": "bottom", "font_size": 18},
                    "highlight": {"type": "cursor", "animation": "pulse"},
                    "callout": {"type": "arrow", "position": "dynamic"}
                },
                {
                    "type": "conclusion",
                    "duration": 10.0,
                    "background": {"type": "solid", "color": "#10b981"},
                    "text": {"content": "Tutorial Complete!", "animation": "celebration", "font_size": 32},
                    "next_video": {"content": "{{next_tutorial}}", "position": "bottom"}
                }
            ],
            customizable_elements=["tutorial_title", "total_steps", "step_instruction", "screen_recording", "next_tutorial"],
            color_scheme={
                "primary": "#059669",
                "secondary": "#10b981",
                "accent": "#f59e0b",
                "text": "#ffffff",
                "background": "#1f2937"
            },
            tags=["tutorial", "howto", "education", "step-by-step"]
        )
    
    async def get_templates(
        self,
        category: Optional[TemplateCategory] = None,
        style: Optional[TemplateStyle] = None,
        is_premium: Optional[bool] = None
    ) -> List[VideoTemplate]:
        """Get templates with optional filtering"""
        templates = list(self.templates.values())
        
        if category:
            templates = [t for t in templates if t.category == category]
        if style:
            templates = [t for t in templates if t.style == style]
        if is_premium is not None:
            templates = [t for t in templates if t.is_premium == is_premium]
        
        return sorted(templates, key=lambda t: (t.usage_count, t.created_at), reverse=True)
    
    async def get_template(self, template_id: str) -> Optional[VideoTemplate]:
        """Get a specific template by ID"""
        return self.templates.get(template_id)
    
    async def create_custom_template(
        self,
        name: str,
        description: str,
        category: TemplateCategory,
        style: TemplateStyle,
        scenes: List[Dict[str, Any]],
        **kwargs
    ) -> VideoTemplate:
        """Create a new custom template"""
        template = VideoTemplate(
            name=name,
            description=description,
            category=category,
            style=style,
            scenes=scenes,
            **kwargs
        )
        
        self.templates[template.id] = template
        await self._save_template(template)
        
        return template
    
    async def customize_template(
        self,
        template_id: str,
        customizations: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Apply customizations to a template and return rendering configuration"""
        template = await self.get_template(template_id)
        if not template:
            raise ValueError(f"Template {template_id} not found")
        
        # Create a copy of the template with customizations applied
        customized_config = {
            "template_id": template_id,
            "template_name": template.name,
            "scenes": [],
            "global_settings": {
                "duration": template.duration,
                "aspect_ratio": template.aspect_ratio,
                "resolution": template.resolution,
                "fps": template.fps,
                "color_scheme": template.color_scheme
            }
        }
        
        # Apply customizations to each scene
        for scene in template.scenes:
            customized_scene = scene.copy()
            
            # Replace template variables with custom values
            for key, value in customizations.items():
                if isinstance(value, str):
                    customized_scene = self._replace_template_variables(
                        customized_scene, f"{{{{{key}}}}}", value
                    )
            
            customized_config["scenes"].append(customized_scene)
        
        # Update template usage count
        template.usage_count += 1
        
        return customized_config
    
    def _replace_template_variables(self, obj: Any, template_var: str, value: str) -> Any:
        """Recursively replace template variables in configuration"""
        if isinstance(obj, dict):
            return {k: self._replace_template_variables(v, template_var, value) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._replace_template_variables(item, template_var, value) for item in obj]
        elif isinstance(obj, str):
            return obj.replace(template_var, value)
        else:
            return obj
    
    async def save_preset(
        self,
        name: str,
        template_id: str,
        settings: Dict[str, Any],
        description: str = ""
    ) -> TemplatePreset:
        """Save a customized template as a preset"""
        preset = TemplatePreset(
            name=name,
            template_id=template_id,
            settings=settings,
            description=description
        )
        
        self.presets[preset.id] = preset
        await self._save_preset(preset)
        
        return preset
    
    async def get_presets(self, template_id: Optional[str] = None) -> List[TemplatePreset]:
        """Get presets, optionally filtered by template"""
        presets = list(self.presets.values())
        
        if template_id:
            presets = [p for p in presets if p.template_id == template_id]
        
        return presets
    
    async def apply_preset(self, preset_id: str) -> Dict[str, Any]:
        """Apply a saved preset to generate video configuration"""
        preset = self.presets.get(preset_id)
        if not preset:
            raise ValueError(f"Preset {preset_id} not found")
        
        return await self.customize_template(preset.template_id, preset.settings)
    
    async def _save_template(self, template: VideoTemplate):
        """Save template to disk"""
        template_file = self.templates_dir / f"template_{template.id}.json"
        with open(template_file, 'w', encoding='utf-8') as f:
            json.dump(template.dict(), f, indent=2, ensure_ascii=False, default=str)
    
    async def _save_preset(self, preset: TemplatePreset):
        """Save preset to disk"""
        preset_file = self.templates_dir / f"preset_{preset.id}.json"
        with open(preset_file, 'w', encoding='utf-8') as f:
            json.dump(preset.dict(), f, indent=2, ensure_ascii=False, default=str)
    
    async def load_templates_from_disk(self):
        """Load saved templates and presets from disk"""
        if not self.templates_dir.exists():
            return
        
        # Load templates
        for template_file in self.templates_dir.glob("template_*.json"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    template = VideoTemplate(**data)
                    self.templates[template.id] = template
            except Exception as e:
                print(f"Error loading template from {template_file}: {e}")
        
        # Load presets
        for preset_file in self.templates_dir.glob("preset_*.json"):
            try:
                with open(preset_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    preset = TemplatePreset(**data)
                    self.presets[preset.id] = preset
            except Exception as e:
                print(f"Error loading preset from {preset_file}: {e}")


# Global instance
_templates_system = None

def get_templates_system() -> TemplatesSystem:
    """Get the global templates system instance"""
    global _templates_system
    if _templates_system is None:
        _templates_system = TemplatesSystem()
    return _templates_system