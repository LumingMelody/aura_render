"""
Main Video Processing Pipeline

Orchestrates the complete video generation workflow including:
- Content analysis and planning
- Material matching and acquisition
- Video composition and rendering
- Quality validation and optimization
"""

import asyncio
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
import json

# Comment out unavailable imports to fix import error
# Import real VGP nodes
from video_generate_protocol.nodes.video_type_identification_node import VideoTypeIdentificationNode
from video_generate_protocol.nodes.emotion_analysis_node import EmotionAnalysisNode
from video_generate_protocol.nodes.shot_block_generation_node import ShotBlockGenerationNode

# Import mock nodes for unavailable components
try:
    from nodes.effects_processor import EffectsProcessor
except ImportError:
    EffectsProcessor = None
try:
    from nodes.audio_processor import AudioProcessor
except ImportError:
    AudioProcessor = None
try:
    from nodes.subtitle_generator import SubtitleGenerator
except ImportError:
    SubtitleGenerator = None
try:
    from nodes.render_compositor import RenderCompositor
except ImportError:
    RenderCompositor = None
from render_engine.render_manager import RenderManager
from materials_supplies.matcher.main_video_matcher import MainVideoMatcher

logger = logging.getLogger(__name__)


@dataclass
class PipelineConfig:
    """Pipeline configuration"""
    quality: str = "standard"  # low, standard, high, ultra
    format: str = "mp4"
    resolution: str = "1920x1080"
    fps: int = 30
    bitrate: int = 2000
    audio_quality: str = "standard"
    enable_subtitles: bool = True
    enable_effects: bool = True
    max_duration: int = 300
    output_path: str = "./outputs"
    temp_path: str = "./temp"
    
    # Advanced options
    enable_optimization: bool = True
    enable_caching: bool = True
    parallel_processing: bool = True
    debug_mode: bool = False


@dataclass
class PipelineResult:
    """Pipeline execution result"""
    success: bool
    output_path: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    analytics: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    processing_time: float = 0.0
    stages_completed: List[str] = field(default_factory=list)


class PipelineStage:
    """Base class for pipeline stages"""
    
    def __init__(self, name: str, description: str):
        self.name = name
        self.description = description
        self.start_time = None
        self.end_time = None
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Execute this stage"""
        raise NotImplementedError
        
    def get_progress(self) -> float:
        """Get stage progress (0.0 to 1.0)"""
        return 1.0 if self.end_time else 0.0


class ContentAnalysisStage(PipelineStage):
    """Content analysis and planning stage"""
    
    def __init__(self):
        super().__init__("content_analysis", "Analyzing content and generating plan")
        self.video_type_node = VideoTypeIdentificationNode()
        self.emotion_node = EmotionAnalysisNode()
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze input content and create generation plan"""
        self.start_time = datetime.now()
        
        try:
            user_input = context.get("user_input", {})
            text_content = user_input.get("text", "")
            
            # Identify video type
            video_type_result = await self.video_type_node.process({"text": text_content})
            
            # Analyze emotions
            emotion_result = await self.emotion_node.process({"text": text_content})
            
            # Generate analysis summary
            analysis = {
                "video_type": video_type_result.get("video_type", "promotional"),
                "emotions": emotion_result.get("emotions", []),
                "themes": emotion_result.get("themes", []),
                "tone": emotion_result.get("tone", "neutral"),
                "complexity_score": self._calculate_complexity(text_content),
                "duration_estimate": self._estimate_duration(text_content),
                "content_structure": self._analyze_structure(text_content)
            }
            
            self.end_time = datetime.now()
            
            return {
                "analysis": analysis,
                "stage_result": "success"
            }
            
        except Exception as e:
            logger.error(f"Content analysis failed: {e}")
            return {
                "stage_result": "error",
                "error": str(e)
            }
    
    def _calculate_complexity(self, text: str) -> float:
        """Calculate content complexity score"""
        if not text:
            return 0.0
        
        # Simple complexity scoring based on text characteristics
        factors = {
            "length": min(len(text) / 1000, 1.0) * 0.3,
            "sentences": min(len(text.split('.')) / 10, 1.0) * 0.2,
            "unique_words": min(len(set(text.lower().split())) / 100, 1.0) * 0.3,
            "technical_terms": len([w for w in text.split() if len(w) > 8]) / len(text.split()) * 0.2
        }
        
        return sum(factors.values())
    
    def _estimate_duration(self, text: str) -> float:
        """Estimate video duration based on content"""
        if not text:
            return 10.0
        
        # Rough estimation: ~150 words per minute for narration
        word_count = len(text.split())
        base_duration = (word_count / 150) * 60  # Convert to seconds
        
        # Add buffer for visuals and effects
        return max(base_duration * 1.5, 10.0)
    
    def _analyze_structure(self, text: str) -> Dict[str, Any]:
        """Analyze content structure"""
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        return {
            "sentence_count": len(sentences),
            "has_introduction": bool(sentences and any(word in sentences[0].lower() 
                                                     for word in ["hello", "welcome", "today", "let"])),
            "has_conclusion": bool(sentences and any(word in sentences[-1].lower() 
                                                   for word in ["thank", "conclusion", "finally", "end"])),
            "key_sections": self._identify_sections(sentences)
        }
    
    def _identify_sections(self, sentences: List[str]) -> List[Dict[str, Any]]:
        """Identify key sections in content"""
        sections = []
        current_section = []
        
        for i, sentence in enumerate(sentences):
            current_section.append(sentence)
            
            # Simple section break detection
            if (i > 0 and len(current_section) >= 2 and 
                any(word in sentence.lower() for word in ["next", "now", "then", "also", "furthermore"])):
                
                sections.append({
                    "content": ". ".join(current_section[:-1]),
                    "start_sentence": len(sections) * 2,
                    "sentence_count": len(current_section) - 1
                })
                current_section = [sentence]
        
        if current_section:
            sections.append({
                "content": ". ".join(current_section),
                "start_sentence": len(sections) * 2,
                "sentence_count": len(current_section)
            })
        
        return sections


class ShotGenerationStage(PipelineStage):
    """Shot block generation stage"""
    
    def __init__(self):
        super().__init__("shot_generation", "Generating shot blocks and timeline")
        self.shot_node = ShotBlockGenerationNode()
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate shot blocks and timeline"""
        self.start_time = datetime.now()
        
        try:
            analysis = context.get("analysis", {})
            user_input = context.get("user_input", {})
            
            # Generate shot blocks
            shot_input = {
                "text": user_input.get("text", ""),
                "video_type": analysis.get("video_type", "promotional"),
                "emotions": analysis.get("emotions", []),
                "duration_estimate": analysis.get("duration_estimate", 30)
            }
            
            shot_result = await self.shot_node.process(shot_input)
            
            self.end_time = datetime.now()
            
            return {
                "shot_blocks": shot_result.get("shot_blocks", []),
                "timeline": shot_result.get("timeline", {}),
                "stage_result": "success"
            }
            
        except Exception as e:
            logger.error(f"Shot generation failed: {e}")
            return {
                "stage_result": "error",
                "error": str(e)
            }


class MaterialMatchingStage(PipelineStage):
    """Material matching and acquisition stage"""
    
    def __init__(self):
        super().__init__("material_matching", "Matching and acquiring materials")
        self.video_matcher = MainVideoMatcher()
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Match and acquire materials for shot blocks"""
        self.start_time = datetime.now()
        
        try:
            shot_blocks = context.get("shot_blocks", [])
            analysis = context.get("analysis", {})
            
            materials = []
            
            for i, shot_block in enumerate(shot_blocks):
                # Generate search query for this shot block
                query = self._generate_search_query(shot_block, analysis)
                
                # Match materials
                matches = await self.video_matcher.find_matches({
                    "query": query,
                    "video_type": analysis.get("video_type", "promotional"),
                    "duration": shot_block.get("duration", 5.0),
                    "shot_id": shot_block.get("id", f"shot_{i}")
                })
                
                materials.extend(matches.get("materials", []))
            
            self.end_time = datetime.now()
            
            return {
                "materials": materials,
                "material_count": len(materials),
                "stage_result": "success"
            }
            
        except Exception as e:
            logger.error(f"Material matching failed: {e}")
            return {
                "stage_result": "error",
                "error": str(e)
            }
    
    def _generate_search_query(self, shot_block: Dict[str, Any], analysis: Dict[str, Any]) -> str:
        """Generate search query for shot block"""
        keywords = []
        
        # Add shot block description
        if "description" in shot_block:
            keywords.append(shot_block["description"])
        
        # Add themes from analysis
        keywords.extend(analysis.get("themes", []))
        
        # Add video type context
        video_type = analysis.get("video_type", "promotional")
        if video_type != "generic":
            keywords.append(video_type)
        
        return " ".join(keywords)


class VideoCompositionStage(PipelineStage):
    """Video composition and effects stage"""
    
    def __init__(self):
        super().__init__("video_composition", "Composing video with effects")
        self.effects_processor = EffectsProcessor() if EffectsProcessor else None
        self.audio_processor = AudioProcessor() if AudioProcessor else None
        self.subtitle_generator = SubtitleGenerator() if SubtitleGenerator else None
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Compose video with materials, audio, and effects"""
        self.start_time = datetime.now()
        
        try:
            materials = context.get("materials", [])
            shot_blocks = context.get("shot_blocks", [])
            timeline = context.get("timeline", {})
            user_input = context.get("user_input", {})
            config = context.get("config", PipelineConfig())
            
            # Process audio
            audio_result = None
            if self.audio_processor and user_input.get("text") and config.enable_subtitles:
                audio_result = await self.audio_processor.process_tts({
                    "text": user_input["text"],
                    "voice": user_input.get("voice_settings", {}),
                    "output_path": f"{config.temp_path}/audio.wav"
                })
            
            # Generate subtitles
            subtitle_result = None
            if self.subtitle_generator and config.enable_subtitles and user_input.get("text"):
                subtitle_result = await self.subtitle_generator.generate_subtitles({
                    "text": user_input["text"],
                    "timeline": timeline,
                    "style": user_input.get("subtitle_style", {})
                })
            
            # Apply effects
            effects_result = None
            if self.effects_processor and config.enable_effects:
                effects_result = await self.effects_processor.apply_shot_effects({
                    "shot_blocks": shot_blocks,
                    "materials": materials,
                    "style": user_input.get("visual_style", {})
                })
            
            self.end_time = datetime.now()
            
            return {
                "audio": audio_result,
                "subtitles": subtitle_result,
                "effects": effects_result,
                "composition_ready": True,
                "stage_result": "success"
            }
            
        except Exception as e:
            logger.error(f"Video composition failed: {e}")
            return {
                "stage_result": "error",
                "error": str(e)
            }


class RenderingStage(PipelineStage):
    """Final rendering stage"""
    
    def __init__(self):
        super().__init__("rendering", "Rendering final video")
        self.render_manager = RenderManager()
        self.compositor = RenderCompositor() if RenderCompositor else None
        
    async def execute(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Render final video output"""
        self.start_time = datetime.now()
        
        try:
            config = context.get("config", PipelineConfig())
            
            # Prepare render configuration
            render_config = {
                "quality": config.quality,
                "format": config.format,
                "resolution": config.resolution,
                "fps": config.fps,
                "bitrate": config.bitrate,
                "output_path": config.output_path,
                "temp_path": config.temp_path
            }
            
            # Compose final video
            composition_result = await self.compositor.compose_final_video({
                "shot_blocks": context.get("shot_blocks", []),
                "materials": context.get("materials", []),
                "audio": context.get("audio"),
                "subtitles": context.get("subtitles"),
                "effects": context.get("effects"),
                "timeline": context.get("timeline", {}),
                "config": render_config
            })
            
            # Render video
            if composition_result.get("success"):
                render_result = await self.render_manager.render_video({
                    "composition": composition_result["composition"],
                    "config": render_config,
                    "metadata": {
                        "generated_at": datetime.now().isoformat(),
                        "pipeline_version": "1.0.0",
                        "user_input": context.get("user_input", {}),
                        "analysis": context.get("analysis", {})
                    }
                })
                
                self.end_time = datetime.now()
                
                return {
                    "output_path": render_result.get("output_path"),
                    "metadata": render_result.get("metadata", {}),
                    "render_stats": render_result.get("stats", {}),
                    "stage_result": "success"
                }
            else:
                return {
                    "stage_result": "error",
                    "error": "Composition failed"
                }
                
        except Exception as e:
            logger.error(f"Rendering failed: {e}")
            return {
                "stage_result": "error",
                "error": str(e)
            }


class MainPipeline:
    """Main video generation pipeline"""
    
    def __init__(self, config: Optional[PipelineConfig] = None):
        self.config = config or PipelineConfig()
        self.stages = [
            ContentAnalysisStage(),
            ShotGenerationStage(),
            MaterialMatchingStage(),
            VideoCompositionStage(),
            RenderingStage()
        ]
        self.progress_callback: Optional[Callable] = None
        self.context: Dict[str, Any] = {}
        
    def set_progress_callback(self, callback: Callable[[str, float, Dict[str, Any]], None]):
        """Set callback for progress updates"""
        self.progress_callback = callback
        
    async def execute(self, user_input: Dict[str, Any]) -> PipelineResult:
        """Execute the complete pipeline"""
        start_time = datetime.now()
        result = PipelineResult(success=False)
        
        try:
            logger.info("Starting main video generation pipeline")
            
            # Initialize context
            self.context = {
                "user_input": user_input,
                "config": self.config,
                "start_time": start_time
            }
            
            # Execute stages
            for i, stage in enumerate(self.stages):
                stage_progress = i / len(self.stages)
                
                if self.progress_callback:
                    self.progress_callback(stage.name, stage_progress, {
                        "stage": stage.name,
                        "description": stage.description
                    })
                
                # Execute stage
                stage_result = await stage.execute(self.context)
                
                # Check for errors
                if stage_result.get("stage_result") == "error":
                    error_msg = f"Stage {stage.name} failed: {stage_result.get('error', 'Unknown error')}"
                    logger.error(error_msg)
                    result.errors.append(error_msg)
                    return result
                
                # Update context with stage results
                self.context.update(stage_result)
                result.stages_completed.append(stage.name)
                
                logger.info(f"Stage {stage.name} completed successfully")
            
            # Pipeline completed successfully
            end_time = datetime.now()
            result.success = True
            result.processing_time = (end_time - start_time).total_seconds()
            result.output_path = self.context.get("output_path")
            result.metadata = {
                "pipeline_version": "1.0.0",
                "config": self.config.__dict__,
                "stages": [stage.name for stage in self.stages],
                "start_time": start_time.isoformat(),
                "end_time": end_time.isoformat()
            }
            
            # Collect analytics
            result.analytics = {
                "total_processing_time": result.processing_time,
                "stage_times": {
                    stage.name: (stage.end_time - stage.start_time).total_seconds() 
                    if stage.start_time and stage.end_time else 0
                    for stage in self.stages
                },
                "material_count": len(self.context.get("materials", [])),
                "shot_count": len(self.context.get("shot_blocks", [])),
                "video_duration": self.context.get("analysis", {}).get("duration_estimate", 0)
            }
            
            if self.progress_callback:
                self.progress_callback("completed", 1.0, {
                    "stage": "completed",
                    "description": "Pipeline completed successfully",
                    "output_path": result.output_path
                })
            
            logger.info(f"Pipeline completed successfully in {result.processing_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Pipeline execution failed: {e}")
            result.errors.append(str(e))
            
        return result
    
    def get_stage_progress(self) -> Dict[str, float]:
        """Get progress for each stage"""
        return {
            stage.name: stage.get_progress()
            for stage in self.stages
        }
    
    def validate_input(self, user_input: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate user input"""
        errors = []
        
        # Check required fields
        if not user_input.get("text"):
            errors.append("Text content is required")
        
        # Check text length
        text = user_input.get("text", "")
        if len(text) > 10000:
            errors.append("Text content is too long (max 10,000 characters)")
        elif len(text) < 10:
            errors.append("Text content is too short (min 10 characters)")
        
        # Validate configuration
        if "config" in user_input:
            config_errors = self._validate_config(user_input["config"])
            errors.extend(config_errors)
        
        return len(errors) == 0, errors
    
    def _validate_config(self, config: Dict[str, Any]) -> List[str]:
        """Validate configuration parameters"""
        errors = []
        
        # Check video quality
        valid_qualities = ["low", "standard", "high", "ultra"]
        if config.get("quality") and config["quality"] not in valid_qualities:
            errors.append(f"Invalid quality: {config['quality']}. Must be one of {valid_qualities}")
        
        # Check format
        valid_formats = ["mp4", "avi", "mov", "mkv"]
        if config.get("format") and config["format"] not in valid_formats:
            errors.append(f"Invalid format: {config['format']}. Must be one of {valid_formats}")
        
        # Check duration
        if config.get("max_duration") and config["max_duration"] > 600:
            errors.append("Maximum duration cannot exceed 600 seconds")
        
        return errors


# Factory function for easy instantiation
def create_pipeline(config: Optional[Dict[str, Any]] = None) -> MainPipeline:
    """Create a new pipeline instance"""
    pipeline_config = PipelineConfig()
    
    if config:
        for key, value in config.items():
            if hasattr(pipeline_config, key):
                setattr(pipeline_config, key, value)
    
    return MainPipeline(pipeline_config)


# Async context manager for pipeline execution
class PipelineRunner:
    """Context manager for pipeline execution"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        self.pipeline = create_pipeline(config)
        self.result: Optional[PipelineResult] = None
        
    async def __aenter__(self):
        return self.pipeline
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        # Clean up temp files if needed
        if self.pipeline.config.temp_path:
            temp_path = Path(self.pipeline.config.temp_path)
            if temp_path.exists():
                try:
                    import shutil
                    shutil.rmtree(temp_path)
                    logger.info(f"Cleaned up temp directory: {temp_path}")
                except Exception as e:
                    logger.warning(f"Failed to clean up temp directory: {e}")