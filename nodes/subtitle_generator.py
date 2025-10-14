"""
Subtitle Generator Node

Handles subtitle generation, timing synchronization, and styling
for video content including multi-language support and accessibility features.
"""

import asyncio
import json
import re
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import timedelta

from .base_node import BaseNode, NodeConfig, NodeResult, ProcessingContext, NodeStatus
from ai_service import get_enhanced_qwen_service, get_prompt_manager


@dataclass
class SubtitleConfig(NodeConfig):
    """Configuration for subtitle generation node"""
    default_language: str = "zh-CN"
    max_chars_per_line: int = 40
    max_lines_per_subtitle: int = 2
    min_duration: float = 0.8  # Minimum subtitle duration in seconds
    max_duration: float = 5.0  # Maximum subtitle duration in seconds
    reading_speed_wpm: int = 150  # Words per minute reading speed
    line_break_strategy: str = "smart"  # Options: smart, word_wrap, manual
    font_family: str = "Arial"
    font_size: int = 24
    font_color: str = "#FFFFFF"
    background_color: str = "#000000"
    background_opacity: float = 0.7
    position: str = "bottom"  # Options: top, center, bottom
    enable_animations: bool = True
    enable_multilingual: bool = False
    supported_languages: List[str] = field(default_factory=lambda: ["zh-CN", "en-US"])


class SubtitleGeneratorNode(BaseNode):
    """Subtitle generator node for video generation pipeline"""
    
    def __init__(self, config: SubtitleConfig):
        super().__init__(config)
        self.config: SubtitleConfig = config
        self.ai_service = get_enhanced_qwen_service()
        self.prompt_manager = get_prompt_manager()
        
    def get_required_inputs(self) -> List[str]:
        """Required inputs for subtitle generation"""
        return ['voice_audio_path', 'audio_segments', 'script_content']
    
    def get_output_schema(self) -> Dict[str, Any]:
        """Output schema for subtitle generation"""
        return {
            'subtitle_tracks': 'list',
            'subtitle_files': 'dict',
            'timing_data': 'list',
            'styling_config': 'dict',
            'subtitle_metadata': 'dict'
        }
    
    def validate_input(self, context: ProcessingContext) -> bool:
        """Validate input for subtitle generation"""
        required_keys = ['audio_segments', 'script_content']
        
        for key in required_keys:
            if key not in context.intermediate_results:
                self.logger.error(f"Missing required input: {key}")
                return False
        
        # Validate audio segments structure
        audio_segments = context.intermediate_results.get('audio_segments')
        if not isinstance(audio_segments, list):
            self.logger.error("audio_segments must be a list")
            return False
            
        return True
    
    async def process(self, context: ProcessingContext) -> NodeResult:
        """Process subtitle generation"""
        try:
            self.logger.info("Starting subtitle generation")
            
            # Extract input data
            audio_segments = context.intermediate_results['audio_segments']
            script_content = context.intermediate_results['script_content']
            voice_audio_path = context.intermediate_results.get('voice_audio_path')
            
            # Step 1: Extract and prepare text content
            text_segments = await self._extract_text_segments(
                script_content, audio_segments, context
            )
            
            # Step 2: Generate timing and synchronization
            timed_subtitles = await self._generate_subtitle_timing(
                text_segments, audio_segments, context
            )
            
            # Step 3: Apply text formatting and line breaks
            formatted_subtitles = await self._format_subtitle_text(
                timed_subtitles, context
            )
            
            # Step 4: Generate multilingual subtitles if enabled
            multilingual_subtitles = await self._generate_multilingual_subtitles(
                formatted_subtitles, context
            )
            
            # Step 5: Create subtitle files and styling
            subtitle_output = await self._generate_subtitle_files(
                multilingual_subtitles, context
            )
            
            # Prepare result
            subtitle_data = {
                'subtitle_tracks': subtitle_output.get('tracks', []),
                'subtitle_files': subtitle_output.get('files', {}),
                'timing_data': [sub['timing'] for sub in formatted_subtitles],
                'styling_config': self._get_styling_config(),
                'subtitle_metadata': {
                    'total_subtitles': len(formatted_subtitles),
                    'languages': self.config.supported_languages if self.config.enable_multilingual else [self.config.default_language],
                    'average_duration': sum(sub['duration'] for sub in formatted_subtitles) / len(formatted_subtitles) if formatted_subtitles else 0,
                    'total_characters': sum(len(sub['text']) for sub in formatted_subtitles),
                    'reading_speed_wpm': self.config.reading_speed_wpm,
                    'formatting_config': {
                        'max_chars_per_line': self.config.max_chars_per_line,
                        'max_lines': self.config.max_lines_per_subtitle,
                        'line_break_strategy': self.config.line_break_strategy
                    }
                }
            }
            
            return NodeResult(
                status=NodeStatus.COMPLETED,
                data=subtitle_data,
                next_nodes=['effects_processor', 'render_compositor']
            )
            
        except Exception as e:
            self.logger.error(f"Subtitle generation failed: {e}")
            return NodeResult(
                status=NodeStatus.FAILED,
                error_message=str(e)
            )
    
    async def _extract_text_segments(
        self,
        script_content: Any,
        audio_segments: List[Dict[str, Any]],
        context: ProcessingContext
    ) -> List[Dict[str, Any]]:
        """Extract and prepare text segments from script content"""
        self.logger.info("Extracting text segments")
        
        text_segments = []
        
        # Extract text from various script formats
        if isinstance(script_content, list):
            # List of segments
            for i, segment in enumerate(script_content):
                if isinstance(segment, dict):
                    text = segment.get('text', segment.get('content', ''))
                    if text.strip():
                        text_segments.append({
                            'segment_id': f'text_{i+1:03d}',
                            'text': text.strip(),
                            'original_index': i,
                            'metadata': segment.get('metadata', {})
                        })
                else:
                    text = str(segment).strip()
                    if text:
                        text_segments.append({
                            'segment_id': f'text_{i+1:03d}',
                            'text': text,
                            'original_index': i,
                            'metadata': {}
                        })
        
        elif isinstance(script_content, dict):
            # Dictionary format
            if 'segments' in script_content:
                for i, segment in enumerate(script_content['segments']):
                    text = segment.get('text', '')
                    if text.strip():
                        text_segments.append({
                            'segment_id': f'text_{i+1:03d}',
                            'text': text.strip(),
                            'original_index': i,
                            'metadata': segment.get('metadata', {})
                        })
            elif 'full_script' in script_content:
                # Split full script into segments
                sentences = self._split_text_into_sentences(script_content['full_script'])
                for i, sentence in enumerate(sentences):
                    if sentence.strip():
                        text_segments.append({
                            'segment_id': f'text_{i+1:03d}',
                            'text': sentence.strip(),
                            'original_index': i,
                            'metadata': {}
                        })
        
        else:
            # Plain text
            sentences = self._split_text_into_sentences(str(script_content))
            for i, sentence in enumerate(sentences):
                if sentence.strip():
                    text_segments.append({
                        'segment_id': f'text_{i+1:03d}',
                        'text': sentence.strip(),
                        'original_index': i,
                        'metadata': {}
                    })
        
        return text_segments
    
    async def _generate_subtitle_timing(
        self,
        text_segments: List[Dict[str, Any]],
        audio_segments: List[Dict[str, Any]],
        context: ProcessingContext
    ) -> List[Dict[str, Any]]:
        """Generate timing synchronization for subtitles"""
        self.logger.info("Generating subtitle timing")
        
        timed_subtitles = []
        
        # Try to align text segments with audio segments
        if len(text_segments) == len(audio_segments):
            # Direct alignment
            for i, (text_seg, audio_seg) in enumerate(zip(text_segments, audio_segments)):
                subtitle = {
                    'subtitle_id': f'sub_{i+1:03d}',
                    'text': text_seg['text'],
                    'start_time': audio_seg.get('start_time', 0.0),
                    'duration': audio_seg.get('duration', 2.0),
                    'end_time': audio_seg.get('start_time', 0.0) + audio_seg.get('duration', 2.0),
                    'audio_segment_id': audio_seg.get('segment_id'),
                    'text_segment_id': text_seg['segment_id'],
                    'metadata': {**text_seg.get('metadata', {}), **audio_seg.get('metadata', {})}
                }
                timed_subtitles.append(subtitle)
        
        else:
            # Smart alignment based on text length and audio timing
            total_audio_duration = sum(seg.get('duration', 0) for seg in audio_segments)
            total_text_length = sum(len(seg['text']) for seg in text_segments)
            
            current_time = 0.0
            for i, text_seg in enumerate(text_segments):
                # Calculate duration based on text length and reading speed
                word_count = len(text_seg['text'].split())
                estimated_duration = max(
                    self.config.min_duration,
                    min(self.config.max_duration, (word_count / self.config.reading_speed_wpm) * 60)
                )
                
                # Adjust for remaining time
                if i == len(text_segments) - 1 and total_audio_duration > 0:
                    remaining_time = total_audio_duration - current_time
                    if remaining_time > 0:
                        estimated_duration = min(estimated_duration, remaining_time)
                
                subtitle = {
                    'subtitle_id': f'sub_{i+1:03d}',
                    'text': text_seg['text'],
                    'start_time': current_time,
                    'duration': estimated_duration,
                    'end_time': current_time + estimated_duration,
                    'text_segment_id': text_seg['segment_id'],
                    'metadata': text_seg.get('metadata', {})
                }
                
                timed_subtitles.append(subtitle)
                current_time += estimated_duration
        
        return timed_subtitles
    
    async def _format_subtitle_text(
        self,
        timed_subtitles: List[Dict[str, Any]],
        context: ProcessingContext
    ) -> List[Dict[str, Any]]:
        """Format subtitle text with proper line breaks and styling"""
        self.logger.info("Formatting subtitle text")
        
        formatted_subtitles = []
        
        for subtitle in timed_subtitles:
            original_text = subtitle['text']
            
            # Apply line breaking strategy
            if self.config.line_break_strategy == "smart":
                formatted_text = self._smart_line_break(original_text)
            elif self.config.line_break_strategy == "word_wrap":
                formatted_text = self._word_wrap_line_break(original_text)
            else:
                formatted_text = original_text
            
            # Validate formatting constraints
            formatted_text = self._validate_subtitle_format(formatted_text)
            
            formatted_subtitle = {
                **subtitle,
                'text': formatted_text,
                'original_text': original_text,
                'line_count': formatted_text.count('\n') + 1,
                'char_count': len(formatted_text),
                'timing': {
                    'start': subtitle['start_time'],
                    'end': subtitle['end_time'],
                    'duration': subtitle['duration']
                },
                'formatting': {
                    'line_break_applied': formatted_text != original_text,
                    'strategy_used': self.config.line_break_strategy
                }
            }
            
            formatted_subtitles.append(formatted_subtitle)
        
        return formatted_subtitles
    
    async def _generate_multilingual_subtitles(
        self,
        formatted_subtitles: List[Dict[str, Any]],
        context: ProcessingContext
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Generate multilingual subtitle tracks if enabled"""
        multilingual_tracks = {self.config.default_language: formatted_subtitles}
        
        if not self.config.enable_multilingual:
            return multilingual_tracks
        
        self.logger.info("Generating multilingual subtitles")
        
        # Generate subtitles for each supported language
        for lang in self.config.supported_languages:
            if lang != self.config.default_language:
                try:
                    translated_subtitles = await self._translate_subtitles(
                        formatted_subtitles, lang, context
                    )
                    multilingual_tracks[lang] = translated_subtitles
                except Exception as e:
                    self.logger.warning(f"Failed to generate subtitles for {lang}: {e}")
        
        return multilingual_tracks
    
    async def _translate_subtitles(
        self,
        subtitles: List[Dict[str, Any]],
        target_language: str,
        context: ProcessingContext
    ) -> List[Dict[str, Any]]:
        """Translate subtitles to target language using AI"""
        translated_subtitles = []
        
        # Batch translation for efficiency
        texts_to_translate = [sub['text'] for sub in subtitles]
        
        translation_prompt = self.prompt_manager.render_prompt(
            'subtitle_translation',
            {
                'source_language': self.config.default_language,
                'target_language': target_language,
                'texts': texts_to_translate,
                'max_chars_per_line': self.config.max_chars_per_line
            }
        )
        
        if translation_prompt:
            try:
                ai_response = await self.ai_service.generate_content(translation_prompt)
                translations = self._parse_translations(ai_response.content, len(texts_to_translate))
            except Exception as e:
                self.logger.error(f"AI translation failed: {e}")
                translations = texts_to_translate  # Fallback to original text
        else:
            translations = texts_to_translate  # Fallback to original text
        
        # Create translated subtitle objects
        for subtitle, translated_text in zip(subtitles, translations):
            translated_subtitle = {
                **subtitle,
                'text': translated_text,
                'language': target_language,
                'translation_metadata': {
                    'source_language': self.config.default_language,
                    'original_text': subtitle['text']
                }
            }
            translated_subtitles.append(translated_subtitle)
        
        return translated_subtitles
    
    async def _generate_subtitle_files(
        self,
        multilingual_subtitles: Dict[str, List[Dict[str, Any]]],
        context: ProcessingContext
    ) -> Dict[str, Any]:
        """Generate subtitle files in various formats"""
        self.logger.info("Generating subtitle files")
        
        subtitle_files = {}
        subtitle_tracks = []
        
        for language, subtitles in multilingual_subtitles.items():
            # Generate SRT format
            srt_content = self._generate_srt_content(subtitles)
            srt_filename = f"subtitles_{language}_{context.task_id}.srt"
            subtitle_files[f'{language}_srt'] = {
                'filename': srt_filename,
                'format': 'srt',
                'content': srt_content,
                'language': language
            }
            
            # Generate VTT format for web
            vtt_content = self._generate_vtt_content(subtitles)
            vtt_filename = f"subtitles_{language}_{context.task_id}.vtt"
            subtitle_files[f'{language}_vtt'] = {
                'filename': vtt_filename,
                'format': 'vtt',
                'content': vtt_content,
                'language': language
            }
            
            # Create track metadata
            track_info = {
                'language': language,
                'label': self._get_language_label(language),
                'default': language == self.config.default_language,
                'subtitle_count': len(subtitles),
                'files': {
                    'srt': srt_filename,
                    'vtt': vtt_filename
                },
                'styling': self._get_styling_config()
            }
            
            subtitle_tracks.append(track_info)
        
        return {
            'tracks': subtitle_tracks,
            'files': subtitle_files
        }
    
    def _smart_line_break(self, text: str) -> str:
        """Apply smart line breaking for better readability"""
        if len(text) <= self.config.max_chars_per_line:
            return text
        
        # Try to break at natural points
        words = text.split()
        lines = []
        current_line = []
        current_length = 0
        
        for word in words:
            word_length = len(word) + (1 if current_line else 0)  # +1 for space
            
            if current_length + word_length <= self.config.max_chars_per_line:
                current_line.append(word)
                current_length += word_length
            else:
                if current_line:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)
                else:
                    # Word is too long, break it
                    lines.append(word[:self.config.max_chars_per_line])
                    remaining = word[self.config.max_chars_per_line:]
                    while remaining:
                        lines.append(remaining[:self.config.max_chars_per_line])
                        remaining = remaining[self.config.max_chars_per_line:]
        
        if current_line:
            lines.append(' '.join(current_line))
        
        # Limit to max lines
        if len(lines) > self.config.max_lines_per_subtitle:
            lines = lines[:self.config.max_lines_per_subtitle]
        
        return '\n'.join(lines)
    
    def _word_wrap_line_break(self, text: str) -> str:
        """Simple word wrap line breaking"""
        import textwrap
        wrapped = textwrap.fill(
            text,
            width=self.config.max_chars_per_line,
            max_lines=self.config.max_lines_per_subtitle
        )
        return wrapped
    
    def _validate_subtitle_format(self, text: str) -> str:
        """Validate and fix subtitle format constraints"""
        lines = text.split('\n')
        
        # Limit number of lines
        if len(lines) > self.config.max_lines_per_subtitle:
            lines = lines[:self.config.max_lines_per_subtitle]
        
        # Ensure each line doesn't exceed max chars
        validated_lines = []
        for line in lines:
            if len(line) <= self.config.max_chars_per_line:
                validated_lines.append(line)
            else:
                # Truncate with ellipsis
                validated_lines.append(line[:self.config.max_chars_per_line-3] + "...")
        
        return '\n'.join(validated_lines)
    
    def _split_text_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for subtitle processing"""
        # Enhanced sentence splitting with Chinese support
        sentence_endings = r'[。！？；.!?;]'
        sentences = re.split(sentence_endings, text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _generate_srt_content(self, subtitles: List[Dict[str, Any]]) -> str:
        """Generate SRT format subtitle content"""
        srt_content = []
        
        for i, subtitle in enumerate(subtitles, 1):
            start_time = self._format_srt_time(subtitle['start_time'])
            end_time = self._format_srt_time(subtitle['end_time'])
            
            srt_content.extend([
                str(i),
                f"{start_time} --> {end_time}",
                subtitle['text'],
                ""
            ])
        
        return '\n'.join(srt_content)
    
    def _generate_vtt_content(self, subtitles: List[Dict[str, Any]]) -> str:
        """Generate WebVTT format subtitle content"""
        vtt_content = ["WEBVTT", ""]
        
        for subtitle in subtitles:
            start_time = self._format_vtt_time(subtitle['start_time'])
            end_time = self._format_vtt_time(subtitle['end_time'])
            
            vtt_content.extend([
                f"{start_time} --> {end_time}",
                subtitle['text'],
                ""
            ])
        
        return '\n'.join(vtt_content)
    
    def _format_srt_time(self, seconds: float) -> str:
        """Format time for SRT format (HH:MM:SS,mmm)"""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millisecs = int((td.total_seconds() % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millisecs:03d}"
    
    def _format_vtt_time(self, seconds: float) -> str:
        """Format time for VTT format (HH:MM:SS.mmm)"""
        td = timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        secs = int(td.total_seconds() % 60)
        millisecs = int((td.total_seconds() % 1) * 1000)
        
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millisecs:03d}"
    
    def _get_styling_config(self) -> Dict[str, Any]:
        """Get subtitle styling configuration"""
        return {
            'font_family': self.config.font_family,
            'font_size': self.config.font_size,
            'font_color': self.config.font_color,
            'background_color': self.config.background_color,
            'background_opacity': self.config.background_opacity,
            'position': self.config.position,
            'animations_enabled': self.config.enable_animations
        }
    
    def _get_language_label(self, language_code: str) -> str:
        """Get human-readable language label"""
        language_labels = {
            'zh-CN': '中文 (简体)',
            'zh-TW': '中文 (繁體)',
            'en-US': 'English',
            'ja-JP': '日本語',
            'ko-KR': '한국어',
            'es-ES': 'Español',
            'fr-FR': 'Français',
            'de-DE': 'Deutsch'
        }
        
        return language_labels.get(language_code, language_code)
    
    def _parse_translations(self, ai_response: str, expected_count: int) -> List[str]:
        """Parse translations from AI response"""
        try:
            # Try to extract JSON format
            import re
            json_match = re.search(r'\[.*\]', ai_response, re.DOTALL)
            if json_match:
                translations = json.loads(json_match.group())
                if isinstance(translations, list) and len(translations) == expected_count:
                    return translations
        except:
            pass
        
        # Fallback: split by lines/numbers
        lines = ai_response.split('\n')
        translations = []
        for line in lines:
            line = line.strip()
            if line and not line.isdigit():
                # Remove numbering if present
                cleaned = re.sub(r'^\d+[.\)]\s*', '', line)
                if cleaned:
                    translations.append(cleaned)
        
        # Ensure we have the right number of translations
        while len(translations) < expected_count:
            translations.append(f"[Translation {len(translations)+1}]")
        
        return translations[:expected_count]