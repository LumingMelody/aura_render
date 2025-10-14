"""
Text-to-Speech (TTS) Service

Provides TTS capabilities using multiple providers including:
- Azure TTS
- Edge TTS
- OpenAI TTS
- Custom TTS models
"""

import os
import asyncio
import tempfile
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
import hashlib

try:
    import azure.cognitiveservices.speech as speechsdk
    AZURE_TTS_AVAILABLE = True
except ImportError:
    AZURE_TTS_AVAILABLE = False

try:
    import edge_tts
    EDGE_TTS_AVAILABLE = True
except ImportError:
    EDGE_TTS_AVAILABLE = False

from config import Settings
from monitoring import get_error_handler, get_metrics_collector
from monitoring.error_handler import ErrorCategory, ErrorSeverity


class TTSProvider:
    """Base TTS provider interface"""
    
    def __init__(self, name: str, config: Dict[str, Any] = None):
        self.name = name
        self.config = config or {}
        self.logger = logging.getLogger(f"{__name__}.{name}")
        
    async def synthesize(
        self,
        text: str,
        voice: str,
        output_path: str,
        **kwargs
    ) -> bool:
        """Synthesize text to speech"""
        raise NotImplementedError
        
    def get_available_voices(self) -> List[Dict[str, str]]:
        """Get list of available voices"""
        raise NotImplementedError
        
    def is_available(self) -> bool:
        """Check if provider is available"""
        return True


class AzureTTSProvider(TTSProvider):
    """Azure Cognitive Services TTS provider"""
    
    def __init__(self, api_key: str, region: str = "eastus"):
        super().__init__("azure")
        self.api_key = api_key
        self.region = region
        self.speech_config = None
        
        if AZURE_TTS_AVAILABLE and api_key:
            try:
                self.speech_config = speechsdk.SpeechConfig(
                    subscription=api_key, 
                    region=region
                )
            except Exception as e:
                self.logger.error(f"Failed to initialize Azure TTS: {e}")
                
    async def synthesize(
        self,
        text: str,
        voice: str = "zh-CN-XiaoxiaoNeural",
        output_path: str = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%"
    ) -> bool:
        """Synthesize text using Azure TTS"""
        
        if not self.speech_config:
            return False
            
        try:
            # Configure voice and output
            self.speech_config.speech_synthesis_voice_name = voice
            
            if output_path:
                audio_config = speechsdk.audio.AudioOutputConfig(filename=output_path)
            else:
                audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
                
            # Create synthesizer
            synthesizer = speechsdk.SpeechSynthesizer(
                speech_config=self.speech_config,
                audio_config=audio_config
            )
            
            # Build SSML
            ssml = f"""
            <speak version="1.0" xmlns="https://www.w3.org/2001/10/synthesis" xml:lang="zh-CN">
                <voice name="{voice}">
                    <prosody rate="{rate}" pitch="{pitch}" volume="{volume}">
                        {text}
                    </prosody>
                </voice>
            </speak>
            """
            
            # Synthesize
            result = synthesizer.speak_ssml_async(ssml).get()
            
            if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
                self.logger.info(f"Azure TTS synthesis completed: {output_path}")
                return True
            else:
                self.logger.error(f"Azure TTS synthesis failed: {result.reason}")
                return False
                
        except Exception as e:
            self.logger.error(f"Azure TTS error: {e}")
            return False
            
    def get_available_voices(self) -> List[Dict[str, str]]:
        """Get Azure TTS voices"""
        return [
            {"name": "zh-CN-XiaoxiaoNeural", "gender": "Female", "language": "zh-CN"},
            {"name": "zh-CN-YunyangNeural", "gender": "Male", "language": "zh-CN"},
            {"name": "zh-CN-XiaochenNeural", "gender": "Female", "language": "zh-CN"},
            {"name": "zh-CN-XiaohanNeural", "gender": "Female", "language": "zh-CN"},
            {"name": "zh-CN-XiaomengNeural", "gender": "Female", "language": "zh-CN"},
            {"name": "zh-CN-XiaomoNeural", "gender": "Female", "language": "zh-CN"},
            {"name": "zh-CN-XiaoqiuNeural", "gender": "Female", "language": "zh-CN"},
            {"name": "zh-CN-XiaoruiNeural", "gender": "Female", "language": "zh-CN"},
            {"name": "zh-CN-XiaoshuangNeural", "gender": "Female", "language": "zh-CN"},
            {"name": "zh-CN-XiaoxuanNeural", "gender": "Female", "language": "zh-CN"},
            {"name": "zh-CN-XiaoyanNeural", "gender": "Female", "language": "zh-CN"},
            {"name": "zh-CN-XiaoyouNeural", "gender": "Female", "language": "zh-CN"},
            {"name": "zh-CN-YunjianNeural", "gender": "Male", "language": "zh-CN"},
            {"name": "zh-CN-YunxiNeural", "gender": "Male", "language": "zh-CN"},
            {"name": "zh-CN-YunyeNeural", "gender": "Male", "language": "zh-CN"},
            {"name": "zh-CN-YunzeNeural", "gender": "Male", "language": "zh-CN"}
        ]
        
    def is_available(self) -> bool:
        """Check if Azure TTS is available"""
        return AZURE_TTS_AVAILABLE and bool(self.speech_config)


class EdgeTTSProvider(TTSProvider):
    """Microsoft Edge TTS provider (free)"""
    
    def __init__(self):
        super().__init__("edge")
        
    async def synthesize(
        self,
        text: str,
        voice: str = "zh-CN-XiaoxiaoNeural",
        output_path: str = None,
        rate: str = "+0%",
        pitch: str = "+0Hz",
        volume: str = "+0%"
    ) -> bool:
        """Synthesize text using Edge TTS"""
        
        if not EDGE_TTS_AVAILABLE:
            return False
            
        try:
            # Create Edge TTS communicator
            communicate = edge_tts.Communicate(text, voice)
            
            # Generate audio
            if output_path:
                await communicate.save(output_path)
                self.logger.info(f"Edge TTS synthesis completed: {output_path}")
                return True
            else:
                self.logger.error("Output path required for Edge TTS")
                return False
                
        except Exception as e:
            self.logger.error(f"Edge TTS error: {e}")
            return False
            
    def get_available_voices(self) -> List[Dict[str, str]]:
        """Get Edge TTS voices"""
        # Subset of popular Chinese voices
        return [
            {"name": "zh-CN-XiaoxiaoNeural", "gender": "Female", "language": "zh-CN"},
            {"name": "zh-CN-YunyangNeural", "gender": "Male", "language": "zh-CN"},
            {"name": "zh-CN-XiaochenNeural", "gender": "Female", "language": "zh-CN"},
            {"name": "zh-CN-YunxiNeural", "gender": "Male", "language": "zh-CN"},
        ]
        
    def is_available(self) -> bool:
        """Check if Edge TTS is available"""
        return EDGE_TTS_AVAILABLE


class TTSService:
    """Main TTS service that manages multiple providers"""
    
    def __init__(self, settings: Optional[Settings] = None):
        self.settings = settings or Settings()
        self.logger = logging.getLogger(__name__)
        self.error_handler = get_error_handler()
        self.metrics = get_metrics_collector()
        
        self.providers: Dict[str, TTSProvider] = {}
        self._initialize_providers()
        
    def _initialize_providers(self):
        """Initialize available TTS providers"""
        
        # Azure TTS
        if self.settings.azure_api_key:
            provider = AzureTTSProvider(
                api_key=self.settings.azure_api_key,
                region="eastus"  # Could be configurable
            )
            if provider.is_available():
                self.providers["azure"] = provider
                self.logger.info("Azure TTS provider initialized")
                
        # Edge TTS (free)
        provider = EdgeTTSProvider()
        if provider.is_available():
            self.providers["edge"] = provider
            self.logger.info("Edge TTS provider initialized")
            
        if not self.providers:
            self.logger.warning("No TTS providers available")
            
    async def synthesize(
        self,
        text: str,
        voice: str = None,
        provider: str = None,
        output_path: str = None,
        **kwargs
    ) -> Optional[str]:
        """Synthesize text to speech"""
        
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Select provider
            if provider and provider in self.providers:
                selected_provider = self.providers[provider]
            elif self.settings.tts_provider in self.providers:
                selected_provider = self.providers[self.settings.tts_provider]
            else:
                # Use first available provider
                if not self.providers:
                    raise ValueError("No TTS providers available")
                selected_provider = list(self.providers.values())[0]
                
            # Use default voice if not specified
            if not voice:
                voice = self.settings.tts_voice
                
            # Generate output path if not provided
            if not output_path:
                output_dir = self.settings.temp_dir / "tts"
                output_dir.mkdir(exist_ok=True)
                
                # Create hash of text for filename
                text_hash = hashlib.md5(text.encode()).hexdigest()[:8]
                output_path = str(output_dir / f"tts_{text_hash}_{voice}.wav")
                
            # Synthesize
            success = await selected_provider.synthesize(
                text=text,
                voice=voice,
                output_path=output_path,
                **kwargs
            )
            
            if success and os.path.exists(output_path):
                duration = asyncio.get_event_loop().time() - start_time
                
                # Record metrics
                self.metrics.record_ai_service_call(
                    service="tts",
                    model=selected_provider.name,
                    duration=duration,
                    success=True
                )
                
                self.logger.info(f"TTS synthesis successful: {output_path}")
                return output_path
            else:
                raise RuntimeError("TTS synthesis failed")
                
        except Exception as e:
            duration = asyncio.get_event_loop().time() - start_time
            
            # Record error
            await self.error_handler.handle_error(
                exception=e,
                category=ErrorCategory.AI_SERVICE,
                severity=ErrorSeverity.MEDIUM,
                context={
                    "service": "tts",
                    "text_length": len(text),
                    "voice": voice,
                    "provider": provider
                }
            )
            
            # Record metrics
            self.metrics.record_ai_service_call(
                service="tts",
                model=provider or "unknown",
                duration=duration,
                success=False
            )
            
            return None
            
    async def batch_synthesize(
        self,
        text_segments: List[Dict[str, Any]],
        provider: str = None,
        output_dir: str = None
    ) -> List[Optional[str]]:
        """Synthesize multiple text segments"""
        
        if not output_dir:
            output_dir = self.settings.temp_dir / "tts_batch"
            output_dir.mkdir(exist_ok=True)
            
        tasks = []
        for i, segment in enumerate(text_segments):
            text = segment.get("text", "")
            voice = segment.get("voice", self.settings.tts_voice)
            output_path = str(output_dir / f"segment_{i:03d}_{voice}.wav")
            
            task = self.synthesize(
                text=text,
                voice=voice,
                provider=provider,
                output_path=output_path
            )
            tasks.append(task)
            
        # Execute in parallel with concurrency limit
        semaphore = asyncio.Semaphore(3)  # Max 3 concurrent TTS requests
        
        async def limited_synthesize(task):
            async with semaphore:
                return await task
                
        results = await asyncio.gather(*[limited_synthesize(task) for task in tasks])
        return results
        
    def get_available_voices(self, provider: str = None) -> List[Dict[str, str]]:
        """Get available voices"""
        
        if provider and provider in self.providers:
            return self.providers[provider].get_available_voices()
            
        # Return voices from all providers
        all_voices = []
        for provider_name, provider_obj in self.providers.items():
            voices = provider_obj.get_available_voices()
            for voice in voices:
                voice["provider"] = provider_name
            all_voices.extend(voices)
            
        return all_voices
        
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.providers.keys())
        
    def get_provider_status(self) -> Dict[str, bool]:
        """Get status of all providers"""
        return {name: provider.is_available() for name, provider in self.providers.items()}
        
    async def test_synthesis(self, provider: str = None) -> Dict[str, Any]:
        """Test TTS synthesis with sample text"""
        
        test_text = "这是一个语音合成测试。"
        
        try:
            output_path = await self.synthesize(
                text=test_text,
                provider=provider
            )
            
            if output_path and os.path.exists(output_path):
                file_size = os.path.getsize(output_path)
                return {
                    "success": True,
                    "output_path": output_path,
                    "file_size": file_size,
                    "provider": provider or "auto"
                }
            else:
                return {
                    "success": False,
                    "error": "No output file generated"
                }
                
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }


# Global TTS service instance
_tts_service: Optional[TTSService] = None


def get_tts_service(settings: Optional[Settings] = None) -> TTSService:
    """Get global TTS service instance"""
    global _tts_service
    if _tts_service is None:
        _tts_service = TTSService(settings)
    return _tts_service