"""
TTS语音合成节点 - 集成多种TTS服务的统一合成节点
"""
from typing import Dict, List, Any, Optional, Tuple
import asyncio
import os
import tempfile
from dataclasses import dataclass, asdict
from pathlib import Path

from tts_services import (
    TTSServiceManager,
    TTSRequest,
    TTSResponse,
    VoiceSelectionCriteria,
    VoiceGender,
    VoiceStyle,
    TTSProvider,
    create_default_tts_manager
)


@dataclass
class AudioSegment:
    """音频片段"""
    text: str
    audio_path: str
    duration: float
    start_time: float
    end_time: float
    voice_id: str
    emotions: Optional[Dict[str, float]] = None
    volume: float = 1.0


@dataclass
class TTSNodeRequest:
    """TTS节点请求"""
    text_segments: List[Dict[str, Any]]  # 文本片段列表
    voice_config: Dict[str, Any]  # 语音配置
    output_config: Dict[str, Any]  # 输出配置
    emotion_analysis: Optional[Dict[str, Any]] = None  # 情感分析结果


@dataclass
class TTSNodeResponse:
    """TTS节点响应"""
    audio_segments: List[AudioSegment]
    total_duration: float
    combined_audio_path: Optional[str] = None
    synthesis_stats: Dict[str, Any] = None
    cost_breakdown: Dict[str, float] = None


class TTSSynthesisNode:
    """TTS语音合成节点"""

    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.tts_manager: Optional[TTSServiceManager] = None
        self.temp_dir = tempfile.mkdtemp(prefix="tts_synthesis_")

        # 默认配置
        self.default_voice_config = {
            "language": "zh-CN",
            "gender": "female",
            "style": "normal",
            "speed": 1.0,
            "pitch": 1.0,
            "volume": 1.0,
            "prefer_premium": False
        }

        self.default_output_config = {
            "format": "mp3",
            "sample_rate": 24000,
            "combine_audio": True,
            "save_individual": True
        }

    async def initialize(self):
        """初始化TTS服务管理器"""
        if self.tts_manager is None:
            # 从配置或环境变量获取API密钥
            azure_key = self.config.get("azure_api_key") or os.getenv("AZURE_TTS_KEY")
            azure_region = self.config.get("azure_region", "eastus")
            openai_key = self.config.get("openai_api_key") or os.getenv("OPENAI_API_KEY")
            openai_model = self.config.get("openai_model", "tts-1-hd")

            self.tts_manager = create_default_tts_manager(
                azure_key=azure_key,
                azure_region=azure_region,
                openai_key=openai_key,
                openai_model=openai_model,
                include_edge=True
            )

            # 等待服务注册完成
            await asyncio.sleep(0.1)

    async def process(self, request: TTSNodeRequest) -> TTSNodeResponse:
        """处理TTS合成请求"""
        await self.initialize()

        # 合并配置
        voice_config = {**self.default_voice_config, **request.voice_config}
        output_config = {**self.default_output_config, **request.output_config}

        # 选择最佳语音
        voice_criteria = VoiceSelectionCriteria(
            language=voice_config["language"],
            gender=VoiceGender(voice_config["gender"]),
            style=VoiceStyle(voice_config["style"]),
            prefer_premium=voice_config["prefer_premium"]
        )

        provider_voice = await self.tts_manager.select_best_voice(voice_criteria)
        if not provider_voice:
            raise Exception("No suitable voice found")

        selected_provider, selected_voice = provider_voice
        print(f"🎤 Selected voice: {selected_voice.name} from {selected_provider.value}")

        # 处理文本片段
        audio_segments = []
        synthesis_tasks = []
        current_time = 0.0

        for i, segment in enumerate(request.text_segments):
            text = segment.get("text", "")
            if not text.strip():
                continue

            # 应用情感参数
            tts_request = TTSRequest(
                text=text,
                voice_id=selected_voice.voice_id,
                language=voice_config["language"],
                style=VoiceStyle(voice_config["style"]),
                speed=voice_config["speed"],
                pitch=voice_config["pitch"],
                volume=voice_config["volume"],
                format=output_config["format"],
                sample_rate=output_config["sample_rate"]
            )

            # 如果有情感分析结果，应用情感参数
            if request.emotion_analysis and "segments" in request.emotion_analysis:
                emotion_segments = request.emotion_analysis["segments"]
                if i < len(emotion_segments):
                    emotion_data = emotion_segments[i]
                    dominant_emotion = emotion_data.get("dominant_emotion", "neutral")
                    intensity = emotion_data.get("intensity", 0.5)

                    # 通过客户端应用情感参数
                    client = self.tts_manager.clients[selected_provider]
                    tts_request = client.apply_emotion_parameters(
                        tts_request, dominant_emotion, intensity
                    )

            synthesis_tasks.append((i, tts_request, segment))

        # 批量合成
        print(f"🔄 Synthesizing {len(synthesis_tasks)} segments...")
        synthesis_results = await self._batch_synthesize(synthesis_tasks, selected_provider)

        # 处理合成结果
        cost_breakdown = {}
        for i, (segment_index, tts_request, segment_info) in enumerate(synthesis_tasks):
            result = synthesis_results[i]

            if isinstance(result, Exception):
                print(f"❌ Segment {segment_index} synthesis failed: {result}")
                continue

            # 保存音频文件
            audio_filename = f"segment_{segment_index:03d}.{output_config['format']}"
            audio_path = os.path.join(self.temp_dir, audio_filename)

            with open(audio_path, "wb") as f:
                f.write(result.audio_data)

            # 创建音频片段
            audio_segment = AudioSegment(
                text=tts_request.text,
                audio_path=audio_path,
                duration=result.duration,
                start_time=current_time,
                end_time=current_time + result.duration,
                voice_id=result.voice_id,
                emotions=segment_info.get("emotions"),
                volume=tts_request.volume
            )

            audio_segments.append(audio_segment)
            current_time += result.duration

            # 记录成本
            provider_name = selected_provider.value
            if provider_name not in cost_breakdown:
                cost_breakdown[provider_name] = 0
            cost_breakdown[provider_name] += result.cost

        # 合并音频（如果需要）
        combined_audio_path = None
        if output_config["combine_audio"] and audio_segments:
            combined_audio_path = await self._combine_audio_segments(
                audio_segments, output_config["format"]
            )

        # 生成统计信息
        synthesis_stats = {
            "total_segments": len(request.text_segments),
            "successful_segments": len(audio_segments),
            "failed_segments": len(request.text_segments) - len(audio_segments),
            "total_characters": sum(len(seg.get("text", "")) for seg in request.text_segments),
            "selected_provider": selected_provider.value,
            "selected_voice": selected_voice.name,
            "average_duration_per_segment": sum(seg.duration for seg in audio_segments) / len(audio_segments) if audio_segments else 0
        }

        return TTSNodeResponse(
            audio_segments=audio_segments,
            total_duration=current_time,
            combined_audio_path=combined_audio_path,
            synthesis_stats=synthesis_stats,
            cost_breakdown=cost_breakdown
        )

    async def _batch_synthesize(self, synthesis_tasks: List[Tuple], provider: TTSProvider) -> List:
        """批量合成语音"""
        requests = [task[1] for task in synthesis_tasks]

        try:
            return await self.tts_manager.batch_synthesize(requests, provider)
        except Exception as e:
            print(f"Batch synthesis failed: {e}")
            # 逐个重试
            results = []
            for request in requests:
                try:
                    result = await self.tts_manager.synthesize_speech(request, provider)
                    results.append(result)
                except Exception as retry_error:
                    results.append(retry_error)
            return results

    async def _combine_audio_segments(self, segments: List[AudioSegment], format: str) -> str:
        """合并音频片段"""
        try:
            # 使用 ffmpeg 合并音频
            combined_path = os.path.join(self.temp_dir, f"combined.{format}")

            # 创建文件列表
            filelist_path = os.path.join(self.temp_dir, "filelist.txt")
            with open(filelist_path, "w") as f:
                for segment in segments:
                    f.write(f"file '{segment.audio_path}'\n")

            # 使用 ffmpeg concat
            import subprocess
            cmd = [
                "ffmpeg", "-f", "concat", "-safe", "0",
                "-i", filelist_path,
                "-c", "copy",
                combined_path,
                "-y"  # 覆盖输出文件
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return combined_path
            else:
                print(f"FFmpeg error: {result.stderr}")
                # 回退到简单拼接
                return await self._simple_audio_concat(segments, format)

        except Exception as e:
            print(f"Audio combination failed: {e}")
            # 回退到简单拼接
            return await self._simple_audio_concat(segments, format)

    async def _simple_audio_concat(self, segments: List[AudioSegment], format: str) -> str:
        """简单音频拼接（直接连接字节）"""
        combined_path = os.path.join(self.temp_dir, f"combined_simple.{format}")

        with open(combined_path, "wb") as output_file:
            for segment in segments:
                with open(segment.audio_path, "rb") as input_file:
                    output_file.write(input_file.read())

        return combined_path

    async def synthesize_single_text(self, text: str,
                                   voice_config: Dict[str, Any] = None,
                                   emotion_config: Dict[str, Any] = None) -> TTSResponse:
        """合成单个文本"""
        await self.initialize()

        config = {**self.default_voice_config, **(voice_config or {})}

        # 选择语音
        criteria = VoiceSelectionCriteria(
            language=config["language"],
            gender=VoiceGender(config["gender"]),
            style=VoiceStyle(config["style"])
        )

        provider_voice = await self.tts_manager.select_best_voice(criteria)
        if not provider_voice:
            raise Exception("No suitable voice found")

        provider, voice = provider_voice

        # 创建请求
        request = TTSRequest(
            text=text,
            voice_id=voice.voice_id,
            language=config["language"],
            style=VoiceStyle(config["style"]),
            speed=config["speed"],
            pitch=config["pitch"],
            volume=config["volume"]
        )

        # 应用情感参数
        if emotion_config:
            dominant_emotion = emotion_config.get("dominant_emotion", "neutral")
            intensity = emotion_config.get("intensity", 0.5)

            client = self.tts_manager.clients[provider]
            request = client.apply_emotion_parameters(request, dominant_emotion, intensity)

        return await self.tts_manager.synthesize_speech(request, provider)

    async def get_available_voices(self, language: str = None) -> Dict[str, List[Dict]]:
        """获取可用语音列表"""
        await self.initialize()

        criteria = VoiceSelectionCriteria(language=language) if language else None
        voices_by_provider = await self.tts_manager.get_available_voices(criteria)

        # 转换为可序列化的格式
        result = {}
        for provider, voices in voices_by_provider.items():
            result[provider.value] = [asdict(voice) for voice in voices]

        return result

    async def estimate_cost(self, text_segments: List[str],
                          voice_config: Dict[str, Any] = None) -> Dict[str, float]:
        """估算合成成本"""
        await self.initialize()

        config = {**self.default_voice_config, **(voice_config or {})}
        total_text = " ".join(text_segments)

        cost_estimates = {}
        for provider, client in self.tts_manager.clients.items():
            try:
                # 获取默认语音ID
                voices = await client.get_available_voices(config["language"])
                if voices:
                    voice_id = voices[0].voice_id
                    cost = client.estimate_cost(total_text, voice_id)
                    cost_estimates[provider.value] = cost
                else:
                    cost_estimates[provider.value] = 0.0
            except Exception:
                cost_estimates[provider.value] = 0.0

        return cost_estimates

    async def health_check(self) -> Dict[str, Any]:
        """健康检查"""
        await self.initialize()

        health_status = await self.tts_manager.health_check()
        service_stats = await self.tts_manager.get_service_statistics()

        return {
            "provider_health": {k.value: v for k, v in health_status.items()},
            "service_statistics": service_stats,
            "temp_directory": self.temp_dir,
            "temp_directory_exists": os.path.exists(self.temp_dir)
        }

    async def cleanup(self):
        """清理临时文件"""
        try:
            import shutil
            if os.path.exists(self.temp_dir):
                shutil.rmtree(self.temp_dir)
                print(f"🧹 Cleaned up temp directory: {self.temp_dir}")
        except Exception as e:
            print(f"Cleanup failed: {e}")

        if self.tts_manager:
            await self.tts_manager.shutdown()

    def __del__(self):
        """析构函数"""
        try:
            asyncio.create_task(self.cleanup())
        except:
            pass