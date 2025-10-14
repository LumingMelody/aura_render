"""
TTS服务管理器 - 统一管理多个TTS服务提供商
"""
from typing import Dict, List, Any, Optional, Union
import asyncio
from dataclasses import dataclass
from enum import Enum

from .base_tts_client import (
    BaseTTSClient,
    TTSRequest,
    TTSResponse,
    VoiceProfile,
    VoiceGender,
    VoiceStyle
)
from .azure_tts_client import AzureTTSClient
from .openai_tts_client import OpenAITTSClient
from .edge_tts_client import EdgeTTSClient


class TTSProvider(Enum):
    AZURE = "azure"
    OPENAI = "openai"
    EDGE = "edge"


@dataclass
class TTSConfig:
    """TTS服务配置"""
    provider: TTSProvider
    api_key: Optional[str] = None
    region: Optional[str] = None
    model: Optional[str] = None
    enabled: bool = True
    priority: int = 1  # 1=最高优先级
    cost_weight: float = 1.0  # 成本权重
    quality_weight: float = 1.0  # 质量权重


@dataclass
class VoiceSelectionCriteria:
    """语音选择标准"""
    language: Optional[str] = None
    gender: Optional[VoiceGender] = None
    style: Optional[VoiceStyle] = None
    prefer_premium: bool = False
    max_cost_per_char: Optional[float] = None
    min_quality_score: float = 0.5


class TTSServiceManager:
    """TTS服务管理器"""

    def __init__(self):
        self.clients: Dict[TTSProvider, BaseTTSClient] = {}
        self.configs: Dict[TTSProvider, TTSConfig] = {}
        self.voice_cache: Dict[TTSProvider, List[VoiceProfile]] = {}

    async def register_service(self, config: TTSConfig):
        """注册TTS服务"""
        if not config.enabled:
            return

        try:
            if config.provider == TTSProvider.AZURE:
                if not config.api_key:
                    raise ValueError("Azure TTS requires API key")
                client = AzureTTSClient(config.api_key, config.region or "eastus")

            elif config.provider == TTSProvider.OPENAI:
                if not config.api_key:
                    raise ValueError("OpenAI TTS requires API key")
                client = OpenAITTSClient(config.api_key, config.model or "tts-1-hd")

            elif config.provider == TTSProvider.EDGE:
                client = EdgeTTSClient()

            else:
                raise ValueError(f"Unsupported TTS provider: {config.provider}")

            self.clients[config.provider] = client
            self.configs[config.provider] = config

            print(f"✅ Registered {config.provider.value} TTS service")

        except Exception as e:
            print(f"❌ Failed to register {config.provider.value}: {e}")

    async def synthesize_speech(self, request: TTSRequest,
                              provider: Optional[TTSProvider] = None,
                              fallback: bool = True) -> TTSResponse:
        """合成语音"""
        if provider:
            # 使用指定提供商
            if provider in self.clients:
                try:
                    async with self.clients[provider]:
                        return await self.clients[provider].synthesize_speech(request)
                except Exception as e:
                    if not fallback:
                        raise
                    print(f"❌ {provider.value} failed, trying fallback: {e}")

        # 自动选择提供商或使用回退策略
        providers = self._get_ranked_providers(request)

        for provider in providers:
            if provider not in self.clients:
                continue

            try:
                async with self.clients[provider]:
                    response = await self.clients[provider].synthesize_speech(request)
                    print(f"✅ Speech synthesized using {provider.value}")
                    return response

            except Exception as e:
                print(f"❌ {provider.value} synthesis failed: {e}")
                continue

        raise Exception("All TTS providers failed")

    async def get_available_voices(self, criteria: VoiceSelectionCriteria = None) -> Dict[TTSProvider, List[VoiceProfile]]:
        """获取所有提供商的可用语音"""
        all_voices = {}

        for provider, client in self.clients.items():
            try:
                if provider in self.voice_cache:
                    voices = self.voice_cache[provider]
                else:
                    async with client:
                        voices = await client.get_available_voices(
                            criteria.language if criteria else None
                        )
                    self.voice_cache[provider] = voices

                # 应用筛选条件
                if criteria:
                    voices = self._filter_voices(voices, criteria)

                all_voices[provider] = voices

            except Exception as e:
                print(f"Failed to get voices from {provider.value}: {e}")
                all_voices[provider] = []

        return all_voices

    async def select_best_voice(self, criteria: VoiceSelectionCriteria) -> Optional[tuple]:
        """选择最佳语音 (provider, voice_profile)"""
        all_voices = await self.get_available_voices(criteria)

        best_voice = None
        best_provider = None
        best_score = 0

        for provider, voices in all_voices.items():
            config = self.configs.get(provider)
            if not config:
                continue

            for voice in voices:
                score = self._calculate_voice_score(voice, config, criteria)
                if score > best_score:
                    best_score = score
                    best_voice = voice
                    best_provider = provider

        return (best_provider, best_voice) if best_voice else None

    async def batch_synthesize(self, requests: List[TTSRequest],
                             provider: Optional[TTSProvider] = None) -> List[TTSResponse]:
        """批量合成语音"""
        if provider and provider in self.clients:
            # 使用指定提供商
            async with self.clients[provider]:
                return await self.clients[provider].batch_synthesize(requests)

        # 并发处理多个请求
        tasks = []
        for request in requests:
            task = self.synthesize_speech(request)
            tasks.append(task)

        return await asyncio.gather(*tasks, return_exceptions=True)

    async def get_cost_estimate(self, text: str, voice_id: str,
                              provider: TTSProvider) -> float:
        """估算合成成本"""
        if provider not in self.clients:
            return 0.0

        client = self.clients[provider]
        return client.estimate_cost(text, voice_id)

    async def compare_providers(self, request: TTSRequest) -> Dict[TTSProvider, Dict[str, Any]]:
        """比较不同提供商的性能和成本"""
        comparison = {}

        for provider, client in self.clients.items():
            try:
                # 估算成本
                cost = client.estimate_cost(request.text, request.voice_id)

                # 获取语音详情
                async with client:
                    voice = await client.get_voice_details(request.voice_id)

                comparison[provider] = {
                    "estimated_cost": cost,
                    "voice_quality": voice.quality_score if voice else 0.5,
                    "is_premium": voice.is_premium if voice else False,
                    "supported_styles": [s.value for s in voice.styles] if voice else [],
                    "provider_priority": self.configs[provider].priority
                }

            except Exception as e:
                comparison[provider] = {
                    "error": str(e),
                    "available": False
                }

        return comparison

    async def health_check(self) -> Dict[TTSProvider, bool]:
        """健康检查"""
        health_status = {}

        for provider, client in self.clients.items():
            try:
                async with client:
                    health_status[provider] = await client.health_check()
            except Exception:
                health_status[provider] = False

        return health_status

    def _get_ranked_providers(self, request: TTSRequest) -> List[TTSProvider]:
        """根据请求获取排序后的提供商列表"""
        available_providers = []

        for provider, config in self.configs.items():
            if provider not in self.clients or not config.enabled:
                continue

            # 计算提供商分数
            score = self._calculate_provider_score(provider, request)
            available_providers.append((provider, score))

        # 按分数排序
        available_providers.sort(key=lambda x: x[1], reverse=True)
        return [p[0] for p in available_providers]

    def _calculate_provider_score(self, provider: TTSProvider, request: TTSRequest) -> float:
        """计算提供商分数"""
        config = self.configs[provider]
        score = 0

        # 优先级分数
        score += (5 - config.priority) * 0.3

        # 成本分数（Edge免费，得分最高）
        if provider == TTSProvider.EDGE:
            score += 0.4
        elif provider == TTSProvider.OPENAI:
            score += 0.2
        else:  # Azure
            score += 0.1

        # 语言支持分数
        if provider == TTSProvider.EDGE and request.language.startswith("zh"):
            score += 0.2  # Edge对中文支持好
        elif provider == TTSProvider.AZURE:
            score += 0.3  # Azure语音质量最好

        return score

    def _filter_voices(self, voices: List[VoiceProfile], criteria: VoiceSelectionCriteria) -> List[VoiceProfile]:
        """根据条件筛选语音"""
        filtered = voices

        if criteria.gender:
            filtered = [v for v in filtered if v.gender == criteria.gender]

        if criteria.style:
            filtered = [v for v in filtered if criteria.style in v.styles]

        if criteria.min_quality_score:
            filtered = [v for v in filtered if v.quality_score >= criteria.min_quality_score]

        return filtered

    def _calculate_voice_score(self, voice: VoiceProfile, config: TTSConfig,
                             criteria: VoiceSelectionCriteria) -> float:
        """计算语音分数"""
        score = voice.quality_score * config.quality_weight

        # 性别匹配
        if criteria.gender and voice.gender == criteria.gender:
            score += 0.2

        # 样式匹配
        if criteria.style and criteria.style in voice.styles:
            score += 0.2

        # 高级语音偏好
        if criteria.prefer_premium and voice.is_premium:
            score += 0.1

        return score

    async def get_service_statistics(self) -> Dict[str, Any]:
        """获取服务统计信息"""
        stats = {
            "total_providers": len(self.clients),
            "active_providers": len([c for c in self.configs.values() if c.enabled]),
            "provider_status": await self.health_check()
        }

        # 统计语音数量
        all_voices = await self.get_available_voices()
        stats["voice_counts"] = {
            provider.value: len(voices) for provider, voices in all_voices.items()
        }

        return stats

    async def clear_voice_cache(self):
        """清理语音缓存"""
        self.voice_cache.clear()

    async def shutdown(self):
        """关闭所有客户端"""
        for client in self.clients.values():
            if hasattr(client, 'session') and client.session:
                await client.session.close()


# 预设配置
def create_default_tts_manager(
    azure_key: Optional[str] = None,
    azure_region: str = "eastus",
    openai_key: Optional[str] = None,
    openai_model: str = "tts-1-hd",
    include_edge: bool = True
) -> TTSServiceManager:
    """创建默认TTS管理器"""

    manager = TTSServiceManager()

    # 注册Edge TTS（免费）
    if include_edge:
        edge_config = TTSConfig(
            provider=TTSProvider.EDGE,
            enabled=True,
            priority=1,  # 最高优先级（免费）
            cost_weight=1.0,
            quality_weight=0.8
        )
        asyncio.create_task(manager.register_service(edge_config))

    # 注册Azure TTS
    if azure_key:
        azure_config = TTSConfig(
            provider=TTSProvider.AZURE,
            api_key=azure_key,
            region=azure_region,
            enabled=True,
            priority=2,
            cost_weight=0.7,
            quality_weight=1.0
        )
        asyncio.create_task(manager.register_service(azure_config))

    # 注册OpenAI TTS
    if openai_key:
        openai_config = TTSConfig(
            provider=TTSProvider.OPENAI,
            api_key=openai_key,
            model=openai_model,
            enabled=True,
            priority=3,
            cost_weight=0.6,
            quality_weight=0.9
        )
        asyncio.create_task(manager.register_service(openai_config))

    return manager