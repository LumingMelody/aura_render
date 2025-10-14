# nodes/audio_processing_node.py

from video_generate_protocol import BaseNode
from typing import Dict, List, Any
import numpy as np
from scipy import signal
import json
import os
from assistance_analysis.audio_mixer import mix_audio_tracks


# 均衡器预设（EQ Presets）
EQ_PRESETS = {
    "enhance_vocals": {
        "name": "增强人声",
        "bands": [
            {"freq": 100,  "gain_db": -2,  "q": 1.0},   # 衰减低频隆隆声
            {"freq": 300,  "gain_db": 0,   "q": 1.0},
            {"freq": 1000, "gain_db": +3,  "q": 1.2},   # 提升中频清晰度
            {"freq": 3000, "gain_db": +4,  "q": 1.4},   # 提升齿音与穿透力
            {"freq": 8000, "gain_db": +2,  "q": 1.6}    # 提升空气感
        ]
    },
    "broadcast": {
        "name": "广播级",
        "bands": [
            {"freq": 80,   "gain_db": -3},
            {"freq": 200,  "gain_db": +1},
            {"freq": 1000, "gain_db": +2},
            {"freq": 4000, "gain_db": +3},
            {"freq": 10000,"gain_db": +1}
        ]
    },
    "music_clear": {
        "name": "音乐清晰化",
        "bands": [
            {"freq": 60,   "gain_db": -1},
            {"freq": 200,  "gain_db": 0},
            {"freq": 1000, "gain_db": +1},
            {"freq": 5000, "gain_db": +2},
            {"freq": 12000,"gain_db": +1}
        ]
    },
    "film_dialog": {
        "name": "电影对白",
        "bands": [
            {"freq": 120,  "gain_db": -4},
            {"freq": 400,  "gain_db": +1},
            {"freq": 1200, "gain_db": +3},
            {"freq": 3500, "gain_db": +4},
            {"freq": 9000, "gain_db": +2}
        ]
    }
}

# 默认系统参数
DEFAULT_SAMPLE_RATE = 48000  # Hz
DEFAULT_BIT_DEPTH = 16       # 位深
TARGET_RMS_DB = -16.0        # 目标整体响度（LUFS近似）
NOISE_FLOOR_DB = -60.0       # 噪声基底阈值
DENOISE_STRENGTH = 0.2       # 降噪强度（0~1）


class AudioProcessingNode(BaseNode):
    required_inputs = [
        {
            "name": "bgm_composition_id",
            "label": "BGM合成结果列表",
            "type": list,
            "required": False,  # ✅ 改为可选
            "default": [],
            "desc": "包含每段匹配的音乐资源，如 [{'segment_index': 0, 'start_time': 0.0, 'end_time': 10.0, 'mood': '温馨', 'genre': '轻音乐', 'narrative_role': '开场', 'transition': '淡入', 'music_suggestion': {'title': '轻松的早晨', 'artist': '轻音乐大师', 'reason': '适合开场的温馨氛围'}, 'matched_audio': {...}, 'alternatives': [...] }]",
            "field_type": "json"
        },
        {
            "name": "sfx_track_id",
            "label": "音效片段列表",
            "type": list,
            "required": False,  # ✅ 改为可选
            "default": [],
            "desc": "音效片段列表，包含每个片段的开始时间和结束时间，以及音效的描述信息",
            "field_type": "json"
        },
        {
            "name": "tts_track_id",
            "label": "语音音轨",
            "type": float,
            "required": False,
            "default": None,
            "desc": "生成的 TTS 语音音轨列表，如 [{'start': 0.0, 'end': 10.0, 'url': 'https://...'}]",
            "field_type": "text"
        }
    ]

    output_schema=[
        {
            "name": "bgm_composition_balance_id",
            "label": "BGM合成结果列表",
            "type": list,
            "required": True,
            "desc": "包含每段匹配的音乐资源，如 [{'segment_index': 0, 'start_time': 0.0, 'end_time': 10.0, 'mood': '温馨', 'genre': '轻音乐', 'narrative_role': '开场', 'transition': '淡入', 'music_suggestion': {'title': '轻松的早晨', 'artist': '轻音乐大师', 'reason': '适合开场的温馨氛围'}, 'matched_audio': {...}, 'alternatives': [...] }]",
            "field_type": "json"
        },
        {
            "name": "sfx_track_balance_id",
            "label": "音效片段列表",
            "type": list,
            "required": True,
            "desc": "音效片段列表，包含每个片段的开始时间和结束时间，以及音效的描述信息",
            "field_type": "json"
        },
        {
            "name": "tts_track_balance_id",
            "label": "语音音轨",
            "type": float,
            "required": False,
            "desc": "生成的 TTS 语音音轨列表，如 [{'start': 0.0, 'end': 10.0, 'url': 'https://...'}]",
            "field_type": "text"
        }
    ]

    file_upload_config = {
        "audio": {
            "enabled": True,
            "accept": ".wav,.mp3,.aiff",
            "desc": "可上传参考音频或自定义EQ曲线"
        }
    }

    system_parameters = {
        "eq_preset": "enhance_vocals",        # 均衡预设
        "apply_denoise": True,                # 是否启用降噪
        "denoise_strength": DENOISE_STRENGTH, # 降噪强度
        "normalize_loudness": True,           # 响度归一化
        "target_loudness_db": TARGET_RMS_DB,  # 目标响度
        "sample_rate": DEFAULT_SAMPLE_RATE,   # 重采样率
        "bit_depth": DEFAULT_BIT_DEPTH        # 输出位深
    }

    def __init__(self, node_id: str, name: str = "音频降噪与均衡"):
        super().__init__(node_id=node_id, node_type="audio_processing", name=name)

        # 默认参数
        self.config = {
            "sample_rate": 48000,
            "bit_depth": 16,
            "eq_preset": "enhance_vocals",
            "denoise_strength": 0.2,
            "target_loudness_db": -16.0,
            "apply_denoise": True,
            "normalize_loudness": True
        }

    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        self.validate_context(context)

        bgm_track = context.get("bgm_track")
        sfx_track = context.get("sfx_track")
        tts_track = context.get("tts_track")

        # ✅ 如果所有音频输入都为空，返回空的 audio_tracks 而不是报错
        if not bgm_track and not sfx_track and not tts_track:
            self.logger.warning("所有音频输入为空，返回空的 audio_tracks")
            return {
                "audio_tracks": [],
                "final_mix": None,
                "node_id": self.node_id
            }

        # 生成输出路径
        import uuid
        output_path = f"/tmp/audio_mix_{uuid.uuid4().hex[:8]}.wav"

        # 调用复用模块进行混音处理
        result = mix_audio_tracks(
            bgm_track=bgm_track,
            sfx_track=sfx_track,
            tts_track=tts_track,
            output_path=output_path,
            sample_rate=self.config["sample_rate"],
            eq_preset=self.config["eq_preset"],
            denoise_strength=self.config["denoise_strength"],
            target_loudness_db=self.config["target_loudness_db"],
            bit_depth=self.config["bit_depth"],
            apply_denoise=self.config["apply_denoise"],
            normalize_loudness=self.config["normalize_loudness"]
        )

        # ✅ 修复：返回符合 timeline_integration 期望的格式
        # timeline_integration 期望 audio_tracks 是一个列表
        audio_tracks = []
        if result:
            # 将 final_mix 包装成 audio_tracks 列表格式
            audio_tracks.append({
                "track_name": "master_audio",
                "track_type": "mixed",
                "clips": [{
                    "id": "master_mix_001",
                    "start": 0.0,
                    "end": result.get("duration", 30.0) if isinstance(result, dict) else 30.0,
                    "file_path": result.get("file_path", output_path) if isinstance(result, dict) else output_path,
                    "volume_db": 0.0
                }]
            })

        return {
            "final_mix": result,  # 保留原始输出，向后兼容
            "audio_tracks": audio_tracks  # 新增：符合 timeline_integration 期望的格式
        }

    def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        super().regenerate(context, user_intent)

        override = user_intent.get("audio_override")
        if override:
            if "eq_preset" in override:
                self.config["eq_preset"] = override["eq_preset"]
            if "denoise_strength" in override:
                self.config["denoise_strength"] = override["denoise_strength"]
            if "target_loudness" in override:
                self.config["target_loudness_db"] = override["target_loudness"]

        return self.generate(context)

    # def __init__(self, node_id: str, name: str = "音频降噪与均衡"):
    #     super().__init__(node_id=node_id, node_type="audio_processing", name=name)

    # def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
    #     self.validate_context(context)

    #     audio_tracks: Dict = context["audio_tracks"]
    #     sample_rate = self.system_parameters["sample_rate"]

    #     # 1. 混音：将所有轨道按时间对齐混合
    #     master_audio = self._mix_tracks(audio_tracks, sample_rate)

    #     # 2. 降噪处理
    #     if self.system_parameters["apply_denoise"]:
    #         master_audio = self._apply_denoise(master_audio, sample_rate)

    #     # 3. 均衡处理
    #     eq_preset_name = self.system_parameters["eq_preset"]
    #     eq_config = EQ_PRESETS.get(eq_preset_name)
    #     if eq_config:
    #         master_audio = self._apply_equalization(master_audio, sample_rate, eq_config["bands"])

    #     # 4. 响度归一化
    #     if self.system_parameters["normalize_loudness"]:
    #         target_db = self.system_parameters["target_loudness_db"]
    #         master_audio = self._normalize_loudness(master_audio, target_db)

    #     # 5. 生成最终混音文件
    #     output_path = self._export_wav(master_audio, sample_rate)

    #     return {
    #         "final_mix": {
    #             "file_path": output_path,
    #             "sample_rate": sample_rate,
    #             "bit_depth": self.system_parameters["bit_depth"],
    #             "duration": len(master_audio) / sample_rate,
    #             "processing_log": {
    #                 "eq_used": eq_preset_name,
    #                 "denoised": self.system_parameters["apply_denoise"],
    #                 "normalized": self.system_parameters["normalize_loudness"]
    #             }
    #         }
    #     }

    # def _mix_tracks(self, tracks: Dict, sample_rate: int) -> np.ndarray:
    #     """混合所有音频轨道"""
    #     duration = self._get_max_duration(tracks)
    #     num_samples = int(duration * sample_rate)
    #     mixed = np.zeros(num_samples, dtype=np.float32)

    #     for track_name, track_data in tracks.items():
    #         clips = track_data.get("clips", [])
    #         for clip in clips:
    #             start_sample = int(clip["start"] * sample_rate)
    #             end_sample = start_sample + int(clip["duration"] * sample_rate)
    #             if end_sample > num_samples:
    #                 continue

    #             # 模拟加载音频片段（此处简化为正弦波或随机波，实际应加载文件）
    #             audio_segment = self._load_audio_clip(clip, sample_rate, clip["duration"])

    #             # 调整音量
    #             volume = self._db_to_linear(clip.get("volume_db", 0.0))
    #             audio_segment = audio_segment * volume

    #             # 叠加到主轨道
    #             segment_len = min(len(audio_segment), end_sample - start_sample)
    #             mixed[start_sample:start_sample + segment_len] += audio_segment[:segment_len]

    #     return np.clip(mixed, -1.0, 1.0)  # 防止溢出

    # def _get_max_duration(self, tracks: Dict) -> float:
    #     """获取所有轨道中最长的时间"""
    #     max_end = 0.0
    #     for track in tracks.values():
    #         for clip in track.get("clips", []):
    #             end = clip["start"] + clip["duration"]
    #             max_end = max(max_end, end)
    #     return max_end

    # def _load_audio_clip(self, clip: Dict, sample_rate: int, duration: float) -> np.ndarray:
    #     """模拟加载音频片段（实际项目中应使用 librosa/pydub）"""
    #     # 这里用简单信号代替真实音频加载
    #     t = np.linspace(0, duration, int(sample_rate * duration), False)
    #     if "heartbeat" in clip.get("tags", []):
    #         return 0.3 * np.sin(2 * np.pi * 1.2 * t)  # 心跳低频
    #     elif "explosion" in clip.get("title", ""):
    #         return 0.8 * np.random.rand(len(t)) * np.exp(-t * 2)  # 爆炸噪声衰减
    #     elif "voice" in clip.get("category", "") or "voice" in clip.get("file_path", ""):
    #         return 0.5 * (0.5 * np.sin(2 * np.pi * 500 * t) + 0.3 * np.sin(2 * np.pi * 1500 * t))
    #     else:
    #         return 0.3 * np.random.rand(len(t))  # 默认：白噪声模拟

    # def _apply_denoise(self, audio: np.ndarray, sample_rate: int) -> np.ndarray:
    #     """简单谱减法降噪"""
    #     from scipy.fft import rfft, irfft

    #     strength = self.system_parameters["denoise_strength"]
    #     chunk_size = 1024
    #     step = chunk_size // 2
    #     output = np.zeros_like(audio)

    #     # 估计噪声频谱（取静音段或前100ms）
    #     noise_chunk = audio[:chunk_size]
    #     noise_fft = np.abs(rfft(noise_chunk))

    #     for i in range(0, len(audio) - chunk_size, step):
    #         chunk = audio[i:i + chunk_size]
    #         chunk_fft = rfft(chunk)

    #         # 谱减
    #         magnitude = np.abs(chunk_fft)
    #         phase = np.angle(chunk_fft)
    #         magnitude_denoised = np.maximum(magnitude - strength * noise_fft, 0)

    #         # 逆变换
    #         chunk_clean = irfft(magnitude_denoised * np.exp(1j * phase))
    #         output[i:i + len(chunk_clean)] += chunk_clean * self._window(step)

    #     return np.clip(output, -1.0, 1.0)

    # def _apply_equalization(self, audio: np.ndarray, sample_rate: int, bands: List[Dict]) -> np.ndarray:
    #     """多段均衡器（二阶IIR滤波器级联）"""
    #     filtered = audio.copy()

    #     for band in bands:
    #         freq = band["freq"]
    #         gain_db = band["gain_db"]
    #         q = band.get("q", 1.0)

    #         # 计算增益线性值
    #         A = 10 ** (gain_db / 40.0)

    #         # 设计二阶均衡器（Peaking EQ）
    #         w0 = 2 * np.pi * freq / sample_rate
    #         alpha = np.sin(w0) / (2 * q)

    #         a0 = 1 + alpha / A
    #         b0 = (1 + alpha * A) / a0
    #         b1 = (-2 * np.cos(w0)) / a0
    #         b2 = (1 - alpha * A) / a0
    #         a1 = (-2 * np.cos(w0)) / a0
    #         a2 = (1 - alpha / A) / a0

    #         b = [b0, b1, b2]
    #         a = [1, a1, a2]

    #         filtered = signal.lfilter(b, a, filtered)

    #     return filtered

    # def _normalize_loudness(self, audio: np.ndarray, target_db: float) -> np.ndarray:
    #     """响度归一化（RMS）"""
    #     current_rms = np.sqrt(np.mean(audio ** 2))
    #     if current_rms == 0:
    #         return audio

    #     # 转换为 dB
    #     current_db = 20 * np.log10(current_rms + 1e-10)
    #     gain_db = target_db - current_db
    #     gain_linear = 10 ** (gain_db / 20.0)

    #     return np.clip(audio * gain_linear, -1.0, 1.0)

    # def _db_to_linear(self, db: float) -> float:
    #     """分贝转线性增益"""
    #     return 10 ** (db / 20.0)

    # def _window(self, n: int) -> np.ndarray:
    #     """汉宁窗"""
    #     return np.hanning(n)

    # def _export_wav(self, audio: np.ndarray, sample_rate: int) -> str:
    #     """导出为WAV文件"""
    #     import wave
    #     import struct

    #     bit_depth = self.system_parameters["bit_depth"]
    #     path = self.get_output_path(suffix=".wav")

    #     with wave.open(path, 'w') as wf:
    #         wf.setnchannels(1)
    #         wf.setsampwidth(bit_depth // 8)
    #         wf.setframerate(sample_rate)
    #         if bit_depth == 16:
    #             data = (audio * 32767).astype(np.int16)
    #         elif bit_depth == 24:
    #             # 简化：用3字节表示（实际需特殊处理）
    #             data = (audio * 8388607).astype(np.int32)
    #             data = data << 8  # 左移8位，模拟24位
    #         else:
    #             data = (audio * 2147483647).astype(np.int32)

    #         packed_data = b''.join(struct.pack('<h', sample) for sample in data)
    #         wf.writeframes(packed_data)

    #     return path

    # def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
    #     """支持用户调整参数"""
    #     super().regenerate(context, user_intent)

    #     # 允许用户修改EQ或降噪强度
    #     override = user_intent.get("audio_override")
    #     if override:
    #         if "eq_preset" in override:
    #             self.system_parameters["eq_preset"] = override["eq_preset"]
    #         if "denoise_strength" in override:
    #             self.system_parameters["denoise_strength"] = override["denoise_strength"]
    #         if "target_loudness" in override:
    #             self.system_parameters["target_loudness_db"] = override["target_loudness"]

    #     return self.generate(context)