# utils/audio_mixer.py

"""
独立音频混音与处理模块
支持 BGM/SFX/TTS 多轨道混合，基于 librosa 音频加载
可被 AudioProcessingNode 或其他模块复用
"""

import librosa
import numpy as np
# import soundfile as sf  # Lazy import - only when actually exporting
from scipy import signal
from typing import List, Dict, Any, Optional, Tuple
import os


# ======================
# 均衡器预设
# ======================

EQ_PRESETS = {
    "enhance_vocals": {
        "name": "增强人声",
        "bands": [
            {"freq": 100,  "gain_db": -2,  "q": 1.0},
            {"freq": 300,  "gain_db": 0,   "q": 1.0},
            {"freq": 1000, "gain_db": +3,  "q": 1.2},
            {"freq": 3000, "gain_db": +4,  "q": 1.4},
            {"freq": 8000, "gain_db": +2,  "q": 1.6}
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

# ======================
# 核心混音类
# ======================

class AudioMixer:
    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self._loaded_audio_cache = {}  # 可选：缓存已加载音频

    def mix_tracks(
        self,
        tracks: List[Dict[str, Any]],
        output_path: str,
        eq_preset: str = "enhance_vocals",
        denoise_strength: float = 0.2,
        target_loudness_db: float = -16.0,
        bit_depth: int = 16,
        apply_denoise: bool = True,
        normalize_loudness: bool = True
    ) -> Dict[str, Any]:
        """
        混合多个音轨（BGM/SFX/TTS）并输出最终音频

        Args:
            tracks: 音频轨道列表，每个元素是包含 clips 的 dict
            output_path: 输出 WAV 文件路径
            eq_preset: 均衡器预设名称
            denoise_strength: 降噪强度 (0~1)
            target_loudness_db: 目标响度 (dB)
            bit_depth: 位深 (16/24/32)
            apply_denoise: 是否降噪
            normalize_loudness: 是否归一化

        Returns:
            包含处理日志和文件信息的字典
        """
        # 1. 混音
        master_audio, duration = self._mix_and_align(tracks)

        if len(master_audio) == 0:
            raise ValueError("No audio clips found or all clips are out of range.")

        # 2. 降噪
        if apply_denoise and denoise_strength > 0:
            master_audio = self._apply_denoise(master_audio, denoise_strength)

        # 3. 均衡
        if eq_preset in EQ_PRESETS:
            bands = EQ_PRESETS[eq_preset]["bands"]
            master_audio = self._apply_equalization(master_audio, bands)

        # 4. 响度归一化
        if normalize_loudness:
            master_audio = self._normalize_loudness(master_audio, target_loudness_db)

        # 5. 导出
        self._export_wav(master_audio, output_path, bit_depth)

        return {
            "file_path": output_path,
            "sample_rate": self.sample_rate,
            "bit_depth": bit_depth,
            "duration": duration,
            "processing_log": {
                "eq_used": eq_preset,
                "denoised": apply_denoise,
                "normalized": normalize_loudness,
                "num_clips": sum(len(track.get("clips", [])) for track in tracks)
            }
        }

    def _mix_and_align(self, tracks: List[Dict]) -> Tuple[np.ndarray, float]:
        """对齐并混合所有轨道"""
        duration = self._get_max_duration(tracks)
        num_samples = int(duration * self.sample_rate)
        mixed = np.zeros(num_samples, dtype=np.float32)

        for track in tracks:
            if not track or "clips" not in track:
                continue
            for clip in track["clips"]:
                start_sec = clip["start"]
                start_sample = int(start_sec * self.sample_rate)
                end_sample = start_sample + int(clip["duration"] * self.sample_rate)
                if start_sample >= num_samples:
                    continue
                if end_sample > num_samples:
                    end_sample = num_samples

                # 加载音频
                audio_segment = self._load_and_resample_clip(clip)
                if audio_segment is None:
                    continue

                # 调整音量
                volume = self._db_to_linear(clip.get("volume_db", 0.0))
                audio_segment = audio_segment * volume

                # 叠加
                seg_len = min(len(audio_segment), end_sample - start_sample)
                mixed[start_sample:start_sample + seg_len] += audio_segment[:seg_len]

        return np.clip(mixed, -1.0, 1.0), duration

    def _get_max_duration(self, tracks: List[Dict]) -> float:
        max_end = 0.0
        for track in tracks:
            if not track or "clips" not in track:
                continue
            for clip in track["clips"]:
                end = clip["start"] + clip["duration"]
                max_end = max(max_end, end)
        return max_end

    def _load_and_resample_clip(self, clip: Dict) -> Optional[np.ndarray]:
        """加载音频片段并重采样到目标采样率"""
        url = clip["audio"]["url"]
        if not os.path.exists(url):
            print(f"[警告] 音频文件不存在: {url}")
            return None

        try:
            # 使用 librosa 加载（自动归一化到 [-1,1]）
            audio, sr = librosa.load(url, sr=None, mono=True)  # 强制单声道
            if sr != self.sample_rate:
                audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sample_rate)
            return audio
        except Exception as e:
            print(f"[错误] 加载音频失败 {url}: {str(e)}")
            return None

    def _apply_denoise(self, audio: np.ndarray, strength: float) -> np.ndarray:
        """简单谱减法降噪"""
        from scipy.fft import rfft, irfft

        chunk_size = 1024
        step = chunk_size // 2
        output = np.zeros_like(audio)

        # ✅ 修复：确保 noise_chunk 长度与 chunk_size 一致
        noise_dur = int(0.1 * self.sample_rate)
        if len(audio) < chunk_size:
            # 音频太短，直接返回不降噪
            return audio

        # 取前100ms或chunk_size，然后裁剪/填充到chunk_size
        noise_chunk = audio[:min(noise_dur, chunk_size)]
        if len(noise_chunk) < chunk_size:
            # 填充到chunk_size
            noise_chunk = np.pad(noise_chunk, (0, chunk_size - len(noise_chunk)), mode='constant')
        else:
            noise_chunk = noise_chunk[:chunk_size]

        noise_fft = np.abs(rfft(noise_chunk))

        for i in range(0, len(audio) - chunk_size, step):
            chunk = audio[i:i + chunk_size]
            chunk_fft = rfft(chunk)
            magnitude = np.abs(chunk_fft)
            phase = np.angle(chunk_fft)
            magnitude_denoised = np.maximum(magnitude - strength * noise_fft, 0)
            chunk_clean = irfft(magnitude_denoised * np.exp(1j * phase))
            window = np.hanning(len(chunk_clean))
            output[i:i + len(chunk_clean)] += chunk_clean * window

        return np.clip(output, -1.0, 1.0)

    def _apply_equalization(self, audio: np.ndarray, bands: List[Dict]) -> np.ndarray:
        """多段均衡处理"""
        filtered = audio.copy()
        for band in bands:
            freq = band["freq"]
            gain_db = band["gain_db"]
            q = band.get("q", 1.0)
            A = 10 ** (gain_db / 40.0)
            w0 = 2 * np.pi * freq / self.sample_rate
            alpha = np.sin(w0) / (2 * q)

            a0 = 1 + alpha / A
            b0 = (1 + alpha * A) / a0
            b1 = (-2 * np.cos(w0)) / a0
            b2 = (1 - alpha * A) / a0
            a1 = (-2 * np.cos(w0)) / a0
            a2 = (1 - alpha / A) / a0

            b = [b0, b1, b2]
            a = [1, a1, a2]

            filtered = signal.lfilter(b, a, filtered)

        return filtered

    def _normalize_loudness(self, audio: np.ndarray, target_db: float) -> np.ndarray:
        """RMS 响度归一化"""
        current_rms = np.sqrt(np.mean(audio ** 2))
        if current_rms == 0:
            return audio
        current_db = 20 * np.log10(current_rms + 1e-10)
        gain_db = target_db - current_db
        gain_linear = 10 ** (gain_db / 20.0)
        return np.clip(audio * gain_linear, -1.0, 1.0)

    def _db_to_linear(self, db: float) -> float:
        return 10 ** (db / 20.0)

    def _export_wav(self, audio: np.ndarray, path: str, bit_depth: int):
        """使用 soundfile 导出 WAV，支持 24bit"""
        import soundfile as sf  # Lazy import
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if bit_depth == 16:
            data = (audio * 32767).astype(np.int16)
        elif bit_depth == 24:
            # soundfile 支持 'PCM_24'
            data = (audio * 8388607).astype(np.int32)  # 24-bit range
        elif bit_depth == 32:
            data = (audio * 2147483647).astype(np.int32)
        else:
            data = np.clip(audio, -1.0, 1.0)  # float32

        sf.write(path, data, self.sample_rate, subtype=f'PCM_{bit_depth}')


# ======================
# 便捷函数（可选）
# ======================

def mix_audio_tracks(
    bgm_track: Dict,
    sfx_track: Dict,
    tts_track: Dict,
    output_path: str,
    sample_rate: int = 48000,
    **kwargs
) -> Dict[str, Any]:
    """
    便捷函数：直接混合三个轨道
    """
    tracks = [t for t in [bgm_track, sfx_track, tts_track] if t]
    mixer = AudioMixer(sample_rate=sample_rate)
    return mixer.mix_tracks(tracks, output_path, **kwargs)