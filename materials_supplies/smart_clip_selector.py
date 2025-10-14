"""
智能剪辑系统 - 基于YOLO、Whisper和VL的视频片段选择
"""
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import asyncio
from concurrent.futures import ThreadPoolExecutor
import cv2
import whisper
from ultralytics import YOLO

from llm.qwen import QwenLLM


@dataclass
class VideoSegment:
    """视频片段"""
    start_time: float
    end_time: float
    duration: float
    keyframe_path: str
    scene_change_score: float = 0.0
    face_count: int = 0
    speech_ratio: float = 0.0
    content_relevance: float = 0.0
    transition_quality: float = 0.0


@dataclass
class ClipRequest:
    """剪辑请求"""
    video_url: str
    target_duration: float
    target_description: str
    style_requirements: Dict[str, Any]
    context: Dict[str, Any] = None


class SmartClipSelector:
    """智能剪辑选择器"""

    def __init__(self):
        self.qwen = QwenLLM()

        # 模型初始化
        self.yolo_model = None
        self.whisper_model = None

        # 配置参数
        self.keyframe_interval = 5  # 每5秒提取一个关键帧
        self.scene_change_threshold = 0.3
        self.max_vl_calls = 10  # 最大VL调用次数
        self.quality_threshold = 0.6

    async def select_optimal_clips(self, request: ClipRequest) -> List[VideoSegment]:
        """
        选择最优视频片段
        """
        # Step 1: 下载和预处理视频
        video_path = await self._download_video(request.video_url)
        if not video_path:
            return []

        # Step 2: 场景变化检测
        scene_changes = await self._detect_scene_changes(video_path)

        # Step 3: 人脸和语音检测
        face_segments = await self._detect_faces(video_path)
        speech_segments = await self._detect_speech(video_path)

        # Step 4: 合并分析结果，生成候选片段
        candidate_segments = await self._merge_analysis_results(
            video_path, scene_changes, face_segments, speech_segments
        )

        # Step 5: 智能筛选 - 只对Top候选使用VL
        filtered_candidates = await self._filter_candidates_with_vl(
            candidate_segments, request
        )

        # Step 6: 选择最优片段组合
        optimal_clips = self._select_optimal_combination(
            filtered_candidates, request.target_duration
        )

        return optimal_clips

    async def _download_video(self, video_url: str) -> Optional[str]:
        """下载视频到本地"""
        import tempfile
        import httpx
        import os

        try:
            temp_dir = tempfile.mkdtemp()
            video_filename = f"video_{hash(video_url)}.mp4"
            video_path = os.path.join(temp_dir, video_filename)

            async with httpx.AsyncClient() as client:
                response = await client.get(video_url, timeout=60.0)
                response.raise_for_status()

                with open(video_path, 'wb') as f:
                    f.write(response.content)

            return video_path

        except Exception as e:
            print(f"[视频下载] 失败: {e}")
            return None

    async def _detect_scene_changes(self, video_path: str) -> List[float]:
        """
        检测场景变化点
        """
        def _process_scene_detection():
            try:
                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)

                scene_changes = []
                prev_hist = None
                frame_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 每秒处理一帧
                    if frame_count % int(fps) == 0:
                        # 计算直方图
                        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])

                        if prev_hist is not None:
                            # 计算直方图差异
                            correlation = cv2.compareHist(prev_hist, hist, cv2.HISTCMP_CORREL)

                            # 如果相关性低于阈值，认为是场景变化
                            if correlation < (1 - self.scene_change_threshold):
                                timestamp = frame_count / fps
                                scene_changes.append(timestamp)

                        prev_hist = hist.copy()

                    frame_count += 1

                cap.release()
                return scene_changes

            except Exception as e:
                print(f"[场景检测] 失败: {e}")
                return []

        # 在线程池中运行
        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=2)

        return await loop.run_in_executor(executor, _process_scene_detection)

    async def _detect_faces(self, video_path: str) -> List[Dict[str, Any]]:
        """
        检测人脸信息
        """
        def _process_face_detection():
            try:
                if self.yolo_model is None:
                    self.yolo_model = YOLO('yolov8n.pt')  # 轻量级模型

                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)

                face_segments = []
                frame_count = 0

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # 每5秒检测一次
                    if frame_count % (int(fps) * 5) == 0:
                        # YOLO检测
                        results = self.yolo_model(frame)

                        # 统计人脸数量
                        person_count = 0
                        for result in results:
                            for box in result.boxes:
                                if int(box.cls) == 0:  # person class
                                    person_count += 1

                        timestamp = frame_count / fps
                        face_segments.append({
                            "timestamp": timestamp,
                            "face_count": person_count,
                            "confidence": float(np.mean([box.conf for box in result.boxes]) if result.boxes else 0)
                        })

                    frame_count += 1

                cap.release()
                return face_segments

            except Exception as e:
                print(f"[人脸检测] 失败: {e}")
                return []

        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        return await loop.run_in_executor(executor, _process_face_detection)

    async def _detect_speech(self, video_path: str) -> List[Dict[str, Any]]:
        """
        检测语音信息
        """
        def _process_speech_detection():
            try:
                if self.whisper_model is None:
                    self.whisper_model = whisper.load_model("base")

                # 提取音频并进行语音识别
                result = self.whisper_model.transcribe(video_path)

                speech_segments = []
                for segment in result.get("segments", []):
                    speech_segments.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                        "confidence": segment.get("confidence", 0.0),
                        "no_speech_prob": segment.get("no_speech_prob", 1.0)
                    })

                return speech_segments

            except Exception as e:
                print(f"[语音检测] 失败: {e}")
                return []

        loop = asyncio.get_event_loop()
        executor = ThreadPoolExecutor(max_workers=1)

        return await loop.run_in_executor(executor, _process_speech_detection)

    async def _merge_analysis_results(
        self,
        video_path: str,
        scene_changes: List[float],
        face_segments: List[Dict],
        speech_segments: List[Dict]
    ) -> List[VideoSegment]:
        """
        合并分析结果，生成候选片段
        """
        # 获取视频总时长
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        total_duration = frame_count / fps
        cap.release()

        # 基于场景变化生成候选片段
        segments = []
        start_times = [0.0] + scene_changes + [total_duration]

        for i in range(len(start_times) - 1):
            start = start_times[i]
            end = start_times[i + 1]
            duration = end - start

            # 过滤太短的片段
            if duration < 2.0:
                continue

            # 提取关键帧
            keyframe_path = await self._extract_keyframe(video_path, start + duration/2)

            # 计算该片段的特征
            segment = VideoSegment(
                start_time=start,
                end_time=end,
                duration=duration,
                keyframe_path=keyframe_path
            )

            # 计算场景变化得分
            segment.scene_change_score = self._calculate_scene_stability(start, end, scene_changes)

            # 计算人脸信息
            segment.face_count = self._calculate_face_info(start, end, face_segments)

            # 计算语音比例
            segment.speech_ratio = self._calculate_speech_ratio(start, end, speech_segments)

            segments.append(segment)

        return segments

    async def _extract_keyframe(self, video_path: str, timestamp: float) -> str:
        """提取关键帧"""
        import tempfile
        import os

        try:
            temp_dir = tempfile.mkdtemp()
            keyframe_filename = f"keyframe_{timestamp:.1f}.jpg"
            keyframe_path = os.path.join(temp_dir, keyframe_filename)

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)

            # 跳转到指定时间点
            frame_number = int(timestamp * fps)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

            ret, frame = cap.read()
            if ret:
                cv2.imwrite(keyframe_path, frame)

            cap.release()
            return keyframe_path

        except Exception as e:
            print(f"[关键帧提取] 失败: {e}")
            return ""

    def _calculate_scene_stability(self, start: float, end: float, scene_changes: List[float]) -> float:
        """计算场景稳定性得分"""
        # 计算该时间段内的场景变化次数
        changes_in_segment = len([t for t in scene_changes if start <= t <= end])
        segment_duration = end - start

        # 场景变化率越低，稳定性越高
        change_rate = changes_in_segment / segment_duration
        stability_score = max(0, 1.0 - change_rate)

        return stability_score

    def _calculate_face_info(self, start: float, end: float, face_segments: List[Dict]) -> int:
        """计算平均人脸数量"""
        relevant_faces = [
            face for face in face_segments
            if start <= face["timestamp"] <= end
        ]

        if not relevant_faces:
            return 0

        avg_faces = np.mean([face["face_count"] for face in relevant_faces])
        return int(avg_faces)

    def _calculate_speech_ratio(self, start: float, end: float, speech_segments: List[Dict]) -> float:
        """计算语音比例"""
        segment_duration = end - start
        speech_duration = 0.0

        for speech in speech_segments:
            # 计算重叠部分
            overlap_start = max(start, speech["start"])
            overlap_end = min(end, speech["end"])

            if overlap_start < overlap_end:
                # 考虑置信度
                confidence = 1.0 - speech.get("no_speech_prob", 0.0)
                speech_duration += (overlap_end - overlap_start) * confidence

        return min(speech_duration / segment_duration, 1.0)

    async def _filter_candidates_with_vl(
        self,
        candidates: List[VideoSegment],
        request: ClipRequest
    ) -> List[VideoSegment]:
        """
        使用VL模型筛选候选片段（控制调用次数）
        """
        # 预筛选：根据基础特征排序
        pre_filtered = self._pre_filter_candidates(candidates, request)

        # 只对前N个候选使用VL验证
        top_candidates = pre_filtered[:self.max_vl_calls]

        # 并发VL验证
        tasks = []
        for candidate in top_candidates:
            task = self._evaluate_candidate_with_vl(candidate, request)
            tasks.append(task)

        vl_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 更新候选片段的相关性得分
        filtered_candidates = []
        for candidate, result in zip(top_candidates, vl_results):
            if isinstance(result, Exception):
                # VL失败时使用降级评分
                candidate.content_relevance = self._fallback_relevance_score(candidate, request)
            else:
                candidate.content_relevance = result.get("relevance_score", 0.0)
                candidate.transition_quality = result.get("transition_quality", 0.0)

            # 只保留质量较高的候选
            if candidate.content_relevance >= self.quality_threshold:
                filtered_candidates.append(candidate)

        # 加入预筛选中未使用VL的候选（使用降级评分）
        remaining_candidates = pre_filtered[self.max_vl_calls:]
        for candidate in remaining_candidates:
            candidate.content_relevance = self._fallback_relevance_score(candidate, request)
            if candidate.content_relevance >= self.quality_threshold * 0.8:  # 略微降低阈值
                filtered_candidates.append(candidate)

        return filtered_candidates

    def _pre_filter_candidates(self, candidates: List[VideoSegment], request: ClipRequest) -> List[VideoSegment]:
        """基于基础特征预筛选候选片段"""
        # 计算基础得分
        for candidate in candidates:
            base_score = 0.0

            # 场景稳定性得分 (30%)
            base_score += candidate.scene_change_score * 0.3

            # 时长匹配得分 (25%)
            duration_match = 1.0 - min(abs(candidate.duration - request.target_duration) / request.target_duration, 1.0)
            base_score += duration_match * 0.25

            # 人脸存在得分 (25%)
            face_score = min(candidate.face_count / 2.0, 1.0)  # 1-2个人脸为最佳
            base_score += face_score * 0.25

            # 语音质量得分 (20%)
            base_score += candidate.speech_ratio * 0.2

            candidate.content_relevance = base_score

        # 按基础得分排序
        candidates.sort(key=lambda x: x.content_relevance, reverse=True)
        return candidates

    async def _evaluate_candidate_with_vl(self, candidate: VideoSegment, request: ClipRequest) -> Dict[str, Any]:
        """使用VL模型评估候选片段"""
        if not candidate.keyframe_path:
            return {"relevance_score": 0.0, "transition_quality": 0.0}

        evaluation_prompt = f"""
        请分析这个视频关键帧是否符合以下要求：

        【目标描述】: {request.target_description}
        【时长要求】: {request.target_duration}秒
        【风格要求】: {request.style_requirements}

        【片段信息】:
        - 时长: {candidate.duration:.1f}秒
        - 人脸数量: {candidate.face_count}
        - 语音比例: {candidate.speech_ratio:.2f}
        - 场景稳定性: {candidate.scene_change_score:.2f}

        请评估：
        1. 内容相关性 (0-1): 画面内容与目标描述的匹配度
        2. 视觉质量 (0-1): 画面清晰度、构图、光线等
        3. 转场适宜性 (0-1): 作为视频片段的转场自然度
        4. 整体推荐度 (0-1): 综合推荐使用的程度

        输出JSON格式：
        {{
            "relevance_score": 0.85,
            "visual_quality": 0.80,
            "transition_quality": 0.75,
            "overall_recommendation": 0.80
        }}
        """

        try:
            # 读取关键帧图片并转换为base64
            import base64
            with open(candidate.keyframe_path, 'rb') as f:
                image_data = base64.b64encode(f.read()).decode('utf-8')

            loop = asyncio.get_event_loop()
            executor = ThreadPoolExecutor(max_workers=2)

            response = await loop.run_in_executor(
                executor,
                lambda: self.qwen.generate(
                    prompt=evaluation_prompt,
                    images=[f"data:image/jpeg;base64,{image_data}"],
                    max_retries=2
                )
            )

            if response:
                import json
                result = json.loads(response)
                return result

        except Exception as e:
            print(f"[VL评估] 失败: {e}")

        return {"relevance_score": 0.0, "transition_quality": 0.0}

    def _fallback_relevance_score(self, candidate: VideoSegment, request: ClipRequest) -> float:
        """降级相关性评分"""
        # 基于关键词匹配的简单评分
        score = 0.0

        # 时长匹配 (40%)
        duration_match = 1.0 - min(abs(candidate.duration - request.target_duration) / request.target_duration, 1.0)
        score += duration_match * 0.4

        # 场景稳定性 (30%)
        score += candidate.scene_change_score * 0.3

        # 语音质量 (30%)
        score += candidate.speech_ratio * 0.3

        return min(score, 0.7)  # 降级评分最高0.7

    def _select_optimal_combination(self, candidates: List[VideoSegment], target_duration: float) -> List[VideoSegment]:
        """
        选择最优片段组合
        使用动态规划找到最佳组合
        """
        if not candidates:
            return []

        # 按相关性排序
        candidates.sort(key=lambda x: x.content_relevance, reverse=True)

        # 简化版本：贪心选择
        selected_clips = []
        remaining_duration = target_duration
        used_intervals = []

        for candidate in candidates:
            # 检查时间重叠
            if self._has_time_overlap(candidate, used_intervals):
                continue

            # 检查是否需要这个片段
            if candidate.duration <= remaining_duration * 1.2:  # 允许20%的超出
                selected_clips.append(candidate)
                used_intervals.append((candidate.start_time, candidate.end_time))
                remaining_duration -= candidate.duration

                if remaining_duration <= 0:
                    break

        return selected_clips

    def _has_time_overlap(self, candidate: VideoSegment, used_intervals: List[Tuple[float, float]]) -> bool:
        """检查时间重叠"""
        for start, end in used_intervals:
            if not (candidate.end_time <= start or candidate.start_time >= end):
                return True
        return False

    async def batch_select_clips(self, requests: List[ClipRequest]) -> List[List[VideoSegment]]:
        """批量剪辑选择"""
        tasks = []
        for request in requests:
            task = self.select_optimal_clips(request)
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        final_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                print(f"[批量剪辑] 请求 {i} 失败: {result}")
                final_results.append([])
            else:
                final_results.append(result)

        return final_results

    def get_selection_statistics(self, results: List[List[VideoSegment]]) -> Dict[str, Any]:
        """获取选择统计信息"""
        if not results:
            return {}

        total_clips = sum(len(clips) for clips in results)
        avg_clips_per_video = total_clips / len(results) if results else 0

        # 计算平均质量得分
        all_scores = []
        for clips in results:
            all_scores.extend([clip.content_relevance for clip in clips])

        avg_quality = sum(all_scores) / len(all_scores) if all_scores else 0

        return {
            "total_videos_processed": len(results),
            "total_clips_selected": total_clips,
            "average_clips_per_video": avg_clips_per_video,
            "average_quality_score": avg_quality,
            "successful_selections": len([r for r in results if r])
        }