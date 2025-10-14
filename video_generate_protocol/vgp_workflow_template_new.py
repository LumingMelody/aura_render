"""
VGP Workflow Template (New) - 按照用户新需求重构的视频生成流程
根据用户提供的节点流程图，重新组织工作流依赖关系
"""
from typing import Dict, List, Any, Optional
import sys
import os

# Add paths for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'nodes'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'workflow'))

from workflow.workflow_orchestrator import WorkflowOrchestrator, WorkflowConfig
from nodes.base_node import ProcessingContext, NodeConfig, ProcessingPriority

# Import all VGP nodes
from video_generate_protocol.nodes.emotion_analysis_node import EmotionAnalysisNode
from video_generate_protocol.nodes.video_type_identification_node import VideoTypeIdentificationNode
from video_generate_protocol.nodes.shot_block_generation_node import ShotBlockGenerationNode
from video_generate_protocol.nodes.bgm_anchor_planning_node import BGMAanchorPlanningNode
from video_generate_protocol.nodes.asset_request_node import AssetRequestNode
from video_generate_protocol.nodes.bgm_composition_node import BGMCompositionNode
from video_generate_protocol.nodes.sfx_integration_node import SFXIntegrationNode
from video_generate_protocol.nodes.subtitle_node import SubtitleNode
from video_generate_protocol.nodes.aux_text_insertion_node import AuxTextInsertionNode
from video_generate_protocol.nodes.intro_outro_node import IntroOutroNode
from video_generate_protocol.nodes.timeline_integration_node import TimelineIntegrationNode


class VGPWorkflowTemplateNew:
    """
    新的视频生成协议工作流模板

    节点流程图:
    主干: 1 → 2 → 3
    分支:
    - 3 → 4 → 9 → 11 → 16
    - 3 → 5 → 6 → 7 → 8 → 16
    - 3 → 10 → 11 → 16
    - 3 → 12 → 16
    - 3 → 13 → 16
    - 3 → 14 → 11 → 16
    - 3 → 15 → 16

    节点说明:
    - Node 1: 情感分析
    - Node 2: 视频类型识别
    - Node 3: 分镜块生成
    - Node 4: BGM锚点规划
    - Node 5: 素材请求（AssetRequestNode - 在这里生成素材）
    - Node 6: 滤镜应用
    - Node 7: 动态效果
    - Node 8: 转场选择
    - Node 9: BGM合成查找（BGMCompositionNode）
    - Node 10: 音效添加（SFXIntegrationNode）
    - Node 11: 音频处理汇聚节点
    - Node 12: TTS合成
    - Node 13: 额外文字插入（AuxTextInsertionNode）
    - Node 14: 字幕节点（SubtitleNode）
    - Node 15: 片头片尾生成（IntroOutroNode）
    - Node 16: 时间轴整合（TimelineIntegrationNode - 最终合成节点）

    重要说明:
    - 节点5是素材生成的核心节点，所有素材在这里生成
    - 节点9-15只负责确定要不要添加、要添加什么（包括素材生成），不进行最终合成
    - 节点16是唯一的合成节点
    """

    @staticmethod
    def create_new_pipeline(config: WorkflowConfig) -> WorkflowOrchestrator:
        """
        创建按照新流程图的完整视频生成管线
        """
        orchestrator = WorkflowOrchestrator(config)

        # Phase 1: 主干节点 (1 → 2 → 3)
        VGPWorkflowTemplateNew._add_backbone_nodes(orchestrator)

        # Phase 2: 素材生成与处理分支 (3 → 5 → 6 → 7 → 8 → 16)
        VGPWorkflowTemplateNew._add_asset_processing_branch(orchestrator)

        # Phase 3: BGM分支 (3 → 4 → 9 → 11 → 16)
        VGPWorkflowTemplateNew._add_bgm_branch(orchestrator)

        # Phase 4: 音效分支 (3 → 10 → 11 → 16)
        VGPWorkflowTemplateNew._add_sfx_branch(orchestrator)

        # Phase 5: 字幕与TTS分支 (3 → 14 → 11 → 16 和 3 → 12 → 16)
        VGPWorkflowTemplateNew._add_subtitle_tts_branch(orchestrator)

        # Phase 6: 辅助文字分支 (3 → 13 → 16)
        VGPWorkflowTemplateNew._add_aux_text_branch(orchestrator)

        # Phase 7: 片头片尾分支 (3 → 15 → 16)
        VGPWorkflowTemplateNew._add_intro_outro_branch(orchestrator)

        # Phase 8: 音频汇聚节点 (11)
        VGPWorkflowTemplateNew._add_audio_merge_node(orchestrator)

        # Phase 9: 最终整合节点 (16)
        VGPWorkflowTemplateNew._add_final_integration_node(orchestrator)

        return orchestrator

    @staticmethod
    def _add_backbone_nodes(orchestrator: WorkflowOrchestrator):
        """主干节点: 1 → 2 → 3"""

        # Node 1: 情感分析节点
        emotion_node = EmotionAnalysisNode(
            node_id="node_1_emotion_analysis",
            name="情感分析"
        )
        orchestrator.add_node(emotion_node, dependencies=[])

        # Node 2: 视频类型识别节点
        video_type_node = VideoTypeIdentificationNode(
            node_id="node_2_video_type",
            name="视频类型识别"
        )
        orchestrator.add_node(video_type_node, dependencies=[])

        # Node 3: 分镜块生成节点（依赖1和2）
        shot_block_node = ShotBlockGenerationNode(
            node_id="node_3_shot_blocks",
            name="分镜块生成"
        )
        orchestrator.add_node(shot_block_node, dependencies=[
            "node_1_emotion_analysis",
            "node_2_video_type"
        ])

    @staticmethod
    def _add_asset_processing_branch(orchestrator: WorkflowOrchestrator):
        """
        素材生成与处理分支: 3 → 5 → 6 → 7 → 8 → 16
        Node 5是素材请求节点，这里生成所有素材
        """

        # Node 5: 素材请求节点（AssetRequestNode）
        # 这是素材生成的核心节点，在这里请求并生成所有素材
        asset_request_node = AssetRequestNode(
            node_id="node_5_asset_request",
            name="素材请求与生成"
        )
        orchestrator.add_node(asset_request_node, dependencies=["node_3_shot_blocks"])

        # Node 6: 滤镜应用节点
        filter_config = NodeConfig(
            node_id="node_6_filter_application",
            name="滤镜应用",
            description="根据素材检测结果决定是否添加滤镜",
            priority=ProcessingPriority.NORMAL,
            timeout=120.0
        )
        filter_node = VGPWorkflowTemplateNew._create_mock_node(filter_config)
        orchestrator.add_node(filter_node, dependencies=["node_5_asset_request"])

        # Node 7: 动态效果节点
        effects_config = NodeConfig(
            node_id="node_7_dynamic_effects",
            name="动态效果",
            description="添加动态视觉效果",
            priority=ProcessingPriority.NORMAL,
            timeout=120.0
        )
        effects_node = VGPWorkflowTemplateNew._create_mock_node(effects_config)
        orchestrator.add_node(effects_node, dependencies=["node_6_filter_application"])

        # Node 8: 转场选择节点
        transition_config = NodeConfig(
            node_id="node_8_transition_selection",
            name="转场选择",
            description="选择合适的转场效果",
            priority=ProcessingPriority.NORMAL,
            timeout=90.0
        )
        transition_node = VGPWorkflowTemplateNew._create_mock_node(transition_config)
        orchestrator.add_node(transition_node, dependencies=["node_7_dynamic_effects"])

    @staticmethod
    def _add_bgm_branch(orchestrator: WorkflowOrchestrator):
        """BGM分支: 3 → 4 → 9 → 11"""

        # Node 4: BGM锚点规划节点
        bgm_anchor_node = BGMAanchorPlanningNode(
            node_id="node_4_bgm_anchor",
            name="BGM锚点规划"
        )
        orchestrator.add_node(bgm_anchor_node, dependencies=["node_3_shot_blocks"])

        # Node 9: BGM合成查找节点（BGMCompositionNode）
        # 只负责匹配BGM，不进行最终合成
        bgm_composition_node = BGMCompositionNode(
            node_id="node_9_bgm_composition",
            name="BGM合成查找"
        )
        orchestrator.add_node(bgm_composition_node, dependencies=["node_4_bgm_anchor"])

    @staticmethod
    def _add_sfx_branch(orchestrator: WorkflowOrchestrator):
        """音效分支: 3 → 10 → 11"""

        # Node 10: 音效添加节点（SFXIntegrationNode）
        # 只负责匹配音效，不进行最终合成
        sfx_node = SFXIntegrationNode(
            node_id="node_10_sfx_integration",
            name="音效添加"
        )
        orchestrator.add_node(sfx_node, dependencies=["node_3_shot_blocks"])

    @staticmethod
    def _add_subtitle_tts_branch(orchestrator: WorkflowOrchestrator):
        """字幕与TTS分支: 3 → 14 → 11 和 3 → 12 → 16"""

        # Node 12: TTS合成节点
        tts_config = NodeConfig(
            node_id="node_12_tts_synthesis",
            name="TTS合成",
            description="生成TTS语音，但不进行最终合成",
            priority=ProcessingPriority.NORMAL,
            timeout=180.0
        )
        tts_node = VGPWorkflowTemplateNew._create_mock_node(tts_config)
        orchestrator.add_node(tts_node, dependencies=["node_3_shot_blocks"])

        # Node 14: 字幕节点（SubtitleNode）
        # 负责添加字幕与TTS，但不进行最终合成
        subtitle_node = SubtitleNode(
            node_id="node_14_subtitle",
            name="字幕生成"
        )
        orchestrator.add_node(subtitle_node, dependencies=["node_3_shot_blocks"])

    @staticmethod
    def _add_aux_text_branch(orchestrator: WorkflowOrchestrator):
        """辅助文字分支: 3 → 13 → 16"""

        # Node 13: 额外文字插入节点（AuxTextInsertionNode）
        # 只负责确定要添加的装饰文字，不进行最终合成
        aux_text_node = AuxTextInsertionNode(
            node_id="node_13_aux_text",
            name="额外文字插入"
        )
        orchestrator.add_node(aux_text_node, dependencies=["node_3_shot_blocks"])

    @staticmethod
    def _add_intro_outro_branch(orchestrator: WorkflowOrchestrator):
        """片头片尾分支: 3 → 15 → 16"""

        # Node 15: 片头片尾生成节点（IntroOutroNode）
        # 只负责生成片头片尾素材，不进行最终合成
        intro_outro_node = IntroOutroNode(
            node_id="node_15_intro_outro",
            name="片头片尾生成"
        )
        orchestrator.add_node(intro_outro_node, dependencies=["node_3_shot_blocks"])

    @staticmethod
    def _add_audio_merge_node(orchestrator: WorkflowOrchestrator):
        """
        音频汇聚节点 (Node 11): 汇聚来自 BGM(9)、音效(10)、字幕(14) 的音频轨道
        这个节点不进行最终合成，只是汇总音频信息
        """
        audio_merge_config = NodeConfig(
            node_id="node_11_audio_merge",
            name="音频汇聚",
            description="汇聚所有音频轨道信息（BGM、音效、TTS），准备传给最终合成节点",
            priority=ProcessingPriority.NORMAL,
            timeout=60.0
        )
        audio_merge_node = VGPWorkflowTemplateNew._create_mock_node(audio_merge_config)
        orchestrator.add_node(audio_merge_node, dependencies=[
            "node_9_bgm_composition",  # BGM
            "node_10_sfx_integration",  # 音效
            "node_14_subtitle"  # 字幕（包含TTS）
        ])

    @staticmethod
    def _add_final_integration_node(orchestrator: WorkflowOrchestrator):
        """
        最终整合节点 (Node 16): TimelineIntegrationNode
        这是唯一的合成节点，汇聚所有素材和效果进行最终合成
        """
        # TimelineIntegrationNode 直接使用字符串参数初始化
        timeline_node = TimelineIntegrationNode(
            node_id="node_16_timeline_integration",
            name="时间轴整合"
        )

        orchestrator.add_node(timeline_node, dependencies=[
            "node_8_transition_selection",  # 素材处理链的最后一步
            "node_11_audio_merge",  # 音频汇聚
            "node_12_tts_synthesis",  # TTS
            "node_13_aux_text",  # 辅助文字
            "node_15_intro_outro"  # 片头片尾
        ])

    @staticmethod
    def _create_mock_node(config: NodeConfig):
        """为尚未实现的节点创建mock节点"""
        from nodes.base_node import BaseNode, NodeResult, NodeStatus, ProcessingContext

        class MockVGPNode(BaseNode):
            def _normalize_audio_clip(self, clip):
                """标准化音频 clip 格式，将 start_time/end_time 转换为 start/end"""
                if not isinstance(clip, dict):
                    return None

                normalized = clip.copy()

                # 转换字段名: start_time -> start, end_time -> end
                if "start_time" in normalized:
                    normalized["start"] = normalized.pop("start_time")
                if "end_time" in normalized:
                    normalized["end"] = normalized.pop("end_time")

                # 确保有 start 和 end 字段
                if "start" not in normalized or "end" not in normalized:
                    return None

                return normalized

            async def process(self, context: ProcessingContext) -> NodeResult:
                # 模拟处理时间
                import asyncio
                await asyncio.sleep(0.5)

                # 根据节点ID输出特定格式的数据
                data = {
                    f"{self.node_id}_result": f"Mock processing completed for {self.node_name}",
                    "processed_at": context.timestamp.isoformat(),
                    "mock_data": {
                        "confidence": 0.95,
                        "processing_info": f"Simulated {self.node_id} processing"
                    }
                }

                # Node 8: 转场选择节点 - 应该输出 video_clips
                if self.node_id == "node_8_transition_selection":
                    # 从 context 中获取 preliminary_sequence_id (Node 5 的输出)
                    preliminary_seq = context.intermediate_results.get("preliminary_sequence_id", [])

                    # 如果素材序列为空，创建占位符（用于测试）
                    if not preliminary_seq:
                        shot_blocks = context.intermediate_results.get("shot_blocks_id", [])
                        if shot_blocks:
                            import uuid
                            preliminary_seq = []
                            current_time = 0.0
                            for idx, block in enumerate(shot_blocks[:3]):  # 只取前3个作为示例
                                duration = block.get("duration", 3.0)
                                preliminary_seq.append({
                                    "id": f"placeholder_{uuid.uuid4().hex[:8]}",
                                    "start": current_time,
                                    "end": current_time + duration,
                                    "duration": duration,
                                    "source_url": "https://example.com/placeholder.mp4",
                                    "metadata": {"placeholder": True}
                                })
                                current_time += duration

                    data["video_clips"] = preliminary_seq  # 传递给 Node 16

                # Node 11: 音频汇聚节点 - 应该输出 audio_tracks
                elif self.node_id == "node_11_audio_merge":
                    # 汇聚 BGM、音效、TTS 轨道
                    bgm_tracks = context.intermediate_results.get("bgm_tracks_id", [])
                    sfx_track = context.intermediate_results.get("sfx_track", {})
                    tts_track = context.intermediate_results.get("tts_track_id", {})
                    subtitle_seq = context.intermediate_results.get("subtitle_sequence_id", {})

                    audio_tracks = []

                    # 添加 BGM 轨道（标准化字段名）
                    if bgm_tracks:
                        for bgm in bgm_tracks if isinstance(bgm_tracks, list) else [bgm_tracks]:
                            # 标准化 BGM clip 格式：start_time -> start, end_time -> end
                            normalized_clip = self._normalize_audio_clip(bgm)
                            audio_tracks.append({
                                "track_name": "BGM",
                                "track_type": "background_music",
                                "layer": 0,
                                "clips": [normalized_clip] if normalized_clip else []
                            })

                    # 添加音效轨道（标准化字段名）
                    if sfx_track and isinstance(sfx_track, dict):
                        clips = sfx_track.get("clips", [])
                        if clips:
                            normalized_clips = [self._normalize_audio_clip(c) for c in clips]
                            audio_tracks.append({
                                "track_name": "SFX",
                                "track_type": "sound_effects",
                                "layer": 1,
                                "clips": [c for c in normalized_clips if c]  # 过滤掉 None
                            })

                    # 添加 TTS 轨道（从字幕序列中提取，标准化字段名）
                    if subtitle_seq and isinstance(subtitle_seq, dict):
                        tts_clips = subtitle_seq.get("tts_clips", [])
                        if tts_clips:
                            normalized_clips = [self._normalize_audio_clip(c) for c in tts_clips]
                            audio_tracks.append({
                                "track_name": "TTS",
                                "track_type": "voice",
                                "layer": 2,
                                "clips": [c for c in normalized_clips if c]
                            })

                    data["audio_tracks"] = audio_tracks

                return NodeResult(
                    status=NodeStatus.COMPLETED,
                    data=data,
                    next_nodes=[]
                )

            def validate_input(self, context: ProcessingContext) -> bool:
                return True

            def get_required_inputs(self) -> List[str]:
                return []

            def get_output_schema(self) -> Dict[str, Any]:
                schema = {
                    f"{self.node_id}_result": "string",
                    "processed_at": "string",
                    "mock_data": "object"
                }

                # 根据节点ID添加特定输出
                if self.node_id == "node_8_transition_selection":
                    schema["video_clips"] = "list"
                elif self.node_id == "node_11_audio_merge":
                    schema["audio_tracks"] = "list"

                return schema

        return MockVGPNode(config)


def register_new_vgp_workflow_templates(workflow_manager):
    """注册新的VGP工作流模板"""

    # 注册新的完整管线模板
    workflow_manager.register_workflow_template(
        "vgp_new_pipeline",
        VGPWorkflowTemplateNew.create_new_pipeline
    )

    print("✅ 新的VGP工作流模板注册成功")
    print("   - vgp_new_pipeline: 按照新流程图的完整视频生成管线")
    print("   - 节点5(AssetRequestNode)负责素材生成")
    print("   - 节点9-15只负责生成素材，不进行合成")
    print("   - 节点16(TimelineIntegrationNode)负责最终合成")
