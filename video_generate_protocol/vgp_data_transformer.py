#!/usr/bin/env python3
"""
VGP Data Transformer - 节点间数据转换和映射
解决节点间数据传递不兼容的问题
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# =============================
# 节点输出字段映射定义
# =============================

class VGPDataMapper:
    """VGP节点数据映射器 - 确保节点间数据正确传递"""

    # 定义每个节点应该输出的标准字段
    NODE_OUTPUT_SCHEMA = {
        'conversation_context': {
            'required': ['conversation_id', 'user_description', 'theme', 'keywords', 'target_duration'],
            'optional': ['context_metadata', 'conversation_history']
        },
        'video_type_identification': {
            'required': ['video_type_id', 'structure_template_id'],
            'optional': ['confidence_score', 'sub_type']
        },
        'emotion_analysis': {
            'required': ['emotions_id', 'emotion_curve', 'primary_emotion'],
            'optional': ['emotion_intensity', 'emotion_transitions']
        },
        'shot_block_generation': {
            'required': ['shot_blocks_id'],
            'optional': ['shot_transitions', 'shot_metadata']
        },
        'bgm_anchor_planning': {
            'required': ['bgm_anchors', 'bgm_timeline', 'bgm_segments'],
            'optional': ['bgm_mood_curve', 'bgm_energy_levels']
        },
        'bgm_composition': {
            'required': ['bgm_composition_id', 'bgm_file_path', 'bgm_duration'],
            'optional': ['bgm_metadata', 'bgm_tags']
        },
        'asset_request': {
            'required': ['assets', 'asset_timeline', 'asset_count'],
            'optional': ['asset_categories', 'asset_priorities']
        },
        'audio_processing': {
            'required': ['audio_track_id', 'audio_file_path', 'audio_duration'],
            'optional': ['audio_metadata', 'audio_effects']
        },
        'sfx_integration': {
            'required': ['sfx_track_id', 'sfx_timeline', 'sfx_effects'],
            'optional': ['sfx_metadata', 'sfx_volume_curve']
        },
        'transition_selection': {
            'required': ['transitions', 'transition_timeline', 'transition_types'],
            'optional': ['transition_durations', 'transition_parameters']
        },
        'filter_application': {
            'required': ['filters', 'filter_timeline', 'filter_parameters'],
            'optional': ['filter_intensity', 'filter_metadata']
        },
        'dynamic_effects': {
            'required': ['effects', 'effect_timeline', 'effect_parameters'],
            'optional': ['effect_layers', 'effect_priorities']
        },
        'aux_media_insertion': {
            'required': ['aux_media', 'aux_media_timeline', 'aux_media_types'],
            'optional': ['aux_media_metadata', 'aux_media_sources']
        },
        'aux_text_insertion': {
            'required': ['text_overlays', 'text_timeline', 'text_styles'],
            'optional': ['text_animations', 'text_metadata']
        },
        'subtitle_generation': {
            'required': ['subtitles', 'subtitle_timeline', 'subtitle_text'],
            'optional': ['subtitle_styles', 'subtitle_languages']
        },
        'intro_outro': {
            'required': ['intro_data', 'outro_data', 'intro_outro_duration'],
            'optional': ['intro_style', 'outro_style', 'branding_elements']
        },
        'timeline_integration': {
            'required': ['final_timeline', 'timeline_data', 'total_duration'],
            'optional': ['timeline_metadata', 'timeline_validation']
        }
    }

    # 节点间数据依赖映射
    NODE_DEPENDENCIES = {
        'audio_processing': {
            'bgm_composition': ['bgm_composition_id', 'bgm_file_path'],
            'sfx_integration': ['sfx_track_id', 'sfx_timeline']
        },
        'shot_block_generation': {
            'video_type_identification': ['video_type_id', 'structure_template_id'],
            'emotion_analysis': ['emotions_id']
        },
        'asset_request': {
            'shot_block_generation': ['scheduled_shots', 'shot_blocks_id']  # 修正：使用正确的字段名 shot_blocks_id
        },
        'timeline_integration': {
            'asset_request': ['assets', 'asset_timeline', 'video_clips', 'keyframes'],  # 添加 video_clips 和 keyframes
            'audio_processing': ['audio_tracks'],  # 添加音频轨道
            'transition_selection': ['transitions', 'transition_timeline'],
            'filter_application': ['filters', 'filter_timeline'],
            'subtitle_generation': ['subtitles', 'subtitle_timeline'],
            'intro_outro': ['intro_outro_sequence']  # 添加片头片尾
        }
    }

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VGPDataMapper")

    def transform_node_output(self, node_type: str, raw_output: Dict[str, Any]) -> Dict[str, Any]:
        """
        转换节点输出为标准格式

        Args:
            node_type: 节点类型
            raw_output: 原始输出数据

        Returns:
            标准化的输出数据
        """
        if node_type not in self.NODE_OUTPUT_SCHEMA:
            self.logger.warning(f"Unknown node type: {node_type}")
            return raw_output

        schema = self.NODE_OUTPUT_SCHEMA[node_type]
        transformed = {}

        # 确保必需字段存在
        for field in schema['required']:
            if field in raw_output:
                transformed[field] = raw_output[field]
            else:
                # 生成默认值
                transformed[field] = self._generate_default_value(node_type, field)
                self.logger.warning(f"Missing required field '{field}' for {node_type}, using default")

        # 复制可选字段
        for field in schema.get('optional', []):
            if field in raw_output:
                transformed[field] = raw_output[field]

        # 保留其他字段（向后兼容）
        for key, value in raw_output.items():
            if key not in transformed:
                transformed[key] = value

        return transformed

    def prepare_node_input(self, node_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        为节点准备输入数据，从上下文中提取所需的依赖数据

        Args:
            node_type: 节点类型
            context: 包含所有节点输出的上下文

        Returns:
            准备好的输入数据
        """
        input_data = {}

        # 如果节点有依赖
        if node_type in self.NODE_DEPENDENCIES:
            deps = self.NODE_DEPENDENCIES[node_type]

            for dep_node, dep_fields in deps.items():
                # 检查依赖节点的输出是否存在
                if dep_node in context:
                    dep_output = context[dep_node]

                    # 提取所需字段
                    for field in dep_fields:
                        if field in dep_output:
                            input_data[field] = dep_output[field]
                        else:
                            # 尝试从其他可能的位置查找
                            value = self._find_field_in_context(field, context)
                            if value is not None:
                                input_data[field] = value
                            else:
                                # 生成默认值
                                input_data[field] = self._generate_default_value(dep_node, field)
                                self.logger.warning(
                                    f"Missing dependency field '{field}' from {dep_node} for {node_type}"
                                )

        # 添加基础上下文信息 - 处理参数名映射
        # 通用的标准字段名到带_id后缀字段名的映射
        field_mappings = {
            'theme': 'theme_id',
            'keywords': 'keywords_id',
            'target_duration': 'target_duration_id',
            'user_description': 'user_description_id'
        }

        # 首先尝试直接匹配带_id后缀的字段
        for key in ['theme_id', 'keywords_id', 'target_duration_id', 'user_description_id']:
            if key in context:
                input_data[key] = context[key]

        # 然后尝试从标准字段名映射到_id字段名
        for standard_name, id_name in field_mappings.items():
            if id_name not in input_data:  # 如果还没有找到带_id的字段
                # 先查找标准名称
                value = self._find_field_in_context(standard_name, context)
                if value is not None:
                    input_data[id_name] = value
                    self.logger.info(f"映射字段 {standard_name} -> {id_name}: {value}")
                # 如果在conversation_context中有该字段
                elif 'conversation_context' in context and isinstance(context['conversation_context'], dict):
                    conv_context = context['conversation_context']
                    if standard_name in conv_context:
                        input_data[id_name] = conv_context[standard_name]
                        self.logger.info(f"从conversation_context映射字段 {standard_name} -> {id_name}: {conv_context[standard_name]}")

        # 确保对于conversation_context节点，也要输出标准字段名
        if node_type == 'conversation_context':
            # conversation_context节点需要输出标准字段名
            for standard_name in field_mappings.keys():
                value = self._find_field_in_context(standard_name, context)
                if value is not None and standard_name not in input_data:
                    input_data[standard_name] = value

        return input_data

    def _find_field_in_context(self, field: str, context: Dict[str, Any]) -> Any:
        """在整个上下文中查找字段"""
        # 直接在顶层查找
        if field in context:
            return context[field]

        # 在所有节点输出中查找
        for node_name, node_output in context.items():
            if isinstance(node_output, dict) and field in node_output:
                return node_output[field]

        return None

    def _generate_default_value(self, node_type: str, field: str) -> Any:
        """生成字段的默认值"""
        # 特定字段的默认值
        default_values = {
            # 基础输入参数默认值
            'theme_id': '通用视频',
            'keywords_id': ['视频', '内容'],
            'target_duration_id': 30,
            'user_description_id': '用户输入描述',
            'conversation_id': 'default_conversation_001',
            'user_description': '默认用户描述',
            'theme': '通用主题',
            'keywords': ['默认', '关键词'],
            'target_duration': 30,
            # 节点输出默认值
            'bgm_composition_id': 'default_bgm_001',
            'sfx_track_id': 'default_sfx_001',
            'scheduled_shots': [],
            'shot_blocks': [],
            'assets': [],
            'emotions_id': {'emotions': {'primary': 'neutral', 'secondary': 'calm'}},
            'emotion_curve': ['neutral'] * 10,
            'video_type_id': 'general',
            'structure_template_id': {'intro': '', 'body': '', 'conclusion': ''},
            'bgm_anchors': [],
            'transitions': [],
            'filters': [],
            'effects': [],
            'subtitles': [],
            'audio_tracks': [],  # ✅ 新增：audio_tracks默认为空列表，确保类型正确
            'audio_track_id': 'default_audio_001',
            'audio_file_path': '/tmp/default_audio.mp3',
            'bgm_file_path': '/tmp/default_bgm.mp3',
            'timeline_data': {},
            'final_timeline': {},
            'total_duration': 30,
            'bgm_duration': 30,
            'audio_duration': 30
        }

        if field in default_values:
            return default_values[field]

        # 根据字段名模式生成默认值
        if field.endswith('_id'):
            return f"default_{field}"
        elif field.endswith('_timeline'):
            return []
        elif field.endswith('_count'):
            return 0
        elif field.endswith('_duration'):
            return 30
        elif field.endswith('_path'):
            return f"/tmp/default_{field}.tmp"
        else:
            return {}

    def validate_node_chain(self, nodes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        验证并修复节点链的数据传递

        Args:
            nodes: 节点列表

        Returns:
            修复后的节点列表
        """
        context = {}
        fixed_nodes = []

        for node in nodes:
            node_type = node.get('node_type', '')

            # 准备输入数据
            if 'input_data' not in node or not node['input_data']:
                node['input_data'] = self.prepare_node_input(node_type, context)

            # 转换输出数据
            if 'output_data' in node and node['output_data']:
                node['output_data'] = self.transform_node_output(node_type, node['output_data'])

                # 更新上下文
                context[node_type] = node['output_data']

            fixed_nodes.append(node)

        return fixed_nodes

# =============================
# VGP Data Fixer - 修复现有数据问题
# =============================

class VGPDataFixer:
    """修复VGP节点数据传递问题"""

    def __init__(self):
        self.mapper = VGPDataMapper()
        self.logger = logging.getLogger(f"{__name__}.VGPDataFixer")

    def fix_bgm_composition_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """修复bgm_composition节点输出"""
        # 确保有bgm_composition_id
        if 'bgm_composition_id' not in output_data:
            if 'bgm_id' in output_data:
                output_data['bgm_composition_id'] = output_data['bgm_id']
            elif 'composition_id' in output_data:
                output_data['bgm_composition_id'] = output_data['composition_id']
            else:
                output_data['bgm_composition_id'] = 'bgm_comp_' + str(hash(str(output_data)))[:8]

        # 确保有bgm_file_path
        if 'bgm_file_path' not in output_data:
            if 'file_path' in output_data:
                output_data['bgm_file_path'] = output_data['file_path']
            elif 'bgm_path' in output_data:
                output_data['bgm_file_path'] = output_data['bgm_path']
            else:
                output_data['bgm_file_path'] = f"/tmp/bgm_{output_data['bgm_composition_id']}.mp3"

        return output_data

    def fix_sfx_integration_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """修复sfx_integration节点输出"""
        # 确保有sfx_track_id
        if 'sfx_track_id' not in output_data:
            if 'sfx_id' in output_data:
                output_data['sfx_track_id'] = output_data['sfx_id']
            elif 'track_id' in output_data:
                output_data['sfx_track_id'] = output_data['track_id']
            else:
                output_data['sfx_track_id'] = 'sfx_track_' + str(hash(str(output_data)))[:8]

        # 确保有sfx_timeline
        if 'sfx_timeline' not in output_data:
            if 'timeline' in output_data:
                output_data['sfx_timeline'] = output_data['timeline']
            elif 'sfx_events' in output_data:
                output_data['sfx_timeline'] = output_data['sfx_events']
            else:
                output_data['sfx_timeline'] = []

        return output_data

    def fix_shot_block_generation_output(self, output_data: Dict[str, Any]) -> Dict[str, Any]:
        """修复shot_block_generation节点输出"""
        # 确保有scheduled_shots
        if 'scheduled_shots' not in output_data:
            if 'shots' in output_data:
                output_data['scheduled_shots'] = output_data['shots']
            elif 'shot_list' in output_data:
                output_data['scheduled_shots'] = output_data['shot_list']
            elif 'shot_blocks_id' in output_data:  # 修正：使用正确的字段名 shot_blocks_id
                # 从shot_blocks_id生成scheduled_shots
                output_data['scheduled_shots'] = [
                    {
                        'shot_id': f"shot_{i}",
                        'duration': block.get('duration', 2),
                        'description': block.get('description', ''),
                        'type': block.get('type', 'normal')
                    }
                    for i, block in enumerate(output_data['shot_blocks_id'])
                ]
            elif 'shot_blocks' in output_data:  # 保留兼容性
                # 从shot_blocks生成scheduled_shots
                output_data['scheduled_shots'] = [
                    {
                        'shot_id': f"shot_{i}",
                        'duration': block.get('duration', 2),
                        'description': block.get('description', ''),
                        'type': block.get('type', 'normal')
                    }
                    for i, block in enumerate(output_data['shot_blocks'])
                ]
            else:
                output_data['scheduled_shots'] = []

        return output_data

    def fix_all_node_outputs(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """修复所有节点输出数据,同时保留非节点的原始数据"""
        fixed_context = {}

        # 定义已知的VGP节点类型
        known_node_types = {
            'video_type_identification', 'emotion_analysis', 'shot_block_generation',
            'bgm_anchor_planning', 'bgm_composition', 'asset_request',
            'audio_processing', 'sfx_integration', 'transition_selection',
            'filter_application', 'dynamic_effects', 'aux_media_insertion',
            'aux_text_insertion', 'subtitle_generation', 'intro_outro',
            'timeline_integration', 'multimodal_fusion', 'conversation_context'
        }

        for key, value in context.items():
            # 如果是已知节点类型且是 dict,进行修复和转换
            if key in known_node_types and isinstance(value, dict):
                # 应用特定节点的修复
                if key == 'bgm_composition':
                    value = self.fix_bgm_composition_output(value)
                elif key == 'sfx_integration':
                    value = self.fix_sfx_integration_output(value)
                elif key == 'shot_block_generation':
                    value = self.fix_shot_block_generation_output(value)

                # 应用通用转换
                value = self.mapper.transform_node_output(key, value)

            # 保留所有数据(包括原始输入参数如 theme_id, keywords_id等)
            fixed_context[key] = value

        return fixed_context

# =============================
# 集成到主流程的辅助函数
# =============================

def create_data_transformer() -> VGPDataMapper:
    """创建数据转换器实例"""
    return VGPDataMapper()

def create_data_fixer() -> VGPDataFixer:
    """创建数据修复器实例"""
    return VGPDataFixer()

def prepare_context_for_node(node_type: str, raw_context: Dict[str, Any]) -> Dict[str, Any]:
    """
    为特定节点准备输入上下文

    Args:
        node_type: 节点类型
        raw_context: 原始上下文数据

    Returns:
        准备好的输入数据
    """
    transformer = create_data_transformer()
    fixer = create_data_fixer()

    # 先修复所有节点的输出
    fixed_context = fixer.fix_all_node_outputs(raw_context)

    # 然后准备节点输入
    input_data = transformer.prepare_node_input(node_type, fixed_context)

    # 合并原始上下文中的基础信息
    # 通用的标准字段名到带_id后缀字段名的映射
    field_mappings = {
        'theme': 'theme_id',
        'keywords': 'keywords_id',
        'target_duration': 'target_duration_id',
        'user_description': 'user_description_id'
    }

    # DEBUG: 打印 raw_context 的关键字段
    print(f"\n{'='*80}")
    print(f"🔍 [prepare_context_for_node] node_type={node_type}")
    print(f"🔍 [prepare_context_for_node] raw_context keys={list(raw_context.keys())}")
    print(f"🔍 [prepare_context_for_node] input_data (before)={list(input_data.keys())}")
    logger.info(f"prepare_context_for_node: node_type={node_type}, raw_context keys={list(raw_context.keys())}")

    # 先处理直接匹配的_id字段
    for key in ['theme_id', 'keywords_id', 'target_duration_id', 'user_description_id']:
        if key in raw_context:
            print(f"🔍 [prepare_context_for_node] 发现字段 {key} in raw_context: {raw_context[key]}")
            if key not in input_data:
                input_data[key] = raw_context[key]
                print(f"✅ [prepare_context_for_node] 复制字段 {key}: {raw_context[key]}")
                logger.info(f"prepare_context_for_node: 从 raw_context 复制字段 {key}: {raw_context[key]}")
        else:
            print(f"❌ [prepare_context_for_node] 字段 {key} NOT in raw_context")

    # 然后处理标准字段名到_id字段的映射
    for standard_name, id_name in field_mappings.items():
        if id_name not in input_data and standard_name in raw_context:
            input_data[id_name] = raw_context[standard_name]
            logger.info(f"prepare_context_for_node: 映射字段 {standard_name} -> {id_name}: {raw_context[standard_name]}")

    # DEBUG: 打印最终的 input_data
    print(f"🔍 [prepare_context_for_node] input_data (after)={list(input_data.keys())}")
    print(f"🔍 [prepare_context_for_node] Final values:")
    for key in ['theme_id', 'keywords_id', 'target_duration_id', 'user_description_id']:
        print(f"    {key}: {input_data.get(key, 'NOT_FOUND')}")
    print(f"{'='*80}\n")
    logger.info(f"prepare_context_for_node: 最终准备的 input_data keys={list(input_data.keys())}")

    # 对于conversation_context节点，确保输出标准字段名
    if node_type == 'conversation_context':
        for standard_name in field_mappings.keys():
            if standard_name in raw_context and standard_name not in input_data:
                input_data[standard_name] = raw_context[standard_name]

    return input_data

def transform_node_result(node_type: str, raw_output: Dict[str, Any]) -> Dict[str, Any]:
    """
    转换节点输出为标准格式

    Args:
        node_type: 节点类型
        raw_output: 原始输出

    Returns:
        标准化的输出
    """
    transformer = create_data_transformer()
    return transformer.transform_node_output(node_type, raw_output)

# 测试函数
def test_data_transformer():
    """测试数据转换器"""
    transformer = create_data_transformer()
    fixer = create_data_fixer()

    # 模拟节点输出
    context = {
        'bgm_composition': {
            'composition_id': 'bgm_123',
            'file_path': '/tmp/bgm.mp3'
        },
        'sfx_integration': {
            'sfx_id': 'sfx_456',
            'timeline': [{'time': 0, 'effect': 'boom'}]
        },
        'shot_block_generation': {
            'shot_blocks': [
                {'duration': 3, 'description': 'Opening shot'},
                {'duration': 2, 'description': 'Main content'}
            ]
        }
    }

    # 修复数据
    fixed_context = fixer.fix_all_node_outputs(context)

    print("Fixed context:")
    print(json.dumps(fixed_context, indent=2, ensure_ascii=False))

    # 准备audio_processing的输入
    audio_input = transformer.prepare_node_input('audio_processing', fixed_context)

    print("\nAudio processing input:")
    print(json.dumps(audio_input, indent=2, ensure_ascii=False))

    return fixed_context

if __name__ == "__main__":
    import json
    logging.basicConfig(level=logging.INFO)
    test_data_transformer()