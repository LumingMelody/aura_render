"""
17节点依赖关系管理器
管理视频生成流程中17个VGP节点的依赖关系和增量执行逻辑
"""

import logging
from typing import Dict, List, Set, Any, Optional, Tuple
from enum import Enum
from dataclasses import dataclass
from datetime import datetime
import networkx as nx

logger = logging.getLogger(__name__)


class NodeStatus(Enum):
    """节点状态"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"
    REUSED = "reused"


class ModificationImpact(Enum):
    """修改影响级别"""
    NONE = "none"          # 无影响
    LOW = "low"            # 低影响 - 可复用结果
    MEDIUM = "medium"      # 中影响 - 需要调整
    HIGH = "high"          # 高影响 - 需要重新执行
    CRITICAL = "critical"  # 关键影响 - 影响后续所有节点


@dataclass
class NodeDefinition:
    """节点定义"""
    node_id: str
    name: str
    description: str
    dependencies: List[str]  # 前置依赖节点
    affects: List[str]       # 影响的后续节点
    category: str           # 节点类别：structure, content, style, technical
    execution_time_estimate: float  # 预估执行时间（秒）
    can_skip_if: List[str]  # 在什么情况下可以跳过
    sensitive_to: List[str] # 对什么修改敏感


class NodeDependencyGraph:
    """节点依赖图管理器"""

    def __init__(self):
        self.graph = nx.DiGraph()
        self.nodes_definition = {}
        self.logger = logger.getChild('DependencyGraph')
        self._initialize_vgp_nodes()

    def _initialize_vgp_nodes(self):
        """初始化17个VGP节点的定义"""

        # 定义所有17个节点
        node_definitions = [
            NodeDefinition(
                node_id='video_type_identification',
                name='视频类型识别',
                description='识别要生成的视频类型和基本属性',
                dependencies=[],
                affects=['emotion_analysis', 'shot_block_generation'],
                category='structure',
                execution_time_estimate=5.0,
                can_skip_if=['user_provided_type'],
                sensitive_to=['theme', 'description']
            ),
            NodeDefinition(
                node_id='emotion_analysis',
                name='情感基调分析',
                description='分析视频的情感基调和氛围',
                dependencies=['video_type_identification'],
                affects=['bgm_composition', 'filter_application', 'dynamic_effects'],
                category='style',
                execution_time_estimate=8.0,
                can_skip_if=['style_unchanged'],
                sensitive_to=['keywords', 'mood', 'user_description']
            ),
            NodeDefinition(
                node_id='shot_block_generation',
                name='分镜块生成',
                description='生成视频的分镜脚本和结构',
                dependencies=['video_type_identification'],
                affects=['asset_request', 'subtitle_generation', 'timeline_integration'],
                category='structure',
                execution_time_estimate=15.0,
                can_skip_if=['structure_unchanged'],
                sensitive_to=['duration', 'description', 'structure_keywords']
            ),
            NodeDefinition(
                node_id='bgm_anchor_planning',
                name='BGM锚点规划',
                description='规划背景音乐的锚点和节奏',
                dependencies=['shot_block_generation'],
                affects=['bgm_composition', 'timeline_integration'],
                category='content',
                execution_time_estimate=10.0,
                can_skip_if=['bgm_unchanged'],
                sensitive_to=['duration', 'rhythm', 'emotion']
            ),
            NodeDefinition(
                node_id='bgm_composition',
                name='BGM合成查找',
                description='查找和合成背景音乐',
                dependencies=['emotion_analysis', 'bgm_anchor_planning'],
                affects=['audio_processing', 'timeline_integration'],
                category='content',
                execution_time_estimate=12.0,
                can_skip_if=['music_unchanged'],
                sensitive_to=['mood', 'style', 'duration']
            ),
            NodeDefinition(
                node_id='asset_request',
                name='素材需求解析',
                description='分析和解析所需的视觉素材',
                dependencies=['shot_block_generation'],
                affects=['aux_media_insertion'],
                category='content',
                execution_time_estimate=20.0,
                can_skip_if=['assets_unchanged'],
                sensitive_to=['visual_style', 'content_keywords', 'user_assets']
            ),
            NodeDefinition(
                node_id='audio_processing',
                name='音频处理',
                description='处理音频，包括音量调整和音效',
                dependencies=['bgm_composition'],
                affects=['sfx_integration', 'timeline_integration'],
                category='technical',
                execution_time_estimate=8.0,
                can_skip_if=['audio_unchanged'],
                sensitive_to=['volume', 'audio_effects']
            ),
            NodeDefinition(
                node_id='sfx_integration',
                name='音效添加',
                description='添加音效和声音特效',
                dependencies=['audio_processing', 'shot_block_generation'],
                affects=['timeline_integration'],
                category='content',
                execution_time_estimate=10.0,
                can_skip_if=['sfx_unchanged'],
                sensitive_to=['sound_effects', 'dramatic_moments']
            ),
            NodeDefinition(
                node_id='transition_selection',
                name='转场选择',
                description='选择和应用视频转场效果',
                dependencies=['shot_block_generation'],
                affects=['timeline_integration'],
                category='style',
                execution_time_estimate=6.0,
                can_skip_if=['transitions_unchanged'],
                sensitive_to=['style', 'rhythm', 'transition_type']
            ),
            NodeDefinition(
                node_id='filter_application',
                name='滤镜应用',
                description='应用视觉滤镜和色彩调整',
                dependencies=['emotion_analysis'],
                affects=['timeline_integration'],
                category='style',
                execution_time_estimate=12.0,
                can_skip_if=['visual_style_unchanged'],
                sensitive_to=['color_theme', 'mood', 'visual_style']
            ),
            NodeDefinition(
                node_id='dynamic_effects',
                name='动态特效添加',
                description='添加动态特效和动画',
                dependencies=['emotion_analysis', 'shot_block_generation'],
                affects=['timeline_integration'],
                category='style',
                execution_time_estimate=18.0,
                can_skip_if=['effects_unchanged'],
                sensitive_to=['animation_style', 'dynamic_elements']
            ),
            NodeDefinition(
                node_id='aux_media_insertion',
                name='额外媒体插入',
                description='插入额外的媒体素材',
                dependencies=['asset_request'],
                affects=['timeline_integration'],
                category='content',
                execution_time_estimate=15.0,
                can_skip_if=['media_unchanged'],
                sensitive_to=['user_assets', 'additional_media']
            ),
            NodeDefinition(
                node_id='aux_text_insertion',
                name='装饰文字插入',
                description='插入装饰性文字和图形',
                dependencies=['shot_block_generation'],
                affects=['timeline_integration'],
                category='content',
                execution_time_estimate=8.0,
                can_skip_if=['text_unchanged'],
                sensitive_to=['text_style', 'decorative_elements']
            ),
            NodeDefinition(
                node_id='subtitle_generation',
                name='字幕生成',
                description='生成和同步字幕',
                dependencies=['shot_block_generation'],
                affects=['timeline_integration'],
                category='content',
                execution_time_estimate=10.0,
                can_skip_if=['subtitles_unchanged'],
                sensitive_to=['text_content', 'subtitle_style']
            ),
            NodeDefinition(
                node_id='intro_outro',
                name='片头片尾生成',
                description='生成片头和片尾',
                dependencies=['emotion_analysis'],
                affects=['timeline_integration'],
                category='content',
                execution_time_estimate=12.0,
                can_skip_if=['intro_outro_unchanged'],
                sensitive_to=['branding', 'style']
            ),
            NodeDefinition(
                node_id='timeline_integration',
                name='最终时间线整合',
                description='整合所有元素到最终时间线',
                dependencies=[
                    'shot_block_generation', 'bgm_composition', 'sfx_integration',
                    'transition_selection', 'filter_application', 'dynamic_effects',
                    'aux_media_insertion', 'aux_text_insertion', 'subtitle_generation',
                    'intro_outro'
                ],
                affects=[],
                category='technical',
                execution_time_estimate=25.0,
                can_skip_if=[],  # 通常不能跳过
                sensitive_to=['all_changes']
            )
        ]

        # 添加节点到图中
        for node_def in node_definitions:
            self.nodes_definition[node_def.node_id] = node_def
            self.graph.add_node(node_def.node_id, **node_def.__dict__)

        # 添加依赖边
        for node_def in node_definitions:
            for dependency in node_def.dependencies:
                self.graph.add_edge(dependency, node_def.node_id)

        self.logger.info(f"Initialized {len(node_definitions)} VGP nodes with dependencies")

    def get_execution_order(self) -> List[str]:
        """获取节点的执行顺序（拓扑排序）"""
        try:
            return list(nx.topological_sort(self.graph))
        except nx.NetworkXError as e:
            self.logger.error(f"Dependency cycle detected: {e}")
            return []

    def get_dependencies(self, node_id: str) -> List[str]:
        """获取节点的直接依赖"""
        return list(self.graph.predecessors(node_id))

    def get_affected_nodes(self, node_id: str) -> List[str]:
        """获取受节点影响的所有后续节点"""
        return list(nx.descendants(self.graph, node_id))

    def get_impact_chain(self, modified_nodes: List[str]) -> Set[str]:
        """获取修改节点的完整影响链"""
        impact_chain = set(modified_nodes)

        for node_id in modified_nodes:
            affected = self.get_affected_nodes(node_id)
            impact_chain.update(affected)

        return impact_chain

    def can_skip_node(self, node_id: str, modification_context: Dict[str, Any]) -> bool:
        """判断节点是否可以跳过"""
        node_def = self.nodes_definition.get(node_id)
        if not node_def:
            return False

        # 检查跳过条件
        for skip_condition in node_def.can_skip_if:
            if modification_context.get(skip_condition, False):
                return True

        # 检查敏感性
        for sensitive_field in node_def.sensitive_to:
            if modification_context.get(f"modified_{sensitive_field}", False):
                return False

        return False

    def estimate_execution_time(self, nodes_to_execute: List[str]) -> float:
        """估算执行时间"""
        total_time = 0.0
        for node_id in nodes_to_execute:
            node_def = self.nodes_definition.get(node_id)
            if node_def:
                total_time += node_def.execution_time_estimate
        return total_time


class IncrementalExecutionPlanner:
    """增量执行规划器"""

    def __init__(self, dependency_graph: NodeDependencyGraph):
        self.dependency_graph = dependency_graph
        self.logger = logger.getChild('ExecutionPlanner')

    def plan_incremental_execution(self,
                                 modification_intent: Dict[str, Any],
                                 previous_execution: Dict[str, Any],
                                 conversation_context: Dict[str, Any]) -> Dict[str, Any]:
        """
        规划增量执行策略

        Args:
            modification_intent: 修改意图分析结果
            previous_execution: 上次执行的结果
            conversation_context: 对话上下文

        Returns:
            执行计划
        """
        self.logger.info("Planning incremental execution strategy")

        # 1. 分析修改影响
        impact_analysis = self._analyze_modification_impact(modification_intent)

        # 2. 确定起始节点
        start_nodes = self._determine_start_nodes(impact_analysis, previous_execution)

        # 3. 计算需要执行的节点
        nodes_to_execute = self._calculate_execution_nodes(start_nodes, impact_analysis)

        # 4. 确定可以跳过的节点
        nodes_to_skip = self._determine_skip_nodes(nodes_to_execute, modification_intent, conversation_context)

        # 5. 确定可以复用的节点
        nodes_to_reuse = self._determine_reuse_nodes(previous_execution, nodes_to_execute, impact_analysis)

        # 6. 生成执行计划
        execution_plan = self._generate_execution_plan(
            nodes_to_execute, nodes_to_skip, nodes_to_reuse, impact_analysis
        )

        self.logger.info(f"Planned incremental execution: {len(nodes_to_execute)} nodes to execute")
        return execution_plan

    def _analyze_modification_impact(self, modification_intent: Dict[str, Any]) -> Dict[str, Any]:
        """分析修改的影响程度"""
        impact_analysis = {
            'affected_categories': set(),
            'impact_level': {},
            'modification_fields': set()
        }

        modification_type = modification_intent.get('modification_type', 'adjust')
        targets = modification_intent.get('targets', [])
        modifications = modification_intent.get('modifications', {})

        # 根据修改类型确定影响
        if modification_type == 'complete':
            impact_analysis['impact_level'] = {node_id: ModificationImpact.CRITICAL
                                             for node_id in self.dependency_graph.nodes_definition.keys()}
            return impact_analysis

        # 分析具体修改字段
        for target in targets:
            if target == 'duration':
                impact_analysis['affected_categories'].add('structure')
                impact_analysis['modification_fields'].add('duration')
            elif target == 'style':
                impact_analysis['affected_categories'].add('style')
                impact_analysis['modification_fields'].add('visual_style')
            elif target == 'music':
                impact_analysis['affected_categories'].add('content')
                impact_analysis['modification_fields'].add('bgm')
            elif target == 'effects':
                impact_analysis['affected_categories'].add('style')
                impact_analysis['modification_fields'].add('dynamic_elements')

        # 为每个节点评估影响级别
        for node_id, node_def in self.dependency_graph.nodes_definition.items():
            impact_level = ModificationImpact.NONE

            # 检查节点敏感性
            for sensitive_field in node_def.sensitive_to:
                if any(field in sensitive_field for field in impact_analysis['modification_fields']):
                    if node_def.category in impact_analysis['affected_categories']:
                        impact_level = ModificationImpact.HIGH
                    else:
                        impact_level = ModificationImpact.MEDIUM
                    break

            impact_analysis['impact_level'][node_id] = impact_level

        return impact_analysis

    def _determine_start_nodes(self, impact_analysis: Dict[str, Any],
                             previous_execution: Dict[str, Any]) -> List[str]:
        """确定执行的起始节点"""
        start_nodes = []

        # 找到第一个需要高影响或关键影响的节点
        execution_order = self.dependency_graph.get_execution_order()

        for node_id in execution_order:
            impact_level = impact_analysis['impact_level'].get(node_id, ModificationImpact.NONE)

            if impact_level in [ModificationImpact.HIGH, ModificationImpact.CRITICAL]:
                start_nodes.append(node_id)
                break

        # 如果没有高影响节点，从中影响节点开始
        if not start_nodes:
            for node_id in execution_order:
                impact_level = impact_analysis['impact_level'].get(node_id, ModificationImpact.NONE)
                if impact_level == ModificationImpact.MEDIUM:
                    start_nodes.append(node_id)
                    break

        return start_nodes or [execution_order[0]]  # 默认从第一个节点开始

    def _calculate_execution_nodes(self, start_nodes: List[str],
                                 impact_analysis: Dict[str, Any]) -> List[str]:
        """计算需要执行的节点"""
        nodes_to_execute = set()

        # 添加起始节点
        nodes_to_execute.update(start_nodes)

        # 添加受影响的节点
        for start_node in start_nodes:
            affected_nodes = self.dependency_graph.get_affected_nodes(start_node)
            nodes_to_execute.update(affected_nodes)

        # 按执行顺序排序
        execution_order = self.dependency_graph.get_execution_order()
        return [node for node in execution_order if node in nodes_to_execute]

    def _determine_skip_nodes(self, nodes_to_execute: List[str],
                            modification_intent: Dict[str, Any],
                            conversation_context: Dict[str, Any]) -> List[str]:
        """确定可以跳过的节点"""
        skip_nodes = []

        for node_id in nodes_to_execute:
            if self.dependency_graph.can_skip_node(node_id, modification_intent):
                skip_nodes.append(node_id)

        return skip_nodes

    def _determine_reuse_nodes(self, previous_execution: Dict[str, Any],
                             nodes_to_execute: List[str],
                             impact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """确定可以复用的节点结果"""
        reuse_nodes = {}

        for node_id in nodes_to_execute:
            impact_level = impact_analysis['impact_level'].get(node_id, ModificationImpact.NONE)

            # 低影响或无影响的节点可以复用之前的结果
            if impact_level in [ModificationImpact.NONE, ModificationImpact.LOW]:
                previous_result = previous_execution.get(node_id)
                if previous_result:
                    reuse_nodes[node_id] = previous_result

        return reuse_nodes

    def _generate_execution_plan(self, nodes_to_execute: List[str],
                               nodes_to_skip: List[str],
                               nodes_to_reuse: Dict[str, Any],
                               impact_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """生成最终的执行计划"""
        # 过滤掉跳过和复用的节点
        actual_execution_nodes = [
            node for node in nodes_to_execute
            if node not in nodes_to_skip and node not in nodes_to_reuse
        ]

        execution_plan = {
            'strategy': 'incremental',
            'total_nodes': len(self.dependency_graph.nodes_definition),
            'nodes_to_execute': actual_execution_nodes,
            'nodes_to_skip': nodes_to_skip,
            'nodes_to_reuse': list(nodes_to_reuse.keys()),
            'reuse_data': nodes_to_reuse,
            'estimated_time': self.dependency_graph.estimate_execution_time(actual_execution_nodes),
            'impact_analysis': impact_analysis,
            'execution_order': actual_execution_nodes,
            'performance_gain': {
                'nodes_saved': len(nodes_to_skip) + len(nodes_to_reuse),
                'time_saved_estimate': self.dependency_graph.estimate_execution_time(
                    nodes_to_skip + list(nodes_to_reuse.keys())
                )
            }
        }

        return execution_plan


# 创建全局实例
node_dependency_manager = NodeDependencyGraph()
incremental_execution_planner = IncrementalExecutionPlanner(node_dependency_manager)