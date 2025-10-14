"""
工作流模板管理器 - 预定义的工作流模板
"""
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from .workflow_orchestrator import WorkflowOrchestrator, WorkflowConfig, NodeDependency
from nodes.base_node import BaseNode, NodeConfig, ProcessingPriority

# 导入各种节点类型
from nodes.audio_processor import AudioProcessingNode, AudioProcessingConfig
from nodes.subtitle_generator import SubtitleGeneratorNode, SubtitleConfig
from nodes.effects_processor import EffectsProcessorNode, EffectsConfig
from nodes.transitions_processor import TransitionsProcessorNode, TransitionsConfig
from nodes.render_compositor import RenderCompositorNode, NodeRenderConfig as RenderConfig


@dataclass
class TemplateConfig:
    """模板配置"""
    name: str
    description: str
    category: str
    tags: List[str]
    default_params: Dict[str, Any]


class WorkflowTemplateManager:
    """工作流模板管理器"""

    def __init__(self):
        self.templates: Dict[str, TemplateConfig] = {}
        self._register_default_templates()

    def _register_default_templates(self):
        """注册默认工作流模板"""

        # 基础视频生成模板
        self.templates['basic_video_generation'] = TemplateConfig(
            name="基础视频生成",
            description="标准的文本到视频生成流程",
            category="video_generation",
            tags=["basic", "text-to-video"],
            default_params={
                'quality': 'high',
                'resolution': '1080p',
                'duration_limit': 300
            }
        )

        # 高级视频生成模板
        self.templates['advanced_video_generation'] = TemplateConfig(
            name="高级视频生成",
            description="包含特效和转场的高级视频生成",
            category="video_generation",
            tags=["advanced", "effects", "transitions"],
            default_params={
                'quality': 'ultra',
                'resolution': '4K',
                'enable_effects': True,
                'enable_transitions': True
            }
        )

        # 音频优先模板
        self.templates['audio_focused_generation'] = TemplateConfig(
            name="音频优先生成",
            description="以音频处理为核心的视频生成",
            category="audio_video",
            tags=["audio", "podcast", "voice"],
            default_params={
                'audio_quality': 'studio',
                'noise_reduction': True,
                'audio_enhancement': True
            }
        )

        # 快速原型模板
        self.templates['quick_prototype'] = TemplateConfig(
            name="快速原型",
            description="快速生成低质量预览视频",
            category="prototype",
            tags=["quick", "preview", "draft"],
            default_params={
                'quality': 'low',
                'resolution': '720p',
                'skip_effects': True
            }
        )

    def create_basic_video_workflow(self, config: WorkflowConfig) -> WorkflowOrchestrator:
        """创建基础视频生成工作流"""
        orchestrator = WorkflowOrchestrator(config)

        # 1. 音频处理节点
        audio_config = AudioProcessingConfig(
            node_id="audio_processor",
            name="音频处理器",
            description="处理音频素材和语音合成",
            priority=ProcessingPriority.HIGH
        )
        audio_node = AudioProcessingNode(audio_config)
        orchestrator.add_node(audio_node, dependencies=[])

        # 2. 字幕生成节点
        subtitle_config = SubtitleConfig(
            node_id="subtitle_generator",
            name="字幕生成器",
            description="生成视频字幕",
            priority=ProcessingPriority.NORMAL
        )
        subtitle_node = SubtitleGeneratorNode(subtitle_config)
        orchestrator.add_node(subtitle_node, dependencies=["audio_processor"])

        # 3. 渲染合成节点
        render_config = RenderConfig(
            node_id="render_compositor",
            name="渲染合成器",
            description="最终视频渲染合成",
            priority=ProcessingPriority.CRITICAL
        )
        render_node = RenderCompositorNode(render_config)
        orchestrator.add_node(render_node, dependencies=["subtitle_generator"])

        return orchestrator

    def create_advanced_video_workflow(self, config: WorkflowConfig) -> WorkflowOrchestrator:
        """创建高级视频生成工作流"""
        orchestrator = WorkflowOrchestrator(config)

        # 1. 音频处理节点
        audio_config = AudioProcessingConfig(
            node_id="audio_processor",
            name="音频处理器",
            description="处理音频素材和语音合成",
            priority=ProcessingPriority.HIGH
        )
        audio_node = AudioProcessingNode(audio_config)
        orchestrator.add_node(audio_node, dependencies=[])

        # 2. 字幕生成节点
        subtitle_config = SubtitleConfig(
            node_id="subtitle_generator",
            name="字幕生成器",
            description="生成视频字幕",
            priority=ProcessingPriority.NORMAL
        )
        subtitle_node = SubtitleGeneratorNode(subtitle_config)
        orchestrator.add_node(subtitle_node, dependencies=["audio_processor"])

        # 3. 转场处理节点
        transitions_config = TransitionsConfig(
            node_id="transitions_processor",
            name="转场处理器",
            description="处理视频转场效果",
            priority=ProcessingPriority.NORMAL
        )
        transitions_node = TransitionsProcessorNode(transitions_config)
        orchestrator.add_node(transitions_node, dependencies=["subtitle_generator"])

        # 4. 特效处理节点
        effects_config = EffectsConfig(
            node_id="effects_processor",
            name="特效处理器",
            description="添加视频特效",
            priority=ProcessingPriority.NORMAL
        )
        effects_node = EffectsProcessorNode(effects_config)
        orchestrator.add_node(effects_node, dependencies=["transitions_processor"])

        # 5. 渲染合成节点
        render_config = RenderConfig(
            node_id="render_compositor",
            name="渲染合成器",
            description="最终视频渲染合成",
            priority=ProcessingPriority.CRITICAL
        )
        render_node = RenderCompositorNode(render_config)
        orchestrator.add_node(render_node, dependencies=["effects_processor"])

        return orchestrator

    def create_audio_focused_workflow(self, config: WorkflowConfig) -> WorkflowOrchestrator:
        """创建音频优先工作流"""
        orchestrator = WorkflowOrchestrator(config)

        # 1. 高级音频处理节点
        audio_config = AudioProcessingConfig(
            node_id="audio_processor",
            name="高级音频处理器",
            description="高质量音频处理和增强",
            priority=ProcessingPriority.CRITICAL,
            enable_noise_reduction=True,
            enable_audio_enhancement=True,
            audio_quality="studio"
        )
        audio_node = AudioProcessingNode(audio_config)
        orchestrator.add_node(audio_node, dependencies=[])

        # 2. 字幕生成节点（基于音频）
        subtitle_config = SubtitleConfig(
            node_id="subtitle_generator",
            name="音频同步字幕生成器",
            description="基于音频生成精确同步字幕",
            priority=ProcessingPriority.HIGH,
            sync_with_audio=True
        )
        subtitle_node = SubtitleGeneratorNode(subtitle_config)
        orchestrator.add_node(subtitle_node, dependencies=["audio_processor"])

        # 3. 简化渲染合成
        render_config = RenderConfig(
            node_id="render_compositor",
            name="音频优化渲染器",
            description="音频优先的视频渲染",
            priority=ProcessingPriority.HIGH,
            audio_priority=True
        )
        render_node = RenderCompositorNode(render_config)
        orchestrator.add_node(render_node, dependencies=["subtitle_generator"])

        return orchestrator

    def create_quick_prototype_workflow(self, config: WorkflowConfig) -> WorkflowOrchestrator:
        """创建快速原型工作流"""
        orchestrator = WorkflowOrchestrator(config)

        # 1. 快速音频处理
        audio_config = AudioProcessingConfig(
            node_id="audio_processor",
            name="快速音频处理器",
            description="快速低质量音频处理",
            priority=ProcessingPriority.NORMAL,
            quality_mode="fast"
        )
        audio_node = AudioProcessingNode(audio_config)
        orchestrator.add_node(audio_node, dependencies=[])

        # 2. 简单渲染（跳过字幕和特效）
        render_config = RenderConfig(
            node_id="render_compositor",
            name="快速渲染器",
            description="快速低质量渲染",
            priority=ProcessingPriority.NORMAL,
            quality_mode="fast",
            resolution="720p"
        )
        render_node = RenderCompositorNode(render_config)
        orchestrator.add_node(render_node, dependencies=["audio_processor"])

        return orchestrator

    def get_template_list(self) -> List[Dict[str, Any]]:
        """获取模板列表"""
        templates = []
        for template_id, template_config in self.templates.items():
            templates.append({
                'id': template_id,
                'name': template_config.name,
                'description': template_config.description,
                'category': template_config.category,
                'tags': template_config.tags,
                'default_params': template_config.default_params
            })
        return templates

    def get_template_by_id(self, template_id: str) -> Optional[TemplateConfig]:
        """根据ID获取模板"""
        return self.templates.get(template_id)

    def get_templates_by_category(self, category: str) -> List[Dict[str, Any]]:
        """根据分类获取模板"""
        templates = []
        for template_id, template_config in self.templates.items():
            if template_config.category == category:
                templates.append({
                    'id': template_id,
                    'name': template_config.name,
                    'description': template_config.description,
                    'tags': template_config.tags,
                    'default_params': template_config.default_params
                })
        return templates

    def search_templates(self, query: str) -> List[Dict[str, Any]]:
        """搜索模板"""
        query = query.lower()
        results = []

        for template_id, template_config in self.templates.items():
            # 搜索名称、描述和标签
            searchable_text = (
                template_config.name.lower() +
                template_config.description.lower() +
                ' '.join(template_config.tags)
            )

            if query in searchable_text:
                results.append({
                    'id': template_id,
                    'name': template_config.name,
                    'description': template_config.description,
                    'category': template_config.category,
                    'tags': template_config.tags,
                    'relevance_score': self._calculate_relevance(query, searchable_text)
                })

        # 按相关性排序
        results.sort(key=lambda x: x['relevance_score'], reverse=True)
        return results

    def _calculate_relevance(self, query: str, text: str) -> float:
        """计算搜索相关性得分"""
        query_words = set(query.split())
        text_words = set(text.split())

        # 计算交集比例
        intersection = query_words.intersection(text_words)
        return len(intersection) / len(query_words) if query_words else 0

    def register_custom_template(self, template_id: str, template_config: TemplateConfig,
                                builder_func: callable):
        """注册自定义模板"""
        self.templates[template_id] = template_config

        # 将构建函数动态添加到类中
        setattr(self, f'create_{template_id}_workflow', builder_func)

        print(f"✅ Custom template registered: {template_id}")

    def create_workflow_from_template(self, template_id: str, config: WorkflowConfig) -> WorkflowOrchestrator:
        """根据模板创建工作流"""
        if template_id not in self.templates:
            raise ValueError(f"Unknown template: {template_id}")

        # 获取对应的构建函数
        builder_method_name = f'create_{template_id}_workflow'

        if hasattr(self, builder_method_name):
            builder_func = getattr(self, builder_method_name)
            return builder_func(config)
        else:
            raise NotImplementedError(f"Builder function not implemented for template: {template_id}")

    def validate_template_params(self, template_id: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """验证模板参数"""
        if template_id not in self.templates:
            raise ValueError(f"Unknown template: {template_id}")

        template_config = self.templates[template_id]
        validated_params = template_config.default_params.copy()

        # 合并用户参数
        validated_params.update(params)

        # 这里可以添加参数验证逻辑
        # 例如：检查参数类型、范围等

        return validated_params

    def get_template_schema(self, template_id: str) -> Optional[Dict[str, Any]]:
        """获取模板参数架构"""
        if template_id not in self.templates:
            return None

        template_config = self.templates[template_id]

        # 根据模板类型返回参数架构
        schemas = {
            'basic_video_generation': {
                'quality': {'type': 'string', 'enum': ['low', 'medium', 'high'], 'default': 'high'},
                'resolution': {'type': 'string', 'enum': ['720p', '1080p', '4K'], 'default': '1080p'},
                'duration_limit': {'type': 'integer', 'min': 10, 'max': 600, 'default': 300}
            },
            'advanced_video_generation': {
                'quality': {'type': 'string', 'enum': ['high', 'ultra'], 'default': 'ultra'},
                'resolution': {'type': 'string', 'enum': ['1080p', '4K'], 'default': '4K'},
                'enable_effects': {'type': 'boolean', 'default': True},
                'enable_transitions': {'type': 'boolean', 'default': True}
            },
            'audio_focused_generation': {
                'audio_quality': {'type': 'string', 'enum': ['standard', 'high', 'studio'], 'default': 'studio'},
                'noise_reduction': {'type': 'boolean', 'default': True},
                'audio_enhancement': {'type': 'boolean', 'default': True}
            },
            'quick_prototype': {
                'quality': {'type': 'string', 'enum': ['low', 'medium'], 'default': 'low'},
                'resolution': {'type': 'string', 'enum': ['480p', '720p'], 'default': '720p'},
                'skip_effects': {'type': 'boolean', 'default': True}
            }
        }

        return schemas.get(template_id, {})

    def export_template(self, template_id: str) -> Dict[str, Any]:
        """导出模板配置"""
        if template_id not in self.templates:
            raise ValueError(f"Unknown template: {template_id}")

        template_config = self.templates[template_id]
        schema = self.get_template_schema(template_id)

        return {
            'template_id': template_id,
            'config': {
                'name': template_config.name,
                'description': template_config.description,
                'category': template_config.category,
                'tags': template_config.tags,
                'default_params': template_config.default_params
            },
            'schema': schema,
            'export_timestamp': datetime.now().isoformat()
        }

    def import_template(self, template_data: Dict[str, Any]) -> bool:
        """导入模板配置"""
        try:
            template_id = template_data['template_id']
            config_data = template_data['config']

            template_config = TemplateConfig(
                name=config_data['name'],
                description=config_data['description'],
                category=config_data['category'],
                tags=config_data['tags'],
                default_params=config_data['default_params']
            )

            self.templates[template_id] = template_config
            print(f"✅ Template imported: {template_id}")
            return True

        except Exception as e:
            print(f"❌ Failed to import template: {e}")
            return False