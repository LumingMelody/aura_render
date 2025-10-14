#!/usr/bin/env python3
"""
VGP Protocol Tools - 标准化VGP协议读写验证工具
提供VGPReader、VGPWriter、VGPValidator功能
"""

import json
import logging
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from enum import Enum
import uuid

class VGPJSONEncoder(json.JSONEncoder):
    """自定义JSON编码器，处理不可序列化的对象和循环引用"""

    # 类级别的访问集合,在整个序列化过程中共享
    _global_visited = set()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # 在开始新的序列化时清空访问记录
        if not hasattr(self, '_encoding_started'):
            VGPJSONEncoder._global_visited.clear()
            self._encoding_started = True

    def default(self, obj):
        # 跳过基本类型和None
        if obj is None or isinstance(obj, (str, int, float, bool)):
            return obj

        # 防止循环引用
        obj_id = id(obj)
        if obj_id in VGPJSONEncoder._global_visited:
            return f"<circular:{type(obj).__name__}@{hex(obj_id)[-6:]}>"

        VGPJSONEncoder._global_visited.add(obj_id)

        try:
            # 处理datetime对象
            if isinstance(obj, datetime):
                return obj.isoformat()

            # 处理Enum对象
            if isinstance(obj, Enum):
                return obj.value

            # 处理dataclass对象(在检查__dict__之前)
            if hasattr(obj, '__dataclass_fields__'):
                try:
                    return asdict(obj)
                except Exception:
                    # 如果asdict失败,转为字典手动处理
                    return {k: getattr(obj, k) for k in obj.__dataclass_fields__}

            # 处理有__dict__的自定义对象
            if hasattr(obj, '__dict__'):
                return self._safe_dict(obj.__dict__)

            # 处理列表和元组
            if isinstance(obj, (list, tuple)):
                return [self.default(item) for item in obj]

            # 处理字典
            if isinstance(obj, dict):
                return self._safe_dict(obj)

            # 其他情况转为字符串
            return f"<{type(obj).__name__}>"

        except RecursionError:
            return f"<recursion:{type(obj).__name__}>"
        except Exception as e:
            return f"<error:{type(obj).__name__}:{str(e)[:50]}>"

    def _safe_dict(self, d):
        """安全地序列化字典,跳过不可序列化的值"""
        safe = {}
        for key, value in d.items():
            # 跳过私有属性和方法
            if key.startswith('_'):
                continue
            try:
                # 递归处理嵌套对象
                safe[key] = self.default(value)
            except Exception:
                safe[key] = f"<{type(value).__name__}>"
        return safe

    def encode(self, o):
        """重写encode以确保清理访问集合"""
        try:
            return super().encode(o)
        finally:
            # 编码完成后清空访问集合
            VGPJSONEncoder._global_visited.clear()

logger = logging.getLogger(__name__)

# =============================
# VGP 数据模型定义
# =============================

class VGPVersion(Enum):
    """VGP协议版本"""
    V1_0 = "1.0"
    V2_0 = "2.0"
    CURRENT = V2_0

class NodeStatus(Enum):
    """节点执行状态"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"

@dataclass
class VGPNodeData:
    """VGP节点数据标准格式"""
    node_id: str
    node_type: str
    status: NodeStatus = NodeStatus.PENDING
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())

@dataclass
class VGPDocument:
    """VGP文档标准格式"""
    version: str = VGPVersion.CURRENT.value
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)
    nodes: List[VGPNodeData] = field(default_factory=list)
    pipeline_config: Dict[str, Any] = field(default_factory=dict)
    final_output: Dict[str, Any] = field(default_factory=dict)

# =============================
# VGP Reader - 解析VGP格式输入
# =============================

class VGPReader:
    """VGP格式读取器"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VGPReader")

    def read(self, data: Union[str, Dict, bytes]) -> VGPDocument:
        """
        读取并解析VGP格式数据

        Args:
            data: VGP数据（JSON字符串、字典或字节）

        Returns:
            VGPDocument: 解析后的VGP文档对象
        """
        try:
            # 处理不同输入类型
            if isinstance(data, bytes):
                data = data.decode('utf-8')

            if isinstance(data, str):
                data = json.loads(data)

            # 验证基本结构
            if not isinstance(data, dict):
                raise ValueError("VGP data must be a dictionary")

            # 构建VGP文档
            doc = VGPDocument(
                version=data.get('version', VGPVersion.CURRENT.value),
                task_id=data.get('task_id', str(uuid.uuid4())),
                created_at=data.get('created_at', datetime.now().isoformat()),
                updated_at=data.get('updated_at', datetime.now().isoformat()),
                metadata=data.get('metadata', {}),
                pipeline_config=data.get('pipeline_config', {}),
                final_output=data.get('final_output', {})
            )

            # 解析节点数据
            nodes_data = data.get('nodes', [])
            for node_data in nodes_data:
                node = VGPNodeData(
                    node_id=node_data.get('node_id', ''),
                    node_type=node_data.get('node_type', ''),
                    status=NodeStatus(node_data.get('status', 'pending')),
                    input_data=node_data.get('input_data', {}),
                    output_data=node_data.get('output_data', {}),
                    error_message=node_data.get('error_message'),
                    execution_time=node_data.get('execution_time'),
                    timestamp=node_data.get('timestamp', datetime.now().isoformat())
                )
                doc.nodes.append(node)

            self.logger.info(f"Successfully read VGP document: {doc.task_id}")
            return doc

        except Exception as e:
            self.logger.error(f"Failed to read VGP data: {e}")
            raise ValueError(f"Invalid VGP format: {e}")

    def read_file(self, filepath: str) -> VGPDocument:
        """从文件读取VGP数据"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = f.read()
            return self.read(data)
        except Exception as e:
            self.logger.error(f"Failed to read VGP file {filepath}: {e}")
            raise

# =============================
# VGP Writer - 生成标准VGP格式输出
# =============================

class VGPWriter:
    """VGP格式写入器"""

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VGPWriter")

    def write(self, document: VGPDocument, pretty: bool = True) -> str:
        """
        将VGP文档写入为JSON字符串

        Args:
            document: VGP文档对象
            pretty: 是否格式化输出

        Returns:
            str: JSON格式的VGP数据
        """
        try:
            # 更新时间戳
            document.updated_at = datetime.now().isoformat()

            # 转换为字典
            data = {
                'version': document.version,
                'task_id': document.task_id,
                'created_at': document.created_at,
                'updated_at': document.updated_at,
                'metadata': document.metadata,
                'pipeline_config': document.pipeline_config,
                'nodes': [],
                'final_output': document.final_output
            }

            # 转换节点数据
            for node in document.nodes:
                node_dict = {
                    'node_id': node.node_id,
                    'node_type': node.node_type,
                    'status': node.status.value,
                    'input_data': node.input_data,
                    'output_data': node.output_data,
                    'error_message': node.error_message,
                    'execution_time': node.execution_time,
                    'timestamp': node.timestamp
                }
                data['nodes'].append(node_dict)

            # 生成JSON
            if pretty:
                json_str = json.dumps(data, indent=2, ensure_ascii=False, cls=VGPJSONEncoder)
            else:
                json_str = json.dumps(data, ensure_ascii=False, cls=VGPJSONEncoder)

            self.logger.info(f"Successfully wrote VGP document: {document.task_id}")
            return json_str

        except Exception as e:
            self.logger.error(f"Failed to write VGP document: {e}")
            raise

    def write_file(self, document: VGPDocument, filepath: str, pretty: bool = True):
        """将VGP文档写入文件"""
        try:
            json_str = self.write(document, pretty)
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(json_str)
            self.logger.info(f"VGP document saved to: {filepath}")
        except Exception as e:
            self.logger.error(f"Failed to write VGP file {filepath}: {e}")
            raise

# =============================
# VGP Validator - 验证VGP数据格式正确性
# =============================

class VGPValidator:
    """VGP格式验证器"""

    # 必需的节点类型（16个标准VGP节点）
    REQUIRED_NODE_TYPES = [
        'video_type_identification',
        'emotion_analysis',
        'shot_block_generation',
        'bgm_anchor_planning',
        'bgm_composition',
        'asset_request',
        'audio_processing',
        'sfx_integration',
        'transition_selection',
        'filter_application',
        'dynamic_effects',
        'aux_media_insertion',
        'aux_text_insertion',
        'subtitle_generation',
        'intro_outro',
        'timeline_integration'
    ]

    # 节点间依赖关系
    NODE_DEPENDENCIES = {
        'audio_processing': ['bgm_composition', 'sfx_integration'],
        'timeline_integration': ['asset_request', 'transition_selection', 'filter_application'],
        'intro_outro': ['video_type_identification', 'emotion_analysis']
    }

    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.VGPValidator")
        self.errors = []
        self.warnings = []

    def validate(self, document: VGPDocument) -> bool:
        """
        验证VGP文档格式和内容

        Args:
            document: VGP文档对象

        Returns:
            bool: 验证是否通过
        """
        self.errors = []
        self.warnings = []

        # 验证基本结构
        self._validate_structure(document)

        # 验证节点完整性
        self._validate_nodes(document)

        # 验证节点依赖关系
        self._validate_dependencies(document)

        # 验证数据格式
        self._validate_data_format(document)

        # 输出验证结果
        if self.errors:
            for error in self.errors:
                self.logger.error(f"Validation Error: {error}")

        if self.warnings:
            for warning in self.warnings:
                self.logger.warning(f"Validation Warning: {warning}")

        return len(self.errors) == 0

    def _validate_structure(self, document: VGPDocument):
        """验证文档基本结构"""
        if not document.task_id:
            self.errors.append("Missing task_id")

        if not document.version:
            self.errors.append("Missing version")

        try:
            VGPVersion(document.version)
        except ValueError:
            self.warnings.append(f"Unknown VGP version: {document.version}")

    def _validate_nodes(self, document: VGPDocument):
        """验证节点完整性"""
        existing_types = {node.node_type for node in document.nodes}

        # 检查必需节点
        missing_nodes = set(self.REQUIRED_NODE_TYPES) - existing_types
        if missing_nodes:
            self.warnings.append(f"Missing required nodes: {missing_nodes}")

        # 检查节点数据
        for node in document.nodes:
            if not node.node_id:
                self.errors.append(f"Node missing node_id: {node.node_type}")

            if not node.node_type:
                self.errors.append(f"Node missing node_type: {node.node_id}")

            # 检查失败节点
            if node.status == NodeStatus.FAILED and not node.error_message:
                self.warnings.append(f"Failed node without error message: {node.node_id}")

    def _validate_dependencies(self, document: VGPDocument):
        """验证节点依赖关系"""
        node_outputs = {}

        # 收集节点输出
        for node in document.nodes:
            if node.status == NodeStatus.COMPLETED:
                node_outputs[node.node_type] = node.output_data

        # 检查依赖
        for node in document.nodes:
            if node.node_type in self.NODE_DEPENDENCIES:
                deps = self.NODE_DEPENDENCIES[node.node_type]
                for dep in deps:
                    if dep not in node_outputs:
                        self.warnings.append(
                            f"Node {node.node_type} missing dependency: {dep}"
                        )
                    else:
                        # 检查特定的输出字段
                        self._check_specific_dependencies(node, node_outputs[dep], dep)

    def _check_specific_dependencies(self, node: VGPNodeData, dep_output: Dict, dep_type: str):
        """检查特定的依赖字段"""
        # 音频处理节点的特定依赖
        if node.node_type == 'audio_processing':
            if dep_type == 'bgm_composition' and 'bgm_composition_id' not in dep_output:
                self.errors.append(
                    f"bgm_composition missing required output: bgm_composition_id"
                )
            if dep_type == 'sfx_integration' and 'sfx_track_id' not in dep_output:
                self.warnings.append(
                    f"sfx_integration missing expected output: sfx_track_id"
                )

    def _validate_data_format(self, document: VGPDocument):
        """验证数据格式"""
        for node in document.nodes:
            # 检查输入输出数据类型
            if not isinstance(node.input_data, dict):
                self.errors.append(f"Node {node.node_id} input_data must be dict")

            if not isinstance(node.output_data, dict):
                self.errors.append(f"Node {node.node_id} output_data must be dict")

    def get_validation_report(self) -> Dict[str, Any]:
        """获取验证报告"""
        return {
            'valid': len(self.errors) == 0,
            'errors': self.errors,
            'warnings': self.warnings,
            'error_count': len(self.errors),
            'warning_count': len(self.warnings)
        }

# =============================
# VGP Protocol Manager - 统一管理接口
# =============================

class VGPProtocol:
    """VGP协议管理器 - 提供统一的读写验证接口"""

    def __init__(self):
        self.reader = VGPReader()
        self.writer = VGPWriter()
        self.validator = VGPValidator()
        self.logger = logging.getLogger(f"{__name__}.VGPProtocol")

    def create_document(self, metadata: Optional[Dict] = None) -> VGPDocument:
        """创建新的VGP文档"""
        doc = VGPDocument()
        if metadata:
            doc.metadata = metadata
        return doc

    def add_node(self, document: VGPDocument,
                 node_type: str,
                 input_data: Dict[str, Any],
                 output_data: Optional[Dict[str, Any]] = None) -> VGPNodeData:
        """向文档添加节点"""
        node = VGPNodeData(
            node_id=f"{node_type}_{uuid.uuid4().hex[:8]}",
            node_type=node_type,
            status=NodeStatus.COMPLETED if output_data else NodeStatus.PENDING,
            input_data=input_data,
            output_data=output_data or {}
        )
        document.nodes.append(node)
        return node

    def update_node_output(self, document: VGPDocument,
                          node_id: str,
                          output_data: Dict[str, Any],
                          status: NodeStatus = NodeStatus.COMPLETED):
        """更新节点输出数据"""
        for node in document.nodes:
            if node.node_id == node_id:
                node.output_data = output_data
                node.status = status
                node.timestamp = datetime.now().isoformat()
                return node
        raise ValueError(f"Node not found: {node_id}")

    def validate_and_fix(self, document: VGPDocument) -> VGPDocument:
        """验证并尝试修复VGP文档"""
        # 先验证
        is_valid = self.validator.validate(document)

        if not is_valid:
            self.logger.warning("Document validation failed, attempting to fix...")

            # 尝试修复常见问题
            # 1. 添加缺失的task_id
            if not document.task_id:
                document.task_id = str(uuid.uuid4())

            # 2. 修复节点ID
            for node in document.nodes:
                if not node.node_id:
                    node.node_id = f"{node.node_type}_{uuid.uuid4().hex[:8]}"

            # 3. 确保数据格式正确
            for node in document.nodes:
                if not isinstance(node.input_data, dict):
                    node.input_data = {}
                if not isinstance(node.output_data, dict):
                    node.output_data = {}

        return document

    def save(self, document: VGPDocument, filepath: str):
        """保存VGP文档到文件"""
        # 验证并修复
        document = self.validate_and_fix(document)

        # 写入文件
        self.writer.write_file(document, filepath)

        self.logger.info(f"VGP document saved: {filepath}")

    def load(self, filepath: str) -> VGPDocument:
        """从文件加载VGP文档"""
        document = self.reader.read_file(filepath)

        # 验证加载的文档
        if not self.validator.validate(document):
            self.logger.warning("Loaded document has validation issues")

        return document

# =============================
# 辅助函数
# =============================

def create_vgp_protocol() -> VGPProtocol:
    """创建VGP协议管理器实例"""
    return VGPProtocol()

def test_vgp_protocol():
    """测试VGP协议工具"""
    # 创建协议管理器
    protocol = create_vgp_protocol()

    # 创建文档
    doc = protocol.create_document({
        'title': 'Test Video Generation',
        'description': 'Testing VGP Protocol Tools'
    })

    # 添加节点
    protocol.add_node(doc, 'video_type_identification',
                     {'theme': '科技创新'},
                     {'video_type': '商业类'})

    protocol.add_node(doc, 'emotion_analysis',
                     {'video_type': '商业类'},
                     {'emotions': {'primary': '激励', 'secondary': '期待'}})

    # 验证
    validator = VGPValidator()
    is_valid = validator.validate(doc)
    report = validator.get_validation_report()

    print(f"Validation: {is_valid}")
    print(f"Report: {json.dumps(report, indent=2, ensure_ascii=False)}")

    # 写入
    writer = VGPWriter()
    json_str = writer.write(doc)
    print(f"VGP JSON:\n{json_str}")

    return doc

if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(level=logging.INFO)

    # 测试VGP协议工具
    test_vgp_protocol()