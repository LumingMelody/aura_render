# core/base_node.py
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List, Type, get_origin, get_args,Union
import json
import copy
from datetime import datetime
import json5


class BaseNode(ABC):
    # 声明所需输入参数（用于生成前端表单）
    required_inputs: List[Dict[str, Any]] = []

    output_schema: List[Dict[str, Any]] = []

    # 文件上传配置（可选重写）
    file_upload_config: Dict[str, Any] = {
        "image": {
            "enabled": False,
            "number_limits": 3,
            "detail": "high",
            "transfer_methods": ["remote_url", "local_file"]
        }
    }

    # 系统级参数限制（可选）
    system_parameters: Dict[str, Any] = {
        "file_size_limit": 15,
        "image_file_size_limit": 10,
        "audio_file_size_limit": 50,
        "video_file_size_limit": 100
    }

    def __init__(self, node_id: str, node_type: str, name: str):
        self.node_id = node_id
        self.type = node_type
        self.name = name

        self.generated: Dict[str, Any] = {}
        self.modified: Dict[str, Any] = {}
        self.source_map: Dict[str, dict] = {}

        self.bound_segment: Optional[str] = None
        self.status = "active"

    @abstractmethod
    async def generate(self, context: Dict[str, Any]) -> Dict[str, Any]:
        pass

    @abstractmethod
    async def regenerate(self, context: Dict[str, Any], user_intent: Dict[str, Any]) -> Dict[str, Any]:
        pass

    # def _validate_context(self, context: Dict[str, Any]):
    #     if not self.required_inputs:
    #         return

    #     missing = []
    #     errors = []

    #     for param in self.required_inputs:
    #         name = param["name"]
    #         required = param.get("required", True)
    #         expected_type = param.get("type", Any)
    #         default = param.get("default", None)
    #         field_type = param.get("field_type", "text")  # 如 text, number, textarea 等

    #         if name not in context:
    #             if required and default is None:
    #                 missing.append(name)
    #             elif default is not None:
    #                 context[name] = default
    #         else:
    #             value = context[name]
    #             if not self._is_instance(value, expected_type):
    #                 errors.append(f"参数 '{name}' 类型应为 {expected_type.__name__}, 实际为 {type(value).__name__}")

    #     if missing:
    #         raise ValueError(f"[Node: {self.name}] 缺少必需参数: {', '.join(missing)}")
    #     if errors:
    #         raise TypeError(f"[Node: {self.name}] 参数类型错误:\n" + "\n".join(errors))



    def validate_context(self, context: Dict[str, Any]):
        if not self.required_inputs:
            return

        # DEBUG: Print context keys and required inputs
        print(f"🔍 [DEBUG] Node {self.name} validation:")
        print(f"🔍 [DEBUG] Context keys: {list(context.keys())}")
        print(f"🔍 [DEBUG] Required inputs: {[p['name'] for p in self.required_inputs]}")

        missing = []
        errors = []

        for param in self.required_inputs:
            name = param["name"]
            required = param.get("required", True)
            expected_type = param.get("type", Any)
            default = param.get("default", None)
            field_type = param.get("field_type", "text")

            print(f"🔍 [DEBUG] Checking parameter '{name}': exists={name in context}, value={context.get(name, 'NOT_FOUND')}")

            # 1. 检查字段是否存在
            if name not in context:
                if required and default is None:
                    missing.append(name)
                elif default is not None:
                    context[name] = default
                continue

            value = context[name]

            # 2. 如果已经是期望类型，跳过
            if self._is_instance(value, expected_type):
                continue

            # 3. 尝试转换
            converted_value, success = self._try_convert(value, expected_type, field_type)
            if success:
                context[name] = converted_value  # 更新为转换后的值
            else:
                if required:
                    errors.append(f"参数 '{name}' 类型应为 {expected_type.__name__}, 实际为 {type(value).__name__}, 且无法从 '{value}' 转换")
                elif value is not None:  # 可选字段但提供了错误类型
                    errors.append(f"可选参数 '{name}' 类型错误，期望 {expected_type.__name__}，实际 {type(value).__name__}")

        # 统一抛出异常
        if missing:
            raise ValueError(f"[Node: {self.name}] 缺少必需参数: {', '.join(missing)}")
        if errors:
            raise TypeError(f"[Node: {self.name}] 参数类型错误:\n" + "\n".join(errors))
        
        return context


    # def _try_convert(self, value: Any, expected_type: Type, field_type: str) -> tuple:
    #     """
    #     尝试将 value 转换为 expected_type。
    #     返回 (converted_value, success: bool)
    #     """
    #     # 如果已经是正确类型
    #     if self._is_instance(value, expected_type):
    #         return value, True

    #     # 获取实际类型和泛型信息
    #     origin = get_origin(expected_type)
    #     args = get_args(expected_type)

    #     # 处理 List[T] 或 list
    #     if origin is list or expected_type is list:
    #         target_elem_type = args[0] if origin is list and args else None

    #         # 字符串解析为 list
    #         if isinstance(value, str):
    #             value = value.strip()
    #             if value == "":
    #                 return [], True
    #             try:
    #                 parsed = json.loads(value)
    #                 if not isinstance(parsed, list):
    #                     return None, False
    #                 value = parsed
    #             except (json.JSONDecodeError, TypeError):
    #                 return None, False

    #         if not isinstance(value, list):
    #             return None, False

    #         # 如果没有元素类型要求，直接返回
    #         if target_elem_type is None:
    #             return value, True

    #         # 否则尝试转换每个元素
    #         converted_list = []
    #         for item in value:
    #             # 假设 field_type 对所有元素一致，或传 None
    #             converted_item, success = self._try_convert(item, target_elem_type, "")
    #             if not success:
    #                 return None, False
    #             converted_list.append(converted_item)

    #         return converted_list, True

    #     # 处理 Dict[K, V] 或 dict
    #     if origin is dict or expected_type is dict:
    #         target_key_type, target_value_type = None, None
    #         if origin is dict and args:
    #             if len(args) == 2:
    #                 target_key_type, target_value_type = args
    #             else:
    #                 target_value_type = args[0]  # 单参数时假设是 value 类型

    #         # 字符串解析为 dict
    #         if isinstance(value, str):
    #             value = value.strip()
    #             if value == "":
    #                 return {}, True
    #             try:
    #                 parsed = json.loads(value)
    #                 if not isinstance(parsed, dict):
    #                     return None, False
    #                 value = parsed
    #             except (json.JSONDecodeError, TypeError):
    #                 return None, False

    #         if not isinstance(value, dict):
    #             return None, False

    #         # 如果没有类型要求，直接返回
    #         if target_key_type is None and target_value_type is None:
    #             return value, True

    #         # 转换 key 和 value（可选）
    #         converted_dict = {}
    #         for k, v in value.items():
    #             converted_k, converted_v = k, v
    #             success_k, success_v = True, True

    #             if target_key_type is not None:
    #                 converted_k, success_k = self._try_convert(k, target_key_type, "")
    #             if target_value_type is not None:
    #                 converted_v, success_v = self._try_convert(v, target_value_type, "")

    #             if not success_k or not success_v:
    #                 return None, False

    #             converted_dict[converted_k] = converted_v

    #         return converted_dict, True

    #     # str -> 其他基础类型（保持原有逻辑）
    #     if isinstance(value, str):
    #         value = value.strip()
    #         if value == "":
    #             if expected_type is bool:
    #                 return False, True
    #             elif expected_type in (int, float):
    #                 return None, False
    #             else:
    #                 return "", True  # 或根据需要返回 None

    #         if expected_type is bool:
    #             if value.lower() in ('true', '1', 'yes', 'on'):
    #                 return True, True
    #             elif value.lower() in ('false', '0', 'no', 'off'):
    #                 return False, True
    #             else:
    #                 return None, False

    #         if expected_type is int:
    #             try:
    #                 return int(float(value)), True
    #             except (ValueError, TypeError):
    #                 return None, False

    #         if expected_type is float:
    #             try:
    #                 return float(value), True
    #             except (ValueError, TypeError):
    #                 return None, False

    #     # 数值类型转换
    #     if expected_type is int and isinstance(value, (int, float)):
    #         if value == int(value):
    #             return int(value), True
    #         else:
    #             return None, False

    #     if expected_type is float and isinstance(value, (int, float)):
    #         return float(value), True

    #     if expected_type is str:
    #         return str(value), True

    #     # field_type 推断（保留）
    #     if field_type == "number":
    #         if expected_type is int:
    #             try:
    #                 f = float(value)
    #                 if f == int(f):
    #                     return int(f), True
    #                 return None, False
    #             except (ValueError, TypeError):
    #                 return None, False
    #         elif expected_type is float:
    #             try:
    #                 return float(value), True
    #             except (ValueError, TypeError):
    #                 return None, False

    #     # 默认失败
    #     return None, False




    def _try_convert(self, value: Any, expected_type: Type, field_type: str) -> tuple:
        """
        使用 json5 支持更宽松的 JSON 格式
        """
        origin = get_origin(expected_type)
        args = get_args(expected_type)

        # --- 处理 Optional[T] ---
        if origin is Union and len(args) == 2 and type(None) in args:
            other_type = args[0] if args[1] is type(None) else args[1]
            if value is None or (isinstance(value, str) and value.strip().lower() in ("null", "none", "")):
                return None, True
            return self._try_convert(value, other_type, field_type)

        # --- 已是正确类型 ---
        if self._is_instance(value, expected_type):
            return value, True

        # --- 特殊：Any 类型，仍尝试结构化解析 ---
        if expected_type is Any:
            return self._deep_parse(value)

        # === 处理 List ===
        if origin is list or expected_type is list:
            elem_type = args[0] if origin is list and args else Any

            # 字符串 → list（使用 json5）
            if isinstance(value, str):
                value = value.strip()
                if not value or value.lower() in ("null", "none"):
                    return [], True

                # ✅ 使用 json5.loads，并加 try-except 安全兜底
                parsed = None  # ✅ 修复：初始化parsed变量
                try:
                    parsed = json5.loads(value)  # ← 使用 json5
                except Exception as e:
                    # 情况3: [a, b, c] 无引号格式（仅限中文/简单标识符）
                    if  value.startswith("[") and value.endswith("]"):
                        try:
                            parsed = self._parse_unquoted_list(value)
                        except:
                            print(f"[JSON5 Error] 无法解析 list 字符串: {e}, value={value[:100]}...")
                            return None, False
                    else:
                        # ✅ 修复：如果不是list格式，直接返回错误
                        print(f"[JSON5 Error] 无法解析 list 字符串: {e}, value={value[:100]}...")
                        return None, False

                if not isinstance(parsed, list):
                    return None, False
                value = parsed

            if not isinstance(value, list):
                return None, False

            result = []
            for item in value:
                converted, success = self._try_convert(item, elem_type, "")
                if not success:
                    return None, False
                result.append(converted)
            return result, True

        # === 处理 Dict ===
        if origin is dict or expected_type is dict:
            key_type = args[0] if origin is dict and len(args) >= 1 else str
            value_type = args[1] if origin is dict and len(args) >= 2 else Any

            # 字符串 → dict（使用 json5）
            if isinstance(value, str):
                value = value.strip()
                if not value or value.lower() in ("null", "none"):
                    return {}, True

                try:
                    parsed = json5.loads(value)  # ← 使用 json5
                except Exception as e:
                    print(f"[JSON5 Error] 无法解析 dict 字符串: {e}, value={value[:100]}...")
                    return None, False

                if not isinstance(parsed, dict):
                    return None, False
                value = parsed

            if not isinstance(value, dict):
                return None, False

            result = {}
            for k, v in value.items():
                ck, success_k = self._try_convert(k, key_type, "")
                if not success_k:
                    return None, False
                cv, success_v = self._try_convert(v, value_type, "")
                if not success_v:
                    return None, False
                result[ck] = cv
            return result, True

        # === 基础类型转换 ===
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() == "null":
                if expected_type is bool:
                    return False, True
                elif expected_type in (int, float):
                    return None, False
                elif expected_type in (list, dict):
                    return [] if expected_type is list else {}, True
                else:
                    return None, True

            # bool
            if expected_type is bool:
                return value.lower() in ('true', '1', 'yes', 'on'), True

            # int/float
            if expected_type is int:
                try:
                    return int(float(value)), True
                except:
                    return None, False
            if expected_type is float:
                try:
                    return float(value), True
                except:
                    return None, False

            # str → str
            if expected_type is str:
                return value, True

        # 数值转换
        if expected_type is int and isinstance(value, (int, float)) and value == int(value):
            return int(value), True
        if expected_type is float and isinstance(value, (int, float)):
            return float(value), True
        if expected_type is str:
            return str(value), True

        # field_type 推断
        if field_type == "number" and expected_type in (int, float):
            try:
                num = float(value)
                if expected_type is int:
                    return int(num) if num == int(num) else None, num == int(num)
                return num, True
            except:
                return None, False

        # 默认失败
        return None, False

    def _deep_parse(self, value: Any) -> tuple:
        """
        在 expected_type is Any 时，仍尝试结构化解析：
        - 字符串尝试 JSON 解析
        - 解析后递归处理 list/dict 内部
        """
        if isinstance(value, str):
            value = value.strip()
            if not value or value.lower() == "null":
                return None, True
            try:
                parsed = json.loads(value)
                return self._deep_parse_value(parsed)
            except json.JSONDecodeError:
                return value, True  # 无法解析就当普通字符串

        return self._deep_parse_value(value)


    def _parse_unquoted_list(self,s: str) -> list:
        if not s.startswith("[") or not s.endswith("]"):
            return None
        # 去掉头尾 []
        content = s[1:-1].strip()
        if not content:
            return []
        # 按逗号分割，去除空白
        items = [item.strip() for item in content.split(",")]
        # 过滤空字符串
        return [item for item in items if item]
    
    def _deep_parse_value(self, value: Any) -> tuple:
        """递归解析任意结构"""
        if isinstance(value, list):
            result = []
            for item in value:
                parsed_item, _ = self._deep_parse(item)
                result.append(parsed_item)
            return result, True

        elif isinstance(value, dict):
            result = {}
            for k, v in value.items():
                parsed_k, _ = self._deep_parse(k)
                parsed_v, _ = self._deep_parse(v)
                result[parsed_k] = parsed_v
            return result, True

        else:
            return value, True
    def _is_instance(self, value: Any, expected_type: type) -> bool:
        if expected_type is Any:
            return True
        try:
            origin = get_origin(expected_type)
            if origin is not None:
                if origin is list:
                    arg = get_args(expected_type)[0]
                    return isinstance(value, list) and all(isinstance(i, arg) for i in value)
                elif origin is dict:
                    k_arg, v_arg = get_args(expected_type)
                    return (isinstance(value, dict) and
                            all(isinstance(k, k_arg) and isinstance(v, v_arg) for k, v in value.items()))
                else:
                    return isinstance(value, origin)
            else:
                return isinstance(value, expected_type)
        except Exception:
            return isinstance(value, expected_type)

    def apply_generation(self, config: Dict[str, Any], source: str = "ai", comment: str = ""):
        timestamp = datetime.now().isoformat()
        for key, value in config.items():
            self.generated[key] = value
            self.source_map[key] = {
                "source": source,
                "timestamp": timestamp,
                "comment": comment
            }
            if key not in self.modified:
                setattr(self, key, value)

    def apply_modification(self, config: Dict[str, Any], comment: str = ""):
        timestamp = datetime.now().isoformat()
        for key, value in config.items():
            old_value = self.get_value(key)
            if old_value != value:
                self.modified[key] = value
                self.source_map[key] = {
                    "source": "user",
                    "timestamp": timestamp,
                    "comment": comment
                }
                setattr(self, key, value)

    def get_value(self, key: str) -> Any:
        return self.modified.get(key, self.generated.get(key))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "name": self.name,
            "generated": copy.deepcopy(self.generated),
            "modified": copy.deepcopy(self.modified),
            "source_map": copy.deepcopy(self.source_map),
            "status": self.status,
            "bound_segment": self.bound_segment
        }

    # @classmethod
    # def get_input_schema(cls) -> Dict[str, Any]:
    #     """
    #     返回符合你指定格式的输入 schema，可用于前端动态生成表单
    #     """
    #     user_input_form = []
    #     for param in cls.required_inputs:
    #         field_type = param.get("field_type", "text")
    #         default_value = param.get("default", "")

    #         # 根据类型建议字段类型
    #         if param.get("type") == str:
    #             field_type = param.get("field_type") or ("textarea" if len(str(default_value)) > 100 else "text")
    #         elif param.get("type") == int or param.get("type") == float:
    #             field_type = param.get("field_type") or "number"

    #         user_input_form.append({
    #             "paragraph": {
    #                 "label": param.get("label", param["name"].replace("_", " ").title()),
    #                 "variable": param["name"],
    #                 "required": param.get("required", True),
    #                 "default": default_value,
    #                 "type": field_type,
    #                 "desc": param.get("desc", "")
    #             }
    #         })

    #     return {
    #         "user_input_form": user_input_form,
    #         "file_upload": copy.deepcopy(cls.file_upload_config),
    #         "system_parameters": copy.deepcopy(cls.system_parameters)
    #     }
    
    @classmethod
    def get_input_schema(cls) -> Dict[str, Any]:
        """
        返回符合你指定格式的输入 schema，可用于前端动态生成表单
        """
        return cls.required_inputs
        user_input_form = []
        for param in cls.required_inputs:
            field_type = param.get("field_type", "text")
            default_value = param.get("default", "")

            # 根据类型建议字段类型
            if param.get("type") == str:
                field_type = param.get("field_type") or ("textarea" if len(str(default_value)) > 100 else "text")
            elif param.get("type") == int or param.get("type") == float:
                field_type = param.get("field_type") or "number"

            user_input_form.append({
                "paragraph": {
                    "label": param.get("label", param["name"].replace("_", " ").title()),
                    "variable": param["name"],
                    "required": param.get("required", True),
                    "default": default_value,
                    "type": field_type,
                    "desc": param.get("desc", "")
                }
            })

        return {
            "user_input_form": user_input_form,
            "file_upload": copy.deepcopy(cls.file_upload_config),
            "system_parameters": copy.deepcopy(cls.system_parameters)
        }
    
    @classmethod
    def get_output_schema(cls) -> Dict[str, Any]:
        """
        返回输出 schema（字典格式，用于工作流验证）
        将列表格式转换为字典格式：{output_name: output_type}
        """
        # 将列表格式转为字典格式
        schema_dict = {}
        for output in cls.output_schema:
            output_name = output.get("name")
            output_type = output.get("type", "any")
            if output_name:
                schema_dict[output_name] = output_type
        return schema_dict

    @classmethod
    def get_required_inputs(cls) -> List[str]:
        """
        获取节点所需的输入字段名列表（用于工作流验证）
        """
        return [param["name"] for param in cls.required_inputs if param.get("required", True)]

    async def execute(self, context):
        """
        适配器方法：WorkflowOrchestrator 调用 execute()
        VGP 节点使用 generate()，这里做一个转换
        """
        from nodes.base_node import NodeResult, NodeStatus, ProcessingContext
        from datetime import datetime

        try:
            # 准备 context 数据（VGP 节点期望字典格式）
            context_dict = {}

            # 从 ProcessingContext 提取数据
            if hasattr(context, 'project_data'):
                context_dict.update(context.project_data.get('user_input', {}))
            if hasattr(context, 'intermediate_results'):
                context_dict.update(context.intermediate_results)

            # 调用 generate 方法
            result_data = await self.generate(context_dict)

            # 返回 NodeResult 格式
            return NodeResult(
                status=NodeStatus.COMPLETED,
                data=result_data,
                execution_time=0.0,
                next_nodes=[]
            )

        except Exception as e:
            print(f"❌ VGP Node {self.node_id} execution failed: {e}")
            return NodeResult(
                status=NodeStatus.FAILED,
                data={},
                error_message=str(e),
                execution_time=0.0
            )

    def validate_input(self, context) -> bool:
        """验证输入（WorkflowOrchestrator 需要）"""
        return True  # VGP 节点使用 validate_context，这里简化处理