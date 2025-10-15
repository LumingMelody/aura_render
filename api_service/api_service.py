import requests
from typing import Optional


class APIConfig:
    def __init__(self):
        self.base_url = "https://agent.cstlanbaai.com/gateway"
        self.admin_api_base = f"{self.base_url}/admin-api"
        self.headers = {
            "Content-Type": "application/json",
        }

    def get_headers(self, tenant_id=None):
        """获取请求头，支持租户ID"""
        headers = self.headers.copy()
        if tenant_id:
            headers["Tenant-Id"] = str(tenant_id)
            headers["X-Tenant-Id"] = str(tenant_id)  # 多种格式支持
        return headers

    def update_task_status(self):
        """通用任务状态更新接口 - 所有任务都使用这个"""
        return f"{self.admin_api_base}/agent/task-video-info/update"

    def update_task_video_edit_update(self):
        """数字人视频编辑专用状态更新接口"""
        return f"{self.admin_api_base}/agent/task-video-edit/update"

    def create_resource_url(self):
        """创建资源的接口"""
        return f"{self.admin_api_base}/agent/resource/create"


class APIService:
    def __init__(self, config: Optional[APIConfig] = None):
        self.config = config or APIConfig()
        self.base_url = "https://agent.cstlanbaai.com/gateway"
        self.admin_api_base = f"{self.base_url}/admin-api"

    def _extract_path_from_url(self, url: str) -> str:
        """从完整的OSS URL中提取路径部分（不包含开头的斜杠）"""
        if not url:
            return ""
        
        # 从完整URL中提取路径部分
        # 例如: https://ai-movie-cloud-v2.oss-cn-shanghai.aliyuncs.com/tag_videos/output_1755251188.mp4
        # 提取: tag_videos/output_1755251188.mp4 (移除开头的斜杠)
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            path = parsed.path if parsed.path else url
            # 移除开头的斜杠
            return path.lstrip('/') if path.startswith('/') else path
        except:
            # 如果解析失败，尝试简单的字符串处理
            if 'aliyuncs.com/' in url:
                path = url.split('aliyuncs.com/', 1)[1]
                return path  # 不添加开头的斜杠
            return url

    def update_task_status(self, task_id: str, status: str = "1", tenant_id=None, path: str = "",
                           resource_id=None, resource_ids=None, business_id=None, content=None, api_type="default"):
        """更新任务状态"""
        try:
            # 🔥 根据api_type选择不同的接口
            if api_type == "digital_human":
                url = self.config.update_task_video_edit_update()
                print(f"🤖 [API-UPDATE] 使用数字人专用接口: {url}")
            else:
                url = self.config.update_task_status()
                print(f"📝 [API-UPDATE] 使用通用接口: {url}")

            headers = self.config.get_headers(tenant_id)

            # 提取路径部分，只保存OSS路径而不是完整URL
            extracted_path = self._extract_path_from_url(path)

            payload = {
                "task_id": task_id,
                "status": status,
                "path": extracted_path,
                "resourceId": resource_id,
                "id": business_id
            }

            # 添加resourceIds数组支持
            if resource_ids:
                payload["resourceIds"] = resource_ids

            if content:
                payload["content"] = content

            print(f"🔄 [API-UPDATE] 更新任务状态: {task_id} -> {status} (type: {api_type})")
            print(payload)
            response = requests.put(url, json=payload, headers=headers, timeout=30)

            if response.status_code == 200:
                print(f"✅ [API-UPDATE] 状态更新成功")
                return True
            else:
                print(f"❌ [API-UPDATE] 状态更新失败: {response.status_code}")
                return False

        except Exception as e:
            print(f"❌ [API-UPDATE] 状态更新异常: {str(e)}")
            return False

    def create_resource(self, resource_type: int, name: str, path: str, local_full_path: str, file_type: str, size: int,
                        tenant_id=None):
        """保存资源到素材库"""
        url = self.config.create_resource_url()
        headers = self.config.get_headers(tenant_id)

        # 提取路径部分，只保存OSS路径而不是完整URL
        extracted_path = self._extract_path_from_url(path)

        data = {
            "type": resource_type,
            "name": name,
            "path": extracted_path,
            "fileType": file_type,
            "size": size,
            "configName": "oss-ali-shanghai"
        }

        if tenant_id:
            data["tenantId"] = tenant_id
        if local_full_path:
            data["url"] = local_full_path
        try:
            print(f"create_resource请求体为{data}")
            response = requests.post(url, json=data, headers=headers, timeout=30)

            print(f"✅ 资源保存成功: {name} -> {path}")
            print(f"📤 响应: {response.text}")

            if response.status_code == 200:
                response_data = response.json()

                # 尝试从响应中提取resourceId
                resource_id = None
                possible_id_fields = ['resourceId', 'id', 'data', 'result']
                for field in possible_id_fields:
                    if field in response_data:
                        if field == 'data' and isinstance(response_data[field], dict):
                            resource_id = response_data[field].get('id') or response_data[field].get('resourceId')
                        else:
                            resource_id = response_data[field]
                        if resource_id:
                            break

                return {
                    'response': response_data,
                    'resource_id': resource_id
                }
            else:
                print(f"❌ 资源保存失败: HTTP {response.status_code}")
                return None

        except requests.exceptions.RequestException as e:
            print(f"❌ 资源保存失败: {name}, 错误: {str(e)}")
            return None