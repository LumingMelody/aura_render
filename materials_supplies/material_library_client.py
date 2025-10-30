"""
素材库统一API客户端

用于调用素材库接口获取视频、音频等素材
接口: https://agent.cstlanbaai.com/gateway/admin-api/agent/resource/page
"""

import http.client
import json
import logging
from typing import List, Dict, Any, Optional
from urllib.parse import urlencode

logger = logging.getLogger(__name__)


class MaterialLibraryClient:
    """素材库API客户端"""

    def __init__(self, tenant_id: str, authorization: str = None):
        """
        初始化素材库客户端

        Args:
            tenant_id: 租户ID (从/vgp/generate请求中获取)
            authorization: 认证token (固定值或从环境变量获取)
        """
        self.host = "agent.cstlanbaai.com"
        self.base_path = "/gateway/admin-api/agent/resource/page"
        self.tenant_id = tenant_id
        self.authorization = authorization or self._get_default_authorization()

    def _get_default_authorization(self) -> str:
        """获取默认的Authorization（可以从环境变量或配置读取）"""
        import os
        # TODO: 从环境变量或配置文件读取
        return os.getenv("MATERIAL_LIBRARY_AUTH", "")

    def search_materials(
        self,
        material_type: int,
        name: Optional[str] = None,
        tag: Optional[str] = None,
        file_type: Optional[str] = None,
        page_no: int = 1,
        page_size: int = 10
    ) -> Dict[str, Any]:
        """
        搜索素材

        Args:
            material_type: 资源类型 (1=视频素材库, 2=音频素材库)
            name: 资源名称 (模糊搜索)
            tag: 标签 (用于分类过滤)
            file_type: 文件类型
            page_no: 页码
            page_size: 每页条数

        Returns:
            {
                "code": 0,
                "data": {
                    "list": [
                        {
                            "id": 20944,
                            "name": "xxx",
                            "url": "https://...",
                            "path": "...",
                            "fileType": "1",
                            "size": 0,
                            "createTime": "..."
                        }
                    ],
                    "total": 10
                },
                "msg": "success"
            }
        """
        # 构造查询参数
        params = {
            "type": material_type,
            "pageNo": page_no,
            "pageSize": page_size
        }

        if name:
            params["name"] = name
        if tag:
            params["tag"] = tag
        if file_type:
            params["fileType"] = file_type

        # 构造完整URL
        query_string = urlencode(params)
        full_path = f"{self.base_path}?{query_string}"

        # 构造请求头
        headers = {
            'tenant-id': self.tenant_id,
            'Authorization': self.authorization,
            'Content-Type': 'application/json'
        }

        try:
            # 发送HTTPS请求
            conn = http.client.HTTPSConnection(self.host, timeout=10)
            conn.request("GET", full_path, headers=headers)

            # 获取响应
            response = conn.getresponse()
            data = response.read()
            conn.close()

            # 解析JSON
            result = json.loads(data.decode("utf-8"))

            logger.debug(f"素材库搜索结果: type={material_type}, tag={tag}, 结果数={len(result.get('data', {}).get('list', []))}")

            return result

        except Exception as e:
            logger.error(f"素材库API调用失败: {e}")
            return {
                "code": -1,
                "msg": f"API调用失败: {str(e)}",
                "data": {"list": [], "total": 0}
            }

    def search_videos(
        self,
        keyword: Optional[str] = None,
        tag: Optional[str] = None,
        page_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        搜索视频素材

        Args:
            keyword: 搜索关键字
            tag: 标签
            page_size: 返回结果数量

        Returns:
            [
                {
                    "id": 123,
                    "name": "科技感背景",
                    "url": "https://...",
                    "size": 12345
                },
                ...
            ]
        """
        result = self.search_materials(
            material_type=1,  # 1=视频素材
            name=keyword,
            tag=tag,
            page_size=page_size
        )

        if result.get("code") == 0:
            return result.get("data", {}).get("list", [])
        else:
            logger.warning(f"视频素材搜索失败: {result.get('msg')}")
            return []

    def search_audios(
        self,
        keyword: Optional[str] = None,
        tag: Optional[str] = None,
        page_size: int = 10
    ) -> List[Dict[str, Any]]:
        """
        搜索音频素材 (BGM)

        Args:
            keyword: 搜索关键字
            tag: 标签 (如情绪、风格等)
            page_size: 返回结果数量

        Returns:
            [
                {
                    "id": 456,
                    "name": "冷静科技BGM",
                    "url": "https://...",
                    "size": 3456789
                },
                ...
            ]
        """
        result = self.search_materials(
            material_type=2,  # 2=音频素材
            name=keyword,
            tag=tag,
            page_size=page_size
        )

        if result.get("code") == 0:
            return result.get("data", {}).get("list", [])
        else:
            logger.warning(f"音频素材搜索失败: {result.get('msg')}")
            return []


# 全局客户端实例（需要在请求开始时初始化）
_global_client: Optional[MaterialLibraryClient] = None


def init_material_library_client(tenant_id: str, authorization: str = None):
    """初始化全局素材库客户端"""
    global _global_client
    _global_client = MaterialLibraryClient(tenant_id, authorization)
    logger.info(f"素材库客户端已初始化: tenant_id={tenant_id}")


def get_material_library_client() -> Optional[MaterialLibraryClient]:
    """获取全局素材库客户端"""
    return _global_client
