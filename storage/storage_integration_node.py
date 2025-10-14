"""
存储集成节点 - 统一的数据库和缓存操作节点
"""
from typing import Dict, List, Any, Optional, Union
import asyncio
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from database import (
    DatabaseManager,
    DatabaseType,
    DatabaseConnectionConfig,
    DatabaseConfig
)
from cache import (
    CacheManager,
    CacheType,
    CacheLayerConfig,
    CacheConfig,
    CachePolicy,
    CacheStrategy
)


@dataclass
class StorageRequest:
    """存储请求"""
    operation: str  # get, set, delete, query, insert, update, upsert
    table_or_key: str
    data: Optional[Dict[str, Any]] = None
    where: Optional[Dict[str, Any]] = None
    params: Optional[Dict[str, Any]] = None
    cache_ttl: Optional[int] = None
    use_cache: bool = True
    database: Optional[str] = None


@dataclass
class StorageResponse:
    """存储响应"""
    success: bool
    data: Any = None
    row_count: int = 0
    affected_rows: int = 0
    cache_hit: bool = False
    execution_time_ms: int = 0
    error_message: Optional[str] = None
    metadata: Dict[str, Any] = None


@dataclass
class StorageNodeConfig:
    """存储节点配置"""
    databases: List[Dict[str, Any]] = None
    caches: List[Dict[str, Any]] = None
    cache_policy: Dict[str, Any] = None
    auto_initialize: bool = True


class StorageIntegrationNode:
    """存储集成节点"""

    def __init__(self, config: StorageNodeConfig = None):
        self.config = config or StorageNodeConfig()

        # 管理器
        self.db_manager = DatabaseManager()
        self.cache_manager = CacheManager()

        # 初始化状态
        self.initialized = False

    async def initialize(self):
        """初始化存储系统"""
        if self.initialized:
            return True

        try:
            # 初始化数据库连接
            await self._initialize_databases()

            # 初始化缓存层
            await self._initialize_caches()

            self.initialized = True
            print("✅ Storage integration node initialized")
            return True

        except Exception as e:
            print(f"❌ Storage initialization failed: {e}")
            return False

    async def _initialize_databases(self):
        """初始化数据库连接"""
        database_configs = self.config.databases or []

        # 如果没有配置，使用默认SQLite数据库
        if not database_configs:
            default_db_path = os.path.join(os.getcwd(), "aura_render.db")
            database_configs = [{
                "name": "default",
                "type": "sqlite",
                "database": default_db_path,
                "primary": True
            }]

        for db_config in database_configs:
            # 创建数据库配置
            if db_config["type"] == "sqlite":
                config = DatabaseConfig(
                    host="localhost",
                    port=0,
                    database=db_config["database"],
                    username="",
                    password=""
                )
            elif db_config["type"] == "postgresql":
                config = DatabaseConfig(
                    host=db_config.get("host", "localhost"),
                    port=db_config.get("port", 5432),
                    database=db_config["database"],
                    username=db_config["username"],
                    password=db_config["password"],
                    pool_size=db_config.get("pool_size", 10)
                )
            else:
                continue

            # 创建连接配置
            connection_config = DatabaseConnectionConfig(
                name=db_config["name"],
                type=DatabaseType(db_config["type"]),
                config=config,
                primary=db_config.get("primary", False),
                read_only=db_config.get("read_only", False),
                enabled=db_config.get("enabled", True)
            )

            await self.db_manager.add_connection(connection_config)

    async def _initialize_caches(self):
        """初始化缓存层"""
        cache_configs = self.config.caches or []

        # 如果没有配置，使用默认内存缓存
        if not cache_configs:
            cache_configs = [{
                "name": "memory",
                "type": "memory",
                "level": 1,
                "max_size": 10000
            }]

        # 创建缓存策略
        policy_config = self.config.cache_policy or {}
        cache_policy = CachePolicy(
            default_ttl=policy_config.get("default_ttl", 3600),
            strategy=CacheStrategy(policy_config.get("strategy", "cache_aside")),
            compression_threshold=policy_config.get("compression_threshold", 1024)
        )

        self.cache_manager = CacheManager(cache_policy)

        for cache_config in cache_configs:
            if cache_config["type"] == "memory":
                config = CacheConfig(
                    host="localhost",
                    port=0,
                    default_ttl=cache_config.get("default_ttl", 3600)
                )
            elif cache_config["type"] == "redis":
                config = CacheConfig(
                    host=cache_config.get("host", "localhost"),
                    port=cache_config.get("port", 6379),
                    password=cache_config.get("password"),
                    database=cache_config.get("database", 0),
                    max_connections=cache_config.get("max_connections", 10),
                    default_ttl=cache_config.get("default_ttl", 3600)
                )
            else:
                continue

            # 创建缓存层配置
            layer_config = CacheLayerConfig(
                name=cache_config["name"],
                type=CacheType(cache_config["type"]),
                config=config,
                level=cache_config.get("level", 1),
                enabled=cache_config.get("enabled", True)
            )

            await self.cache_manager.add_cache_layer(layer_config)

    async def process(self, request: StorageRequest) -> StorageResponse:
        """处理存储请求"""
        if not self.initialized:
            await self.initialize()

        start_time = time.time()

        try:
            # 根据操作类型分发
            if request.operation == "get":
                return await self._handle_get(request, start_time)
            elif request.operation == "set":
                return await self._handle_set(request, start_time)
            elif request.operation == "delete":
                return await self._handle_delete(request, start_time)
            elif request.operation == "query":
                return await self._handle_query(request, start_time)
            elif request.operation == "insert":
                return await self._handle_insert(request, start_time)
            elif request.operation == "update":
                return await self._handle_update(request, start_time)
            elif request.operation == "upsert":
                return await self._handle_upsert(request, start_time)
            else:
                return StorageResponse(
                    success=False,
                    error_message=f"Unsupported operation: {request.operation}",
                    execution_time_ms=int((time.time() - start_time) * 1000)
                )

        except Exception as e:
            return StorageResponse(
                success=False,
                error_message=str(e),
                execution_time_ms=int((time.time() - start_time) * 1000)
            )

    async def _handle_get(self, request: StorageRequest, start_time: float) -> StorageResponse:
        """处理缓存获取请求"""
        cache_hit = False
        data = None

        if request.use_cache:
            data = await self.cache_manager.get(request.table_or_key)
            cache_hit = data is not None

        execution_time_ms = int((time.time() - start_time) * 1000)

        return StorageResponse(
            success=True,
            data=data,
            cache_hit=cache_hit,
            execution_time_ms=execution_time_ms
        )

    async def _handle_set(self, request: StorageRequest, start_time: float) -> StorageResponse:
        """处理缓存设置请求"""
        success = False

        if request.use_cache and request.data is not None:
            success = await self.cache_manager.set(
                request.table_or_key,
                request.data,
                request.cache_ttl
            )

        execution_time_ms = int((time.time() - start_time) * 1000)

        return StorageResponse(
            success=success,
            execution_time_ms=execution_time_ms
        )

    async def _handle_delete(self, request: StorageRequest, start_time: float) -> StorageResponse:
        """处理删除请求"""
        cache_success = True
        db_success = True
        affected_rows = 0

        # 从缓存删除
        if request.use_cache:
            cache_success = await self.cache_manager.delete(request.table_or_key)

        # 从数据库删除
        if request.where:
            try:
                result = await self.db_manager.delete(
                    request.table_or_key,
                    request.where,
                    request.database
                )
                affected_rows = result.affected_rows
            except Exception as e:
                db_success = False
                print(f"Database delete failed: {e}")

        execution_time_ms = int((time.time() - start_time) * 1000)

        return StorageResponse(
            success=cache_success and db_success,
            affected_rows=affected_rows,
            execution_time_ms=execution_time_ms
        )

    async def _handle_query(self, request: StorageRequest, start_time: float) -> StorageResponse:
        """处理数据库查询请求"""
        try:
            # 生成缓存键
            cache_key = None
            cache_hit = False
            data = None

            if request.use_cache:
                cache_key = self._generate_query_cache_key(request)
                data = await self.cache_manager.get(cache_key)
                cache_hit = data is not None

            if not cache_hit:
                # 从数据库查询
                if request.params:
                    # 原生SQL查询
                    result = await self.db_manager.execute_query(
                        request.table_or_key,
                        request.params,
                        request.database
                    )
                else:
                    # 表查询
                    result = await self.db_manager.select(
                        request.table_or_key,
                        where=request.where,
                        database=request.database
                    )

                data = result.rows

                # 缓存查询结果
                if request.use_cache and cache_key and data:
                    await self.cache_manager.set(cache_key, data, request.cache_ttl)

            execution_time_ms = int((time.time() - start_time) * 1000)

            return StorageResponse(
                success=True,
                data=data,
                row_count=len(data) if data else 0,
                cache_hit=cache_hit,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            return StorageResponse(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )

    async def _handle_insert(self, request: StorageRequest, start_time: float) -> StorageResponse:
        """处理插入请求"""
        try:
            result = await self.db_manager.insert(
                request.table_or_key,
                request.data,
                database=request.database
            )

            # 清除相关缓存
            if request.use_cache:
                await self._invalidate_table_cache(request.table_or_key)

            execution_time_ms = int((time.time() - start_time) * 1000)

            return StorageResponse(
                success=True,
                data={"id": result.last_insert_id} if result.last_insert_id else None,
                affected_rows=result.affected_rows,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            return StorageResponse(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )

    async def _handle_update(self, request: StorageRequest, start_time: float) -> StorageResponse:
        """处理更新请求"""
        try:
            result = await self.db_manager.update(
                request.table_or_key,
                request.data,
                request.where,
                request.database
            )

            # 清除相关缓存
            if request.use_cache:
                await self._invalidate_table_cache(request.table_or_key)

            execution_time_ms = int((time.time() - start_time) * 1000)

            return StorageResponse(
                success=True,
                affected_rows=result.affected_rows,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            return StorageResponse(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )

    async def _handle_upsert(self, request: StorageRequest, start_time: float) -> StorageResponse:
        """处理插入或更新请求"""
        try:
            conflict_columns = request.params.get("conflict_columns", []) if request.params else []

            result = await self.db_manager.upsert(
                request.table_or_key,
                request.data,
                conflict_columns,
                request.database
            )

            # 清除相关缓存
            if request.use_cache:
                await self._invalidate_table_cache(request.table_or_key)

            execution_time_ms = int((time.time() - start_time) * 1000)

            return StorageResponse(
                success=True,
                data=result.rows[0] if result.rows else None,
                affected_rows=result.affected_rows,
                execution_time_ms=execution_time_ms
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            return StorageResponse(
                success=False,
                error_message=str(e),
                execution_time_ms=execution_time_ms
            )

    def _generate_query_cache_key(self, request: StorageRequest) -> str:
        """生成查询缓存键"""
        key_parts = [
            "query",
            request.table_or_key,
            str(hash(str(request.where))),
            str(hash(str(request.params)))
        ]
        return ":".join(key_parts)

    async def _invalidate_table_cache(self, table_name: str):
        """清除表相关的缓存"""
        pattern = f"query:{table_name}:*"
        await self.cache_manager.invalidate_pattern(pattern)

    # 便捷方法
    async def get_from_cache(self, key: str, default=None) -> Any:
        """从缓存获取数据"""
        request = StorageRequest(operation="get", table_or_key=key)
        response = await self.process(request)
        return response.data if response.success else default

    async def set_cache(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """设置缓存数据"""
        request = StorageRequest(
            operation="set",
            table_or_key=key,
            data=value,
            cache_ttl=ttl
        )
        response = await self.process(request)
        return response.success

    async def query_table(self, table: str, where: Optional[Dict[str, Any]] = None,
                         use_cache: bool = True, cache_ttl: Optional[int] = None) -> List[Dict[str, Any]]:
        """查询表数据"""
        request = StorageRequest(
            operation="query",
            table_or_key=table,
            where=where,
            use_cache=use_cache,
            cache_ttl=cache_ttl
        )
        response = await self.process(request)
        return response.data if response.success else []

    async def insert_record(self, table: str, data: Dict[str, Any]) -> Optional[int]:
        """插入记录"""
        request = StorageRequest(
            operation="insert",
            table_or_key=table,
            data=data
        )
        response = await self.process(request)
        if response.success and response.data:
            return response.data.get("id")
        return None

    async def update_records(self, table: str, data: Dict[str, Any],
                           where: Dict[str, Any]) -> int:
        """更新记录"""
        request = StorageRequest(
            operation="update",
            table_or_key=table,
            data=data,
            where=where
        )
        response = await self.process(request)
        return response.affected_rows if response.success else 0

    async def delete_records(self, table: str, where: Dict[str, Any]) -> int:
        """删除记录"""
        request = StorageRequest(
            operation="delete",
            table_or_key=table,
            where=where
        )
        response = await self.process(request)
        return response.affected_rows if response.success else 0

    async def get_storage_stats(self) -> Dict[str, Any]:
        """获取存储统计信息"""
        try:
            db_stats = await self.db_manager.get_query_stats()
            cache_stats = await self.cache_manager.get_stats()
            health_status = {
                "database": await self.db_manager.health_check(),
                "cache": await self.cache_manager.health_check()
            }

            return {
                "database_stats": db_stats,
                "cache_stats": cache_stats,
                "health_status": health_status,
                "initialized": self.initialized
            }

        except Exception as e:
            return {"error": str(e)}

    async def optimize_storage(self) -> Dict[str, Any]:
        """优化存储性能"""
        try:
            # 优化数据库
            db_optimization = await self.db_manager.optimize_all_databases()

            # 优化缓存（如果支持）
            cache_optimization = {}
            for name, client in self.cache_manager.cache_layers.items():
                if hasattr(client, 'optimize'):
                    cache_optimization[name] = await client.optimize()

            return {
                "database_optimization": db_optimization,
                "cache_optimization": cache_optimization,
                "timestamp": time.time()
            }

        except Exception as e:
            return {"error": str(e)}

    async def cleanup(self):
        """清理资源"""
        try:
            await self.db_manager.disconnect_all()
            await self.cache_manager.disconnect_all()
            self.initialized = False
            print("✅ Storage integration node cleaned up")
        except Exception as e:
            print(f"❌ Storage cleanup failed: {e}")

    def __del__(self):
        """析构函数"""
        try:
            asyncio.create_task(self.cleanup())
        except:
            pass