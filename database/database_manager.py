"""
数据库管理器 - 统一管理多种数据库连接和操作
"""
from typing import Dict, List, Any, Optional, Union, Type
import asyncio
from dataclasses import dataclass, asdict
from enum import Enum
import time

from .base_database import (
    BaseDatabaseClient,
    DatabaseConfig,
    QueryResult,
    TransactionContext,
    ConnectionStatus
)
from .sqlite_client import SQLiteClient
from .postgresql_client import PostgreSQLClient


class DatabaseType(Enum):
    """数据库类型"""
    SQLITE = "sqlite"
    POSTGRESQL = "postgresql"
    MYSQL = "mysql"


@dataclass
class DatabaseConnectionConfig:
    """数据库连接配置"""
    name: str  # 连接名称
    type: DatabaseType
    config: DatabaseConfig
    primary: bool = False  # 是否为主数据库
    read_only: bool = False  # 是否只读
    enabled: bool = True


@dataclass
class TableMapping:
    """表映射配置"""
    table_name: str
    database_name: str  # 映射到的数据库连接名
    read_preference: Optional[str] = None  # 读取偏好的数据库
    write_preference: Optional[str] = None  # 写入偏好的数据库


class DatabaseManager:
    """数据库管理器"""

    def __init__(self):
        self.connections: Dict[str, BaseDatabaseClient] = {}
        self.configs: Dict[str, DatabaseConnectionConfig] = {}
        self.table_mappings: Dict[str, TableMapping] = {}
        self.primary_db: Optional[str] = None

        # 性能统计
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time_ms": 0,
            "queries_per_db": {},
            "slow_queries": []  # 慢查询记录
        }

    async def add_connection(self, connection_config: DatabaseConnectionConfig) -> bool:
        """添加数据库连接"""
        if not connection_config.enabled:
            return False

        try:
            # 创建数据库客户端
            if connection_config.type == DatabaseType.SQLITE:
                client = SQLiteClient(connection_config.config.database)
            elif connection_config.type == DatabaseType.POSTGRESQL:
                client = PostgreSQLClient(connection_config.config)
            else:
                raise ValueError(f"Unsupported database type: {connection_config.type}")

            # 连接数据库
            if await client.connect():
                self.connections[connection_config.name] = client
                self.configs[connection_config.name] = connection_config

                # 设置主数据库
                if connection_config.primary or not self.primary_db:
                    self.primary_db = connection_config.name

                # 初始化统计
                self.query_stats["queries_per_db"][connection_config.name] = {
                    "total": 0,
                    "successful": 0,
                    "failed": 0,
                    "avg_time_ms": 0.0
                }

                print(f"✅ Database connected: {connection_config.name} ({connection_config.type.value})")
                return True
            else:
                print(f"❌ Failed to connect database: {connection_config.name}")
                return False

        except Exception as e:
            print(f"❌ Database connection error for {connection_config.name}: {e}")
            return False

    async def remove_connection(self, name: str) -> bool:
        """移除数据库连接"""
        if name not in self.connections:
            return False

        try:
            client = self.connections[name]
            await client.disconnect()

            del self.connections[name]
            del self.configs[name]

            if name in self.query_stats["queries_per_db"]:
                del self.query_stats["queries_per_db"][name]

            # 如果移除的是主数据库，重新选择主数据库
            if self.primary_db == name:
                self.primary_db = next(iter(self.connections.keys())) if self.connections else None

            print(f"✅ Database disconnected: {name}")
            return True

        except Exception as e:
            print(f"❌ Database disconnection error for {name}: {e}")
            return False

    def map_table(self, table_name: str, database_name: str,
                  read_preference: Optional[str] = None,
                  write_preference: Optional[str] = None):
        """映射表到特定数据库"""
        self.table_mappings[table_name] = TableMapping(
            table_name=table_name,
            database_name=database_name,
            read_preference=read_preference,
            write_preference=write_preference
        )

    def get_database_for_table(self, table_name: str, operation: str = "read") -> str:
        """获取表对应的数据库连接名"""
        if table_name in self.table_mappings:
            mapping = self.table_mappings[table_name]

            if operation == "read" and mapping.read_preference:
                return mapping.read_preference
            elif operation == "write" and mapping.write_preference:
                return mapping.write_preference
            else:
                return mapping.database_name

        # 默认使用主数据库
        return self.primary_db or next(iter(self.connections.keys()), None)

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None,
                          database: Optional[str] = None,
                          table_hint: Optional[str] = None) -> QueryResult:
        """执行查询"""
        # 确定使用的数据库
        if database:
            db_name = database
        elif table_hint:
            operation = "write" if query.strip().upper().startswith(('INSERT', 'UPDATE', 'DELETE')) else "read"
            db_name = self.get_database_for_table(table_hint, operation)
        else:
            db_name = self.primary_db

        if not db_name or db_name not in self.connections:
            raise ValueError(f"Database not found: {db_name}")

        start_time = time.time()
        client = self.connections[db_name]

        try:
            result = await client.execute_query(query, params)
            execution_time_ms = int((time.time() - start_time) * 1000)

            # 更新统计
            self._update_query_stats(db_name, execution_time_ms, True, query)

            return result

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self._update_query_stats(db_name, execution_time_ms, False, query)
            raise e

    async def execute_transaction(self, operations: List[Dict[str, Any]],
                                database: Optional[str] = None,
                                isolation_level: str = "READ_COMMITTED") -> List[QueryResult]:
        """执行事务"""
        db_name = database or self.primary_db
        if not db_name or db_name not in self.connections:
            raise ValueError(f"Database not found: {db_name}")

        client = self.connections[db_name]
        results = []

        async with client.transaction(isolation_level):
            for operation in operations:
                query = operation.get("query")
                params = operation.get("params")
                table_hint = operation.get("table_hint")

                result = await self.execute_query(query, params, db_name, table_hint)
                results.append(result)

        return results

    async def insert(self, table: str, data: Dict[str, Any],
                    on_conflict: str = "RAISE",
                    database: Optional[str] = None) -> QueryResult:
        """插入数据"""
        db_name = database or self.get_database_for_table(table, "write")
        client = self.connections[db_name]
        return await client.insert(table, data, on_conflict)

    async def select(self, table: str,
                    columns: List[str] = None,
                    where: Dict[str, Any] = None,
                    order_by: List[str] = None,
                    limit: int = None,
                    offset: int = None,
                    database: Optional[str] = None) -> QueryResult:
        """查询数据"""
        db_name = database or self.get_database_for_table(table, "read")
        client = self.connections[db_name]
        return await client.select(table, columns, where, order_by, limit, offset)

    async def update(self, table: str, data: Dict[str, Any],
                    where: Dict[str, Any],
                    database: Optional[str] = None) -> QueryResult:
        """更新数据"""
        db_name = database or self.get_database_for_table(table, "write")
        client = self.connections[db_name]
        return await client.update(table, data, where)

    async def delete(self, table: str, where: Dict[str, Any],
                    database: Optional[str] = None) -> QueryResult:
        """删除数据"""
        db_name = database or self.get_database_for_table(table, "write")
        client = self.connections[db_name]
        return await client.delete(table, where)

    async def bulk_insert(self, table: str, data_list: List[Dict[str, Any]],
                         batch_size: int = 1000,
                         database: Optional[str] = None) -> List[QueryResult]:
        """批量插入数据"""
        db_name = database or self.get_database_for_table(table, "write")
        client = self.connections[db_name]
        return await client.bulk_insert(table, data_list, batch_size)

    async def upsert(self, table: str, data: Dict[str, Any],
                    conflict_columns: List[str],
                    database: Optional[str] = None) -> QueryResult:
        """插入或更新数据"""
        db_name = database or self.get_database_for_table(table, "write")
        client = self.connections[db_name]
        return await client.upsert(table, data, conflict_columns)

    async def count(self, table: str, where: Dict[str, Any] = None,
                   database: Optional[str] = None) -> int:
        """计数"""
        db_name = database or self.get_database_for_table(table, "read")
        client = self.connections[db_name]
        return await client.count(table, where)

    async def exists(self, table: str, where: Dict[str, Any],
                    database: Optional[str] = None) -> bool:
        """检查记录是否存在"""
        db_name = database or self.get_database_for_table(table, "read")
        client = self.connections[db_name]
        return await client.exists(table, where)

    async def get_table_info(self, table: str,
                           database: Optional[str] = None) -> Dict[str, Any]:
        """获取表信息"""
        db_name = database or self.get_database_for_table(table, "read")
        client = self.connections[db_name]
        return await client.get_table_info(table)

    async def create_table(self, table: str, columns: Dict[str, str],
                          primary_key: List[str] = None,
                          indexes: List[Dict[str, Any]] = None,
                          database: Optional[str] = None) -> bool:
        """创建表"""
        db_name = database or self.primary_db
        if db_name not in self.connections:
            return False

        client = self.connections[db_name]
        return await client.create_table(table, columns, primary_key, indexes)

    async def migrate_data(self, table: str, source_db: str, target_db: str,
                          batch_size: int = 1000) -> Dict[str, Any]:
        """迁移表数据"""
        if source_db not in self.connections or target_db not in self.connections:
            raise ValueError("Source or target database not found")

        source_client = self.connections[source_db]
        target_client = self.connections[target_db]

        start_time = time.time()
        total_rows = 0
        migrated_rows = 0

        try:
            # 获取源表总行数
            total_rows = await source_client.count(table)

            # 分批迁移数据
            offset = 0
            while offset < total_rows:
                # 从源数据库读取数据
                result = await source_client.select(
                    table,
                    limit=batch_size,
                    offset=offset
                )

                if result.rows:
                    # 写入目标数据库
                    await target_client.bulk_insert(table, result.rows)
                    migrated_rows += len(result.rows)

                offset += batch_size

            execution_time = time.time() - start_time

            return {
                "success": True,
                "total_rows": total_rows,
                "migrated_rows": migrated_rows,
                "execution_time_seconds": execution_time,
                "rows_per_second": migrated_rows / execution_time if execution_time > 0 else 0
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "total_rows": total_rows,
                "migrated_rows": migrated_rows,
                "execution_time_seconds": time.time() - start_time
            }

    async def health_check(self) -> Dict[str, bool]:
        """健康检查"""
        health_status = {}

        for name, client in self.connections.items():
            try:
                health_status[name] = await client.health_check()
            except Exception:
                health_status[name] = False

        return health_status

    async def get_connection_stats(self) -> Dict[str, Any]:
        """获取连接统计信息"""
        stats = {}

        for name, client in self.connections.items():
            try:
                stats[name] = await client.get_connection_info()
            except Exception as e:
                stats[name] = {"error": str(e)}

        return stats

    async def optimize_all_databases(self) -> Dict[str, Any]:
        """优化所有数据库"""
        optimization_results = {}

        for name, client in self.connections.items():
            try:
                if hasattr(client, 'vacuum'):
                    result = await client.vacuum()
                    optimization_results[name] = {"vacuum": result}
                else:
                    optimization_results[name] = {"vacuum": "not_supported"}
            except Exception as e:
                optimization_results[name] = {"error": str(e)}

        return optimization_results

    async def backup_database(self, database_name: str, backup_path: str) -> bool:
        """备份数据库"""
        if database_name not in self.connections:
            return False

        client = self.connections[database_name]

        try:
            if hasattr(client, 'backup_database'):
                return await client.backup_database(backup_path)
            else:
                print(f"Backup not supported for database type: {self.configs[database_name].type}")
                return False
        except Exception as e:
            print(f"Backup failed for {database_name}: {e}")
            return False

    def _update_query_stats(self, db_name: str, execution_time_ms: int,
                           success: bool, query: str):
        """更新查询统计"""
        # 全局统计
        self.query_stats["total_queries"] += 1
        self.query_stats["total_execution_time_ms"] += execution_time_ms

        if success:
            self.query_stats["successful_queries"] += 1
        else:
            self.query_stats["failed_queries"] += 1

        # 记录慢查询（超过1秒）
        if execution_time_ms > 1000:
            self.query_stats["slow_queries"].append({
                "database": db_name,
                "query": query[:200] + "..." if len(query) > 200 else query,
                "execution_time_ms": execution_time_ms,
                "timestamp": time.time()
            })

            # 只保留最近的50个慢查询
            if len(self.query_stats["slow_queries"]) > 50:
                self.query_stats["slow_queries"] = self.query_stats["slow_queries"][-50:]

        # 数据库特定统计
        db_stats = self.query_stats["queries_per_db"][db_name]
        db_stats["total"] += 1

        if success:
            db_stats["successful"] += 1

            # 计算平均时间
            total_time = db_stats["avg_time_ms"] * (db_stats["successful"] - 1) + execution_time_ms
            db_stats["avg_time_ms"] = total_time / db_stats["successful"]
        else:
            db_stats["failed"] += 1

    async def get_query_stats(self) -> Dict[str, Any]:
        """获取查询统计信息"""
        total_queries = self.query_stats["total_queries"]
        avg_time = (self.query_stats["total_execution_time_ms"] / total_queries) if total_queries > 0 else 0

        return {
            "global_stats": {
                "total_queries": total_queries,
                "successful_queries": self.query_stats["successful_queries"],
                "failed_queries": self.query_stats["failed_queries"],
                "success_rate": (self.query_stats["successful_queries"] / total_queries) if total_queries > 0 else 0,
                "average_execution_time_ms": avg_time,
                "slow_queries_count": len(self.query_stats["slow_queries"])
            },
            "per_database_stats": self.query_stats["queries_per_db"],
            "recent_slow_queries": self.query_stats["slow_queries"][-10:]  # 最近10个慢查询
        }

    async def get_database_sizes(self) -> Dict[str, Any]:
        """获取所有数据库的大小信息"""
        sizes = {}

        for name, client in self.connections.items():
            try:
                if hasattr(client, 'get_database_size'):
                    sizes[name] = await client.get_database_size()
                else:
                    sizes[name] = {"error": "Size info not supported"}
            except Exception as e:
                sizes[name] = {"error": str(e)}

        return sizes

    async def disconnect_all(self) -> bool:
        """断开所有数据库连接"""
        success = True

        for name in list(self.connections.keys()):
            if not await self.remove_connection(name):
                success = False

        return success

    def get_available_databases(self) -> List[str]:
        """获取可用的数据库连接名列表"""
        return list(self.connections.keys())

    def get_primary_database(self) -> Optional[str]:
        """获取主数据库名"""
        return self.primary_db

    async def set_primary_database(self, database_name: str) -> bool:
        """设置主数据库"""
        if database_name in self.connections:
            self.primary_db = database_name
            return True
        return False