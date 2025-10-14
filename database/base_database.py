"""
数据库基础接口 - 统一的数据库操作抽象层
"""
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import asyncio
import json
import datetime
from contextlib import asynccontextmanager


class ConnectionStatus(Enum):
    """连接状态"""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


@dataclass
class DatabaseConfig:
    """数据库配置"""
    host: str
    port: int
    database: str
    username: str
    password: str
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    pool_recycle: int = 3600
    ssl: bool = False
    extra_params: Dict[str, Any] = None


@dataclass
class QueryResult:
    """查询结果"""
    rows: List[Dict[str, Any]]
    row_count: int
    execution_time_ms: int
    affected_rows: int = 0
    last_insert_id: Optional[int] = None
    metadata: Dict[str, Any] = None


@dataclass
class TransactionContext:
    """事务上下文"""
    transaction_id: str
    start_time: datetime.datetime
    isolation_level: str = "READ_COMMITTED"
    read_only: bool = False


class BaseDatabaseClient(ABC):
    """数据库客户端基类"""

    def __init__(self, config: DatabaseConfig):
        self.config = config
        self.connection_pool = None
        self.status = ConnectionStatus.DISCONNECTED
        self.connection_count = 0
        self.query_stats = {
            "total_queries": 0,
            "successful_queries": 0,
            "failed_queries": 0,
            "total_execution_time_ms": 0,
            "slow_queries": 0  # > 1000ms
        }

    @abstractmethod
    async def connect(self) -> bool:
        """建立数据库连接"""
        pass

    @abstractmethod
    async def disconnect(self) -> bool:
        """关闭数据库连接"""
        pass

    @abstractmethod
    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """执行查询"""
        pass

    @abstractmethod
    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> QueryResult:
        """批量执行"""
        pass

    @abstractmethod
    async def begin_transaction(self, isolation_level: str = "READ_COMMITTED") -> TransactionContext:
        """开始事务"""
        pass

    @abstractmethod
    async def commit_transaction(self, context: TransactionContext) -> bool:
        """提交事务"""
        pass

    @abstractmethod
    async def rollback_transaction(self, context: TransactionContext) -> bool:
        """回滚事务"""
        pass

    @asynccontextmanager
    async def transaction(self, isolation_level: str = "READ_COMMITTED"):
        """事务上下文管理器"""
        context = await self.begin_transaction(isolation_level)
        try:
            yield context
            await self.commit_transaction(context)
        except Exception as e:
            await self.rollback_transaction(context)
            raise e

    async def insert(self, table: str, data: Dict[str, Any],
                    on_conflict: str = "RAISE") -> QueryResult:
        """插入数据"""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{key}" for key in data.keys())

        conflict_clause = ""
        if on_conflict == "IGNORE":
            conflict_clause = "OR IGNORE"
        elif on_conflict == "REPLACE":
            conflict_clause = "OR REPLACE"

        query = f"INSERT {conflict_clause} INTO {table} ({columns}) VALUES ({placeholders})"
        return await self.execute_query(query, data)

    async def select(self, table: str,
                    columns: List[str] = None,
                    where: Dict[str, Any] = None,
                    order_by: List[str] = None,
                    limit: int = None,
                    offset: int = None) -> QueryResult:
        """查询数据"""
        columns_str = ", ".join(columns) if columns else "*"
        query = f"SELECT {columns_str} FROM {table}"

        params = {}
        if where:
            where_clauses = []
            for key, value in where.items():
                if isinstance(value, (list, tuple)):
                    placeholders = ", ".join(f":{key}_{i}" for i in range(len(value)))
                    where_clauses.append(f"{key} IN ({placeholders})")
                    for i, v in enumerate(value):
                        params[f"{key}_{i}"] = v
                else:
                    where_clauses.append(f"{key} = :{key}")
                    params[key] = value

            query += " WHERE " + " AND ".join(where_clauses)

        if order_by:
            query += " ORDER BY " + ", ".join(order_by)

        if limit:
            query += f" LIMIT {limit}"

        if offset:
            query += f" OFFSET {offset}"

        return await self.execute_query(query, params)

    async def update(self, table: str, data: Dict[str, Any],
                    where: Dict[str, Any]) -> QueryResult:
        """更新数据"""
        set_clauses = []
        params = {}

        for key, value in data.items():
            set_clauses.append(f"{key} = :set_{key}")
            params[f"set_{key}"] = value

        where_clauses = []
        for key, value in where.items():
            where_clauses.append(f"{key} = :where_{key}")
            params[f"where_{key}"] = value

        query = f"UPDATE {table} SET {', '.join(set_clauses)} WHERE {' AND '.join(where_clauses)}"
        return await self.execute_query(query, params)

    async def delete(self, table: str, where: Dict[str, Any]) -> QueryResult:
        """删除数据"""
        where_clauses = []
        params = {}

        for key, value in where.items():
            where_clauses.append(f"{key} = :{key}")
            params[key] = value

        query = f"DELETE FROM {table} WHERE {' AND '.join(where_clauses)}"
        return await self.execute_query(query, params)

    async def bulk_insert(self, table: str, data_list: List[Dict[str, Any]],
                         batch_size: int = 1000) -> List[QueryResult]:
        """批量插入"""
        if not data_list:
            return []

        results = []
        columns = ", ".join(data_list[0].keys())
        placeholders = ", ".join(f":{key}" for key in data_list[0].keys())
        query = f"INSERT INTO {table} ({columns}) VALUES ({placeholders})"

        # 分批处理
        for i in range(0, len(data_list), batch_size):
            batch = data_list[i:i + batch_size]
            result = await self.execute_many(query, batch)
            results.append(result)

        return results

    async def upsert(self, table: str, data: Dict[str, Any],
                    conflict_columns: List[str]) -> QueryResult:
        """插入或更新（需要子类实现具体的UPSERT语法）"""
        raise NotImplementedError("Upsert must be implemented by database-specific client")

    async def count(self, table: str, where: Dict[str, Any] = None) -> int:
        """计数"""
        result = await self.select(table, columns=["COUNT(*) as count"], where=where)
        return result.rows[0]["count"] if result.rows else 0

    async def exists(self, table: str, where: Dict[str, Any]) -> bool:
        """检查记录是否存在"""
        result = await self.select(table, columns=["1"], where=where, limit=1)
        return len(result.rows) > 0

    async def get_table_info(self, table: str) -> Dict[str, Any]:
        """获取表信息"""
        raise NotImplementedError("Table info must be implemented by database-specific client")

    async def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        return {
            "status": self.status.value,
            "connection_count": self.connection_count,
            "config": {
                "host": self.config.host,
                "port": self.config.port,
                "database": self.config.database,
                "pool_size": self.config.pool_size
            },
            "stats": self.query_stats
        }

    async def health_check(self) -> bool:
        """健康检查"""
        try:
            result = await self.execute_query("SELECT 1 as health_check")
            return len(result.rows) > 0 and result.rows[0]["health_check"] == 1
        except Exception:
            return False

    def _update_query_stats(self, execution_time_ms: int, success: bool):
        """更新查询统计"""
        self.query_stats["total_queries"] += 1
        self.query_stats["total_execution_time_ms"] += execution_time_ms

        if success:
            self.query_stats["successful_queries"] += 1
        else:
            self.query_stats["failed_queries"] += 1

        if execution_time_ms > 1000:
            self.query_stats["slow_queries"] += 1

    def get_average_query_time(self) -> float:
        """获取平均查询时间"""
        total_queries = self.query_stats["total_queries"]
        if total_queries == 0:
            return 0.0

        return self.query_stats["total_execution_time_ms"] / total_queries

    async def optimize_table(self, table: str) -> bool:
        """优化表（需要子类实现）"""
        return True

    async def analyze_table(self, table: str) -> Dict[str, Any]:
        """分析表（需要子类实现）"""
        return {"analyzed": True}

    async def vacuum(self) -> bool:
        """清理数据库（需要子类实现）"""
        return True