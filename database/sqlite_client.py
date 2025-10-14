"""
SQLite 数据库客户端
"""
from typing import Dict, List, Any, Optional, Union
import asyncio
import aiosqlite
import time
import uuid
import datetime
import json
from pathlib import Path

from .base_database import (
    BaseDatabaseClient,
    DatabaseConfig,
    QueryResult,
    TransactionContext,
    ConnectionStatus
)


class SQLiteClient(BaseDatabaseClient):
    """SQLite 数据库客户端"""

    def __init__(self, database_path: str, **kwargs):
        """
        初始化SQLite客户端

        Args:
            database_path: 数据库文件路径
            **kwargs: 其他配置参数
        """
        config = DatabaseConfig(
            host="localhost",
            port=0,
            database=database_path,
            username="",
            password="",
            pool_size=kwargs.get("pool_size", 5),
            **kwargs
        )
        super().__init__(config)

        self.database_path = database_path
        self.connection_pool = []
        self.active_transactions = {}
        self.connection_lock = asyncio.Lock()

        # SQLite特定配置
        self.pragma_settings = {
            "journal_mode": "WAL",  # Write-Ahead Logging
            "synchronous": "NORMAL",
            "cache_size": 10000,
            "foreign_keys": "ON",
            "temp_store": "MEMORY"
        }

    async def connect(self) -> bool:
        """建立SQLite连接"""
        try:
            self.status = ConnectionStatus.CONNECTING

            # 确保数据库目录存在
            db_path = Path(self.database_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)

            # 创建连接池
            async with self.connection_lock:
                for _ in range(self.config.pool_size):
                    conn = await aiosqlite.connect(
                        self.database_path,
                        timeout=30.0,
                        check_same_thread=False
                    )

                    # 应用PRAGMA设置
                    for pragma, value in self.pragma_settings.items():
                        await conn.execute(f"PRAGMA {pragma} = {value}")

                    # 启用行工厂
                    conn.row_factory = aiosqlite.Row

                    self.connection_pool.append(conn)
                    self.connection_count += 1

            self.status = ConnectionStatus.CONNECTED
            print(f"✅ SQLite connected: {self.database_path}")
            return True

        except Exception as e:
            self.status = ConnectionStatus.ERROR
            print(f"❌ SQLite connection failed: {e}")
            return False

    async def disconnect(self) -> bool:
        """关闭SQLite连接"""
        try:
            async with self.connection_lock:
                while self.connection_pool:
                    conn = self.connection_pool.pop()
                    await conn.close()
                    self.connection_count -= 1

            self.status = ConnectionStatus.DISCONNECTED
            print("✅ SQLite disconnected")
            return True

        except Exception as e:
            print(f"❌ SQLite disconnect failed: {e}")
            return False

    async def _get_connection(self):
        """获取连接"""
        async with self.connection_lock:
            if not self.connection_pool:
                # 重新创建连接
                conn = await aiosqlite.connect(
                    self.database_path,
                    timeout=30.0,
                    check_same_thread=False
                )
                for pragma, value in self.pragma_settings.items():
                    await conn.execute(f"PRAGMA {pragma} = {value}")
                conn.row_factory = aiosqlite.Row
                return conn
            else:
                return self.connection_pool.pop()

    async def _return_connection(self, conn):
        """归还连接"""
        async with self.connection_lock:
            if len(self.connection_pool) < self.config.pool_size:
                self.connection_pool.append(conn)
            else:
                await conn.close()

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """执行查询"""
        start_time = time.time()
        conn = None

        try:
            conn = await self._get_connection()

            if params:
                cursor = await conn.execute(query, params)
            else:
                cursor = await conn.execute(query)

            # 获取结果
            if query.strip().upper().startswith(('SELECT', 'PRAGMA', 'EXPLAIN')):
                rows_data = await cursor.fetchall()
                rows = [dict(row) for row in rows_data]
                row_count = len(rows)
                affected_rows = 0
            else:
                rows = []
                row_count = 0
                affected_rows = cursor.rowcount
                await conn.commit()

            execution_time_ms = int((time.time() - start_time) * 1000)
            self._update_query_stats(execution_time_ms, True)

            return QueryResult(
                rows=rows,
                row_count=row_count,
                execution_time_ms=execution_time_ms,
                affected_rows=affected_rows,
                last_insert_id=cursor.lastrowid
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self._update_query_stats(execution_time_ms, False)
            print(f"SQLite query failed: {e}")
            raise e

        finally:
            if conn:
                await self._return_connection(conn)

    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> QueryResult:
        """批量执行"""
        start_time = time.time()
        conn = None

        try:
            conn = await self._get_connection()

            cursor = await conn.executemany(query, params_list)
            await conn.commit()

            execution_time_ms = int((time.time() - start_time) * 1000)
            self._update_query_stats(execution_time_ms, True)

            return QueryResult(
                rows=[],
                row_count=0,
                execution_time_ms=execution_time_ms,
                affected_rows=cursor.rowcount,
                last_insert_id=cursor.lastrowid
            )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self._update_query_stats(execution_time_ms, False)
            print(f"SQLite batch execution failed: {e}")
            raise e

        finally:
            if conn:
                await self._return_connection(conn)

    async def begin_transaction(self, isolation_level: str = "DEFERRED") -> TransactionContext:
        """开始事务"""
        transaction_id = str(uuid.uuid4())
        conn = await self._get_connection()

        try:
            # SQLite的隔离级别
            if isolation_level.upper() == "IMMEDIATE":
                await conn.execute("BEGIN IMMEDIATE")
            elif isolation_level.upper() == "EXCLUSIVE":
                await conn.execute("BEGIN EXCLUSIVE")
            else:
                await conn.execute("BEGIN DEFERRED")

            context = TransactionContext(
                transaction_id=transaction_id,
                start_time=datetime.datetime.now(),
                isolation_level=isolation_level
            )

            self.active_transactions[transaction_id] = conn
            return context

        except Exception as e:
            await self._return_connection(conn)
            raise e

    async def commit_transaction(self, context: TransactionContext) -> bool:
        """提交事务"""
        conn = self.active_transactions.get(context.transaction_id)
        if not conn:
            return False

        try:
            await conn.commit()
            return True

        except Exception as e:
            print(f"Transaction commit failed: {e}")
            return False

        finally:
            del self.active_transactions[context.transaction_id]
            await self._return_connection(conn)

    async def rollback_transaction(self, context: TransactionContext) -> bool:
        """回滚事务"""
        conn = self.active_transactions.get(context.transaction_id)
        if not conn:
            return False

        try:
            await conn.rollback()
            return True

        except Exception as e:
            print(f"Transaction rollback failed: {e}")
            return False

        finally:
            del self.active_transactions[context.transaction_id]
            await self._return_connection(conn)

    async def upsert(self, table: str, data: Dict[str, Any],
                    conflict_columns: List[str]) -> QueryResult:
        """SQLite UPSERT (INSERT OR REPLACE)"""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{key}" for key in data.keys())

        # SQLite的UPSERT语法
        conflict_clause = ", ".join(conflict_columns)
        update_clause = ", ".join(f"{key} = excluded.{key}" for key in data.keys() if key not in conflict_columns)

        if update_clause:
            query = f"""
                INSERT INTO {table} ({columns}) VALUES ({placeholders})
                ON CONFLICT ({conflict_clause}) DO UPDATE SET {update_clause}
            """
        else:
            query = f"""
                INSERT OR REPLACE INTO {table} ({columns}) VALUES ({placeholders})
            """

        return await self.execute_query(query, data)

    async def get_table_info(self, table: str) -> Dict[str, Any]:
        """获取表信息"""
        try:
            # 获取表结构
            schema_result = await self.execute_query(f"PRAGMA table_info({table})")
            columns = [
                {
                    "name": row["name"],
                    "type": row["type"],
                    "nullable": not row["notnull"],
                    "default": row["dflt_value"],
                    "primary_key": bool(row["pk"])
                }
                for row in schema_result.rows
            ]

            # 获取索引信息
            index_result = await self.execute_query(f"PRAGMA index_list({table})")
            indexes = [
                {
                    "name": row["name"],
                    "unique": bool(row["unique"]),
                    "partial": bool(row["partial"])
                }
                for row in index_result.rows
            ]

            # 获取表统计信息
            count_result = await self.count(table)

            return {
                "table_name": table,
                "columns": columns,
                "indexes": indexes,
                "row_count": count_result,
                "column_count": len(columns),
                "primary_keys": [col["name"] for col in columns if col["primary_key"]]
            }

        except Exception as e:
            print(f"Failed to get table info for {table}: {e}")
            return {"error": str(e)}

    async def create_table(self, table: str, columns: Dict[str, str],
                          primary_key: List[str] = None,
                          indexes: List[Dict[str, Any]] = None) -> bool:
        """创建表"""
        try:
            # 构建列定义
            column_defs = []
            for col_name, col_type in columns.items():
                column_defs.append(f"{col_name} {col_type}")

            # 添加主键约束
            if primary_key:
                pk_clause = f"PRIMARY KEY ({', '.join(primary_key)})"
                column_defs.append(pk_clause)

            columns_clause = ", ".join(column_defs)
            query = f"CREATE TABLE IF NOT EXISTS {table} ({columns_clause})"

            await self.execute_query(query)

            # 创建索引
            if indexes:
                for index in indexes:
                    index_name = index.get("name", f"idx_{table}_{index['column']}")
                    column = index["column"]
                    unique = "UNIQUE" if index.get("unique", False) else ""

                    index_query = f"CREATE {unique} INDEX IF NOT EXISTS {index_name} ON {table} ({column})"
                    await self.execute_query(index_query)

            return True

        except Exception as e:
            print(f"Failed to create table {table}: {e}")
            return False

    async def drop_table(self, table: str) -> bool:
        """删除表"""
        try:
            query = f"DROP TABLE IF EXISTS {table}"
            await self.execute_query(query)
            return True

        except Exception as e:
            print(f"Failed to drop table {table}: {e}")
            return False

    async def get_database_size(self) -> Dict[str, Any]:
        """获取数据库大小信息"""
        try:
            # 获取页面大小和页面数量
            page_size_result = await self.execute_query("PRAGMA page_size")
            page_count_result = await self.execute_query("PRAGMA page_count")

            page_size = page_size_result.rows[0]["page_size"]
            page_count = page_count_result.rows[0]["page_count"]

            total_size = page_size * page_count

            # 获取各表大小
            tables_result = await self.execute_query(
                "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'"
            )

            table_sizes = {}
            for row in tables_result.rows:
                table_name = row["name"]
                count = await self.count(table_name)
                table_sizes[table_name] = count

            return {
                "total_size_bytes": total_size,
                "total_size_mb": round(total_size / (1024 * 1024), 2),
                "page_size": page_size,
                "page_count": page_count,
                "table_row_counts": table_sizes
            }

        except Exception as e:
            print(f"Failed to get database size: {e}")
            return {"error": str(e)}

    async def optimize_table(self, table: str) -> bool:
        """优化表"""
        try:
            # SQLite的ANALYZE命令
            await self.execute_query(f"ANALYZE {table}")
            return True

        except Exception as e:
            print(f"Failed to optimize table {table}: {e}")
            return False

    async def vacuum(self) -> bool:
        """执行VACUUM清理数据库"""
        try:
            await self.execute_query("VACUUM")
            return True

        except Exception as e:
            print(f"Failed to vacuum database: {e}")
            return False

    async def backup_database(self, backup_path: str) -> bool:
        """备份数据库"""
        try:
            conn = await self._get_connection()
            backup_conn = await aiosqlite.connect(backup_path)

            try:
                await conn.backup(backup_conn)
                return True

            finally:
                await backup_conn.close()
                await self._return_connection(conn)

        except Exception as e:
            print(f"Failed to backup database: {e}")
            return False

    async def get_query_plan(self, query: str, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """获取查询执行计划"""
        try:
            explain_query = f"EXPLAIN QUERY PLAN {query}"
            result = await self.execute_query(explain_query, params)
            return result.rows

        except Exception as e:
            print(f"Failed to get query plan: {e}")
            return []

    async def enable_foreign_keys(self) -> bool:
        """启用外键约束"""
        try:
            await self.execute_query("PRAGMA foreign_keys = ON")
            return True
        except Exception as e:
            print(f"Failed to enable foreign keys: {e}")
            return False

    async def get_foreign_key_violations(self) -> List[Dict[str, Any]]:
        """检查外键约束违规"""
        try:
            result = await self.execute_query("PRAGMA foreign_key_check")
            return result.rows
        except Exception as e:
            print(f"Failed to check foreign keys: {e}")
            return []