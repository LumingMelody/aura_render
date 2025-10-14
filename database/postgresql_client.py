"""
PostgreSQL 数据库客户端
"""
from typing import Dict, List, Any, Optional, Union
import asyncio
import asyncpg
import time
import uuid
import datetime
import json
from contextlib import asynccontextmanager

from .base_database import (
    BaseDatabaseClient,
    DatabaseConfig,
    QueryResult,
    TransactionContext,
    ConnectionStatus
)


class PostgreSQLClient(BaseDatabaseClient):
    """PostgreSQL 数据库客户端"""

    def __init__(self, config: DatabaseConfig):
        super().__init__(config)
        self.connection_pool = None
        self.active_transactions = {}

    async def connect(self) -> bool:
        """建立PostgreSQL连接池"""
        try:
            self.status = ConnectionStatus.CONNECTING

            # 构建连接字符串
            dsn = f"postgresql://{self.config.username}:{self.config.password}@{self.config.host}:{self.config.port}/{self.config.database}"

            # 创建连接池
            self.connection_pool = await asyncpg.create_pool(
                dsn,
                min_size=1,
                max_size=self.config.pool_size,
                max_queries=50000,
                max_inactive_connection_lifetime=300.0,
                timeout=self.config.pool_timeout,
                command_timeout=60,
                server_settings={
                    'jit': 'off',  # 关闭JIT以提高连接速度
                    'application_name': 'aura_render'
                }
            )

            self.status = ConnectionStatus.CONNECTED
            self.connection_count = self.config.pool_size
            print(f"✅ PostgreSQL connected: {self.config.host}:{self.config.port}/{self.config.database}")
            return True

        except Exception as e:
            self.status = ConnectionStatus.ERROR
            print(f"❌ PostgreSQL connection failed: {e}")
            return False

    async def disconnect(self) -> bool:
        """关闭PostgreSQL连接池"""
        try:
            if self.connection_pool:
                await self.connection_pool.close()
                self.connection_pool = None

            self.status = ConnectionStatus.DISCONNECTED
            self.connection_count = 0
            print("✅ PostgreSQL disconnected")
            return True

        except Exception as e:
            print(f"❌ PostgreSQL disconnect failed: {e}")
            return False

    async def execute_query(self, query: str, params: Optional[Dict[str, Any]] = None) -> QueryResult:
        """执行查询"""
        start_time = time.time()

        try:
            async with self.connection_pool.acquire() as conn:
                # 转换参数格式（从字典到位置参数）
                if params:
                    query_args, converted_query = self._convert_params(query, params)
                else:
                    query_args = []
                    converted_query = query

                if converted_query.strip().upper().startswith(('SELECT', 'WITH')):
                    # 查询操作
                    rows_data = await conn.fetch(converted_query, *query_args)
                    rows = [dict(row) for row in rows_data]
                    row_count = len(rows)
                    affected_rows = 0
                    last_insert_id = None

                elif converted_query.strip().upper().startswith('INSERT'):
                    # 插入操作，可能需要返回ID
                    if 'RETURNING' in converted_query.upper():
                        rows_data = await conn.fetch(converted_query, *query_args)
                        rows = [dict(row) for row in rows_data]
                        row_count = len(rows)
                        affected_rows = row_count
                        last_insert_id = rows[0].get('id') if rows else None
                    else:
                        result = await conn.execute(converted_query, *query_args)
                        rows = []
                        row_count = 0
                        affected_rows = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
                        last_insert_id = None

                else:
                    # 其他操作（UPDATE, DELETE等）
                    result = await conn.execute(converted_query, *query_args)
                    rows = []
                    row_count = 0
                    affected_rows = int(result.split()[-1]) if result.split()[-1].isdigit() else 0
                    last_insert_id = None

                execution_time_ms = int((time.time() - start_time) * 1000)
                self._update_query_stats(execution_time_ms, True)

                return QueryResult(
                    rows=rows,
                    row_count=row_count,
                    execution_time_ms=execution_time_ms,
                    affected_rows=affected_rows,
                    last_insert_id=last_insert_id
                )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self._update_query_stats(execution_time_ms, False)
            print(f"PostgreSQL query failed: {e}")
            raise e

    async def execute_many(self, query: str, params_list: List[Dict[str, Any]]) -> QueryResult:
        """批量执行"""
        start_time = time.time()

        try:
            async with self.connection_pool.acquire() as conn:
                # 转换所有参数
                converted_params = []
                converted_query = query

                for params in params_list:
                    query_args, converted_query = self._convert_params(query, params)
                    converted_params.append(query_args)

                # 使用executemany
                await conn.executemany(converted_query, converted_params)

                execution_time_ms = int((time.time() - start_time) * 1000)
                self._update_query_stats(execution_time_ms, True)

                return QueryResult(
                    rows=[],
                    row_count=0,
                    execution_time_ms=execution_time_ms,
                    affected_rows=len(params_list)
                )

        except Exception as e:
            execution_time_ms = int((time.time() - start_time) * 1000)
            self._update_query_stats(execution_time_ms, False)
            print(f"PostgreSQL batch execution failed: {e}")
            raise e

    async def begin_transaction(self, isolation_level: str = "READ_COMMITTED") -> TransactionContext:
        """开始事务"""
        transaction_id = str(uuid.uuid4())

        try:
            conn = await self.connection_pool.acquire()

            # 设置隔离级别
            isolation_mapping = {
                "READ_UNCOMMITTED": "READ UNCOMMITTED",
                "READ_COMMITTED": "READ COMMITTED",
                "REPEATABLE_READ": "REPEATABLE READ",
                "SERIALIZABLE": "SERIALIZABLE"
            }

            isolation = isolation_mapping.get(isolation_level.upper(), "READ COMMITTED")

            transaction = conn.transaction(isolation=isolation)
            await transaction.start()

            context = TransactionContext(
                transaction_id=transaction_id,
                start_time=datetime.datetime.now(),
                isolation_level=isolation_level
            )

            self.active_transactions[transaction_id] = {
                "connection": conn,
                "transaction": transaction
            }

            return context

        except Exception as e:
            if 'conn' in locals():
                await self.connection_pool.release(conn)
            raise e

    async def commit_transaction(self, context: TransactionContext) -> bool:
        """提交事务"""
        transaction_info = self.active_transactions.get(context.transaction_id)
        if not transaction_info:
            return False

        try:
            await transaction_info["transaction"].commit()
            return True

        except Exception as e:
            print(f"Transaction commit failed: {e}")
            return False

        finally:
            await self.connection_pool.release(transaction_info["connection"])
            del self.active_transactions[context.transaction_id]

    async def rollback_transaction(self, context: TransactionContext) -> bool:
        """回滚事务"""
        transaction_info = self.active_transactions.get(context.transaction_id)
        if not transaction_info:
            return False

        try:
            await transaction_info["transaction"].rollback()
            return True

        except Exception as e:
            print(f"Transaction rollback failed: {e}")
            return False

        finally:
            await self.connection_pool.release(transaction_info["connection"])
            del self.active_transactions[context.transaction_id]

    def _convert_params(self, query: str, params: Dict[str, Any]) -> tuple:
        """将命名参数转换为位置参数"""
        import re

        # 找到所有命名参数
        param_pattern = r':(\w+)'
        matches = re.findall(param_pattern, query)

        # 构建参数列表
        query_args = []
        converted_query = query

        for i, param_name in enumerate(matches, 1):
            if param_name in params:
                query_args.append(params[param_name])
                converted_query = converted_query.replace(f':{param_name}', f'${i}', 1)

        return query_args, converted_query

    async def upsert(self, table: str, data: Dict[str, Any],
                    conflict_columns: List[str]) -> QueryResult:
        """PostgreSQL UPSERT (ON CONFLICT)"""
        columns = ", ".join(data.keys())
        placeholders = ", ".join(f":{key}" for key in data.keys())

        conflict_clause = ", ".join(conflict_columns)
        update_clause = ", ".join(f"{key} = EXCLUDED.{key}" for key in data.keys() if key not in conflict_columns)

        if update_clause:
            query = f"""
                INSERT INTO {table} ({columns}) VALUES ({placeholders})
                ON CONFLICT ({conflict_clause}) DO UPDATE SET {update_clause}
                RETURNING *
            """
        else:
            query = f"""
                INSERT INTO {table} ({columns}) VALUES ({placeholders})
                ON CONFLICT ({conflict_clause}) DO NOTHING
                RETURNING *
            """

        return await self.execute_query(query, data)

    async def get_table_info(self, table: str) -> Dict[str, Any]:
        """获取表信息"""
        try:
            # 获取列信息
            columns_query = """
                SELECT
                    column_name,
                    data_type,
                    is_nullable,
                    column_default,
                    character_maximum_length,
                    numeric_precision,
                    numeric_scale
                FROM information_schema.columns
                WHERE table_name = :table_name
                ORDER BY ordinal_position
            """

            columns_result = await self.execute_query(columns_query, {"table_name": table})

            # 获取主键信息
            pk_query = """
                SELECT column_name
                FROM information_schema.table_constraints tc
                JOIN information_schema.key_column_usage kcu
                    ON tc.constraint_name = kcu.constraint_name
                WHERE tc.table_name = :table_name
                    AND tc.constraint_type = 'PRIMARY KEY'
            """

            pk_result = await self.execute_query(pk_query, {"table_name": table})
            primary_keys = [row["column_name"] for row in pk_result.rows]

            # 获取索引信息
            indexes_query = """
                SELECT
                    indexname as name,
                    indexdef as definition
                FROM pg_indexes
                WHERE tablename = :table_name
            """

            indexes_result = await self.execute_query(indexes_query, {"table_name": table})

            # 获取表统计信息
            stats_query = """
                SELECT
                    n_tup_ins as inserts,
                    n_tup_upd as updates,
                    n_tup_del as deletes,
                    n_live_tup as live_tuples,
                    n_dead_tup as dead_tuples
                FROM pg_stat_user_tables
                WHERE relname = :table_name
            """

            stats_result = await self.execute_query(stats_query, {"table_name": table})
            stats = stats_result.rows[0] if stats_result.rows else {}

            return {
                "table_name": table,
                "columns": [
                    {
                        "name": row["column_name"],
                        "type": row["data_type"],
                        "nullable": row["is_nullable"] == "YES",
                        "default": row["column_default"],
                        "max_length": row["character_maximum_length"],
                        "precision": row["numeric_precision"],
                        "scale": row["numeric_scale"]
                    }
                    for row in columns_result.rows
                ],
                "primary_keys": primary_keys,
                "indexes": [
                    {
                        "name": row["name"],
                        "definition": row["definition"]
                    }
                    for row in indexes_result.rows
                ],
                "statistics": stats
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
                    method = index.get("method", "btree")

                    index_query = f"CREATE {unique} INDEX IF NOT EXISTS {index_name} ON {table} USING {method} ({column})"
                    await self.execute_query(index_query)

            return True

        except Exception as e:
            print(f"Failed to create table {table}: {e}")
            return False

    async def analyze_table(self, table: str) -> Dict[str, Any]:
        """分析表"""
        try:
            # 执行ANALYZE
            await self.execute_query(f"ANALYZE {table}")

            # 获取表统计信息
            stats_query = """
                SELECT
                    schemaname,
                    tablename,
                    attname,
                    n_distinct,
                    correlation
                FROM pg_stats
                WHERE tablename = :table_name
            """

            result = await self.execute_query(stats_query, {"table_name": table})

            return {
                "analyzed": True,
                "statistics": result.rows
            }

        except Exception as e:
            print(f"Failed to analyze table {table}: {e}")
            return {"error": str(e)}

    async def vacuum(self, table: str = None, analyze: bool = True) -> bool:
        """执行VACUUM"""
        try:
            if table:
                query = f"VACUUM {'ANALYZE' if analyze else ''} {table}"
            else:
                query = f"VACUUM {'ANALYZE' if analyze else ''}"

            await self.execute_query(query)
            return True

        except Exception as e:
            print(f"Failed to vacuum: {e}")
            return False

    async def get_database_size(self) -> Dict[str, Any]:
        """获取数据库大小信息"""
        try:
            # 数据库总大小
            db_size_query = """
                SELECT pg_size_pretty(pg_database_size(current_database())) as size,
                       pg_database_size(current_database()) as size_bytes
            """

            db_size_result = await self.execute_query(db_size_query)

            # 各表大小
            table_sizes_query = """
                SELECT
                    tablename,
                    pg_size_pretty(pg_total_relation_size(tablename::regclass)) as size,
                    pg_total_relation_size(tablename::regclass) as size_bytes
                FROM pg_tables
                WHERE schemaname = 'public'
                ORDER BY pg_total_relation_size(tablename::regclass) DESC
            """

            table_sizes_result = await self.execute_query(table_sizes_query)

            return {
                "database_size": db_size_result.rows[0] if db_size_result.rows else {},
                "table_sizes": table_sizes_result.rows
            }

        except Exception as e:
            print(f"Failed to get database size: {e}")
            return {"error": str(e)}

    async def get_connection_info(self) -> Dict[str, Any]:
        """获取连接信息"""
        base_info = await super().get_connection_info()

        try:
            # PostgreSQL特定信息
            pg_info_query = """
                SELECT
                    version() as version,
                    current_database() as database,
                    current_user as user,
                    inet_server_addr() as server_addr,
                    inet_server_port() as server_port
            """

            result = await self.execute_query(pg_info_query)
            pg_info = result.rows[0] if result.rows else {}

            base_info.update({
                "postgresql_info": pg_info,
                "pool_info": {
                    "size": self.connection_pool.get_size() if self.connection_pool else 0,
                    "min_size": self.connection_pool.get_min_size() if self.connection_pool else 0,
                    "max_size": self.connection_pool.get_max_size() if self.connection_pool else 0
                }
            })

        except Exception as e:
            base_info["error"] = str(e)

        return base_info

    async def explain_query(self, query: str, params: Optional[Dict[str, Any]] = None,
                           analyze: bool = False, buffers: bool = False) -> List[Dict[str, Any]]:
        """获取查询执行计划"""
        try:
            options = []
            if analyze:
                options.append("ANALYZE")
            if buffers:
                options.append("BUFFERS")

            options_str = f"({', '.join(options)})" if options else ""
            explain_query = f"EXPLAIN {options_str} {query}"

            result = await self.execute_query(explain_query, params)
            return result.rows

        except Exception as e:
            print(f"Failed to explain query: {e}")
            return []