"""
Database Performance Optimizer
数据库性能优化器 - 提供查询优化、连接池管理和性能监控
"""
import sqlite3
import asyncio
import aiosqlite
import time
import threading
from typing import Dict, List, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from contextlib import asynccontextmanager, contextmanager
from concurrent.futures import ThreadPoolExecutor
import logging
import json

from cache.redis_cache_manager import get_cache_manager, cache_result


@dataclass
class QueryStats:
    """查询统计"""
    query_hash: str
    sql: str
    execution_count: int = 0
    total_time: float = 0.0
    avg_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    last_executed: datetime = field(default_factory=datetime.now)
    rows_affected: int = 0

    def add_execution(self, execution_time: float, rows: int = 0):
        """添加执行记录"""
        self.execution_count += 1
        self.total_time += execution_time
        self.avg_time = self.total_time / self.execution_count
        self.min_time = min(self.min_time, execution_time)
        self.max_time = max(self.max_time, execution_time)
        self.last_executed = datetime.now()
        self.rows_affected += rows


@dataclass
class ConnectionPoolStats:
    """连接池统计"""
    total_connections: int = 0
    active_connections: int = 0
    idle_connections: int = 0
    peak_connections: int = 0
    connection_errors: int = 0
    total_queries: int = 0
    avg_query_time: float = 0.0


class SQLiteOptimizer:
    """SQLite性能优化器"""

    def __init__(self, db_path: str, max_connections: int = 10):
        self.db_path = db_path
        self.max_connections = max_connections
        self.query_stats: Dict[str, QueryStats] = {}
        self.connection_stats = ConnectionPoolStats()
        self.logger = logging.getLogger(__name__)

        # 连接池
        self._connection_pool = []
        self._pool_lock = threading.Lock()
        self._active_connections = set()

        # 性能优化设置
        self.enable_query_cache = True
        self.cache_ttl = 300  # 5分钟
        self.slow_query_threshold = 1.0  # 1秒

        # 初始化优化
        self._init_database_optimizations()

    def _init_database_optimizations(self):
        """初始化数据库优化设置"""
        optimization_queries = [
            # 启用WAL模式以提高并发性能
            "PRAGMA journal_mode = WAL;",

            # 优化同步设置
            "PRAGMA synchronous = NORMAL;",

            # 增加缓存大小（以KB为单位）
            "PRAGMA cache_size = -64000;",  # 64MB

            # 启用内存映射
            "PRAGMA mmap_size = 134217728;",  # 128MB

            # 优化临时存储
            "PRAGMA temp_store = MEMORY;",

            # 设置页面大小
            "PRAGMA page_size = 4096;",

            # 启用外键约束
            "PRAGMA foreign_keys = ON;",

            # 分析统计信息
            "PRAGMA optimize;",
        ]

        try:
            with sqlite3.connect(self.db_path) as conn:
                for query in optimization_queries:
                    conn.execute(query)
                conn.commit()

            self.logger.info("Database optimizations applied successfully")
        except Exception as e:
            self.logger.error(f"Failed to apply database optimizations: {e}")

    @contextmanager
    def get_connection(self):
        """获取数据库连接（同步版本）"""
        conn = None
        try:
            with self._pool_lock:
                if self._connection_pool:
                    conn = self._connection_pool.pop()
                else:
                    conn = sqlite3.connect(
                        self.db_path,
                        check_same_thread=False,
                        timeout=10.0
                    )
                    # 为每个连接应用优化设置
                    conn.execute("PRAGMA journal_mode = WAL;")
                    conn.execute("PRAGMA synchronous = NORMAL;")
                    conn.execute("PRAGMA cache_size = -8000;")  # 8MB per connection
                    conn.execute("PRAGMA foreign_keys = ON;")

                self._active_connections.add(conn)
                self.connection_stats.active_connections += 1
                self.connection_stats.total_connections = max(
                    self.connection_stats.total_connections,
                    len(self._active_connections)
                )
                self.connection_stats.peak_connections = max(
                    self.connection_stats.peak_connections,
                    self.connection_stats.active_connections
                )

            yield conn

        except Exception as e:
            self.connection_stats.connection_errors += 1
            self.logger.error(f"Database connection error: {e}")
            raise
        finally:
            if conn:
                with self._pool_lock:
                    self._active_connections.discard(conn)
                    self.connection_stats.active_connections -= 1

                    if len(self._connection_pool) < self.max_connections:
                        self._connection_pool.append(conn)
                    else:
                        conn.close()

    async def execute_query(self, sql: str, params: Tuple = (),
                           fetch: str = "none") -> Any:
        """执行查询（异步版本）"""
        query_hash = hash(sql)
        start_time = time.time()

        try:
            # 检查缓存（对于SELECT查询）
            if self.enable_query_cache and sql.strip().upper().startswith('SELECT'):
                cache = get_cache_manager()
                cache_key = f"query:{query_hash}:{hash(params)}"
                cached_result = await cache.get(cache_key)
                if cached_result is not None:
                    return cached_result

            # 执行查询
            result = await self._execute_with_stats(sql, params, fetch, query_hash)

            # 缓存结果（对于SELECT查询）
            if (self.enable_query_cache and
                sql.strip().upper().startswith('SELECT') and
                result is not None):
                cache_key = f"query:{query_hash}:{hash(params)}"
                await cache.set(cache_key, result, self.cache_ttl)

            return result

        except Exception as e:
            self.logger.error(f"Query execution failed: {e}\nSQL: {sql}")
            raise
        finally:
            execution_time = time.time() - start_time
            self._record_query_stats(query_hash, sql, execution_time)

    async def _execute_with_stats(self, sql: str, params: Tuple,
                                 fetch: str, query_hash: str) -> Any:
        """执行查询并收集统计信息"""
        loop = asyncio.get_event_loop()

        def sync_execute():
            with self.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql, params)

                if fetch == "all":
                    return cursor.fetchall()
                elif fetch == "one":
                    return cursor.fetchone()
                elif fetch == "many":
                    return cursor.fetchmany()
                else:
                    conn.commit()
                    return cursor.rowcount

        result = await loop.run_in_executor(None, sync_execute)

        self.connection_stats.total_queries += 1
        return result

    def _record_query_stats(self, query_hash: str, sql: str, execution_time: float):
        """记录查询统计信息"""
        if query_hash not in self.query_stats:
            self.query_stats[query_hash] = QueryStats(
                query_hash=str(query_hash),
                sql=sql[:200] + "..." if len(sql) > 200 else sql
            )

        stats = self.query_stats[query_hash]
        stats.add_execution(execution_time)

        # 记录慢查询
        if execution_time > self.slow_query_threshold:
            self.logger.warning(
                f"Slow query detected: {execution_time:.3f}s\n"
                f"SQL: {sql[:100]}..."
            )

        # 更新连接池统计
        if self.connection_stats.total_queries > 0:
            total_time = sum(stat.total_time for stat in self.query_stats.values())
            self.connection_stats.avg_query_time = (
                total_time / self.connection_stats.total_queries
            )

    async def execute_batch(self, sql: str, params_list: List[Tuple]) -> int:
        """批量执行查询"""
        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()

            def sync_batch_execute():
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    cursor.executemany(sql, params_list)
                    conn.commit()
                    return cursor.rowcount

            result = await loop.run_in_executor(None, sync_batch_execute)

            execution_time = time.time() - start_time
            query_hash = hash(sql)
            self._record_query_stats(query_hash, f"BATCH: {sql}", execution_time)

            return result

        except Exception as e:
            self.logger.error(f"Batch execution failed: {e}\nSQL: {sql}")
            raise

    def create_index(self, table: str, columns: List[str],
                    unique: bool = False, if_not_exists: bool = True):
        """创建索引"""
        index_name = f"idx_{table}_{'_'.join(columns)}"
        unique_clause = "UNIQUE " if unique else ""
        if_not_exists_clause = "IF NOT EXISTS " if if_not_exists else ""

        sql = (f"CREATE {unique_clause}INDEX {if_not_exists_clause}"
               f"{index_name} ON {table} ({', '.join(columns)})")

        try:
            with self.get_connection() as conn:
                conn.execute(sql)
                conn.commit()
                self.logger.info(f"Index created: {index_name}")
        except Exception as e:
            self.logger.error(f"Failed to create index {index_name}: {e}")

    def analyze_table(self, table: str):
        """分析表统计信息"""
        try:
            with self.get_connection() as conn:
                conn.execute(f"ANALYZE {table}")
                conn.commit()
                self.logger.info(f"Table analyzed: {table}")
        except Exception as e:
            self.logger.error(f"Failed to analyze table {table}: {e}")

    async def vacuum_database(self):
        """清理数据库"""
        start_time = time.time()

        try:
            loop = asyncio.get_event_loop()

            def sync_vacuum():
                with self.get_connection() as conn:
                    conn.execute("VACUUM")
                    conn.commit()

            await loop.run_in_executor(None, sync_vacuum)

            execution_time = time.time() - start_time
            self.logger.info(f"Database vacuumed in {execution_time:.2f}s")

        except Exception as e:
            self.logger.error(f"Vacuum failed: {e}")

    def get_query_performance_report(self) -> Dict[str, Any]:
        """获取查询性能报告"""
        # 按执行次数排序的热门查询
        hot_queries = sorted(
            self.query_stats.values(),
            key=lambda x: x.execution_count,
            reverse=True
        )[:10]

        # 按平均执行时间排序的慢查询
        slow_queries = sorted(
            self.query_stats.values(),
            key=lambda x: x.avg_time,
            reverse=True
        )[:10]

        # 按总执行时间排序的耗时查询
        time_consuming_queries = sorted(
            self.query_stats.values(),
            key=lambda x: x.total_time,
            reverse=True
        )[:10]

        return {
            "connection_pool": {
                "total_connections": self.connection_stats.total_connections,
                "active_connections": self.connection_stats.active_connections,
                "peak_connections": self.connection_stats.peak_connections,
                "connection_errors": self.connection_stats.connection_errors,
                "total_queries": self.connection_stats.total_queries,
                "avg_query_time": self.connection_stats.avg_query_time
            },
            "hot_queries": [
                {
                    "sql": q.sql,
                    "execution_count": q.execution_count,
                    "avg_time": q.avg_time,
                    "total_time": q.total_time
                } for q in hot_queries
            ],
            "slow_queries": [
                {
                    "sql": q.sql,
                    "avg_time": q.avg_time,
                    "max_time": q.max_time,
                    "execution_count": q.execution_count
                } for q in slow_queries
            ],
            "time_consuming_queries": [
                {
                    "sql": q.sql,
                    "total_time": q.total_time,
                    "execution_count": q.execution_count,
                    "avg_time": q.avg_time
                } for q in time_consuming_queries
            ],
            "cache_enabled": self.enable_query_cache,
            "cache_ttl": self.cache_ttl
        }

    def optimize_materials_database(self):
        """优化素材数据库的特定设置"""
        optimizations = [
            # 为素材表创建常用索引
            ("materials", ["media_type"], False),
            ("materials", ["created_at"], False),
            ("materials", ["file_size"], False),
            ("materials", ["material_id"], True),  # 唯一索引

            # 分析表统计信息
            "materials"
        ]

        try:
            for optimization in optimizations:
                if isinstance(optimization, tuple):
                    table, columns, unique = optimization
                    self.create_index(table, columns, unique)
                else:
                    self.analyze_table(optimization)

            self.logger.info("Materials database optimizations completed")

        except Exception as e:
            self.logger.error(f"Materials database optimization failed: {e}")

    async def get_database_info(self) -> Dict[str, Any]:
        """获取数据库信息"""
        info_queries = [
            ("database_size", "SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()"),
            ("table_count", "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"),
            ("index_count", "SELECT COUNT(*) FROM sqlite_master WHERE type='index'"),
            ("journal_mode", "PRAGMA journal_mode"),
            ("cache_size", "PRAGMA cache_size"),
            ("synchronous", "PRAGMA synchronous"),
        ]

        info = {}
        for name, sql in info_queries:
            try:
                result = await self.execute_query(sql, fetch="one")
                info[name] = result[0] if result else None
            except Exception as e:
                info[name] = f"Error: {e}"

        return info

    def close(self):
        """关闭所有连接"""
        with self._pool_lock:
            for conn in self._connection_pool:
                conn.close()
            for conn in self._active_connections.copy():
                conn.close()

            self._connection_pool.clear()
            self._active_connections.clear()


# 全局优化器实例
_global_db_optimizer: Optional[SQLiteOptimizer] = None


def get_db_optimizer(db_path: str = None) -> SQLiteOptimizer:
    """获取全局数据库优化器实例"""
    global _global_db_optimizer
    if _global_db_optimizer is None and db_path:
        _global_db_optimizer = SQLiteOptimizer(db_path)
    return _global_db_optimizer


def optimize_materials_db(db_path: str) -> SQLiteOptimizer:
    """优化素材数据库"""
    optimizer = SQLiteOptimizer(db_path)
    optimizer.optimize_materials_database()
    return optimizer


# 使用示例和测试
async def test_database_optimizer():
    """测试数据库优化器"""
    print("🧪 测试数据库性能优化器")
    print("=" * 50)

    # 创建测试数据库
    test_db_path = "/tmp/test_optimizer.db"

    # 初始化优化器
    optimizer = SQLiteOptimizer(test_db_path)

    try:
        # 创建测试表
        await optimizer.execute_query("""
            CREATE TABLE IF NOT EXISTS test_materials (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL,
                size INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        print("✅ 测试表创建成功")

        # 插入测试数据
        test_data = [
            (f"material_{i}", "video" if i % 2 == 0 else "audio", i * 1000)
            for i in range(100)
        ]

        await optimizer.execute_batch(
            "INSERT INTO test_materials (name, type, size) VALUES (?, ?, ?)",
            test_data
        )

        print(f"✅ 插入 {len(test_data)} 条测试数据")

        # 创建索引
        optimizer.create_index("test_materials", ["type"])
        optimizer.create_index("test_materials", ["size"])

        print("✅ 创建索引完成")

        # 执行一些测试查询
        queries = [
            ("SELECT COUNT(*) FROM test_materials", "one"),
            ("SELECT * FROM test_materials WHERE type = ?", "all", ("video",)),
            ("SELECT AVG(size) FROM test_materials", "one"),
            ("SELECT * FROM test_materials ORDER BY size DESC LIMIT 10", "all"),
        ]

        for i, query_info in enumerate(queries):
            if len(query_info) == 3:
                sql, fetch, params = query_info
            else:
                sql, fetch = query_info
                params = ()

            result = await optimizer.execute_query(sql, params, fetch)
            print(f"  查询 {i+1}: 返回 {len(result) if isinstance(result, list) else 1} 行")

        # 分析表统计
        optimizer.analyze_table("test_materials")

        # 获取性能报告
        performance_report = optimizer.get_query_performance_report()

        print(f"\n📊 性能统计:")
        print(f"  总查询数: {performance_report['connection_pool']['total_queries']}")
        print(f"  平均查询时间: {performance_report['connection_pool']['avg_query_time']:.4f}s")
        print(f"  峰值连接数: {performance_report['connection_pool']['peak_connections']}")
        print(f"  热门查询数: {len(performance_report['hot_queries'])}")

        # 获取数据库信息
        db_info = await optimizer.get_database_info()
        print(f"\n🔧 数据库信息:")
        for key, value in db_info.items():
            print(f"  {key}: {value}")

        print("\n✅ 数据库优化器测试完成")

    finally:
        optimizer.close()

        # 清理测试数据库
        import os
        try:
            os.remove(test_db_path)
            # 清理WAL和SHM文件
            for suffix in ['-wal', '-shm']:
                try:
                    os.remove(test_db_path + suffix)
                except FileNotFoundError:
                    pass
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    asyncio.run(test_database_optimizer())