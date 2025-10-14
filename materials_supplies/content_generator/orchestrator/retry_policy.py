# ai_content_pipeline/orchestrator/retry_policy.py
import asyncio
import random
from typing import Optional, List, Callable, Awaitable
from ai_content_pipeline.core.types import RetryPolicyConfig

class RetryPolicy:
    """
    通用重试策略引擎
    支持指数退避 + 随机抖动
    """

    DEFAULT_CONFIG: RetryPolicyConfig = {
        "max_retries": 3,
        "base_delay": 1.0,
        "max_delay": 30.0,
        "backoff_factor": 2.0,
        "jitter": True,
        "retry_on": ["NetworkError", "TimeoutError", "503", "504", "ConnectionError"]
    }

    def __init__(self, config: Optional[RetryPolicyConfig] = None):
        merged = self.DEFAULT_CONFIG.copy()
        if config:
            merged.update({k: v for k, v in config.items() if v is not None})
        self.config = merged

    async def execute(
        self,
        func: Callable[[], Awaitable],
        is_retryable: Optional[Callable[[Exception], bool]] = None
    ) -> Awaitable:
        """
        执行带重试的异步函数

        Args:
            func: 异步函数
            is_retryable: 自定义判断是否可重试的函数

        Returns:
            函数执行结果
        """
        last_exception = None

        for attempt in range(self.config["max_retries"] + 1):
            try:
                return await func()
            except Exception as e:
                last_exception = e

                if attempt == self.config["max_retries"]:
                    break  # 最后一次尝试，直接抛出

                if not self._is_error_retryable(e, is_retryable):
                    break  # 不可重试，直接抛出

                delay = self._calculate_delay(attempt)
                await asyncio.sleep(delay)

        raise last_exception

    def _is_error_retryable(
        self,
        error: Exception,
        custom_check: Optional[Callable[[Exception], bool]]
    ) -> bool:
        """
        判断错误是否可重试
        """
        if custom_check and custom_check(error):
            return True

        error_str = str(type(error).__name__)
        retry_on = self.config["retry_on"]

        return any(keyword in error_str for keyword in retry_on)

    def _calculate_delay(self, attempt: int) -> float:
        """
        计算退避延迟时间
        公式：min(max_delay, base_delay * (backoff_factor ^ attempt))
        """
        exponential = self.config["base_delay"] * (
            self.config["backoff_factor"] ** attempt
        )
        delay = min(exponential, self.config["max_delay"])

        if self.config["jitter"]:
            delay = delay * (0.5 + random.random() * 0.5)  # 0.5~1.0 倍随机抖动

        return delay


# -----------------------------
# 预设策略工厂
# -----------------------------

def simple_retry(max_retries: int = 3) -> RetryPolicy:
    return RetryPolicy({"max_retries": max_retries, "jitter": False})


def network_friendly_retry() -> RetryPolicy:
    return RetryPolicy({
        "max_retries": 5,
        "base_delay": 1.0,
        "max_delay": 60.0,
        "backoff_factor": 2.0,
        "jitter": True,
        "retry_on": ["TimeoutError", "ConnectionError", "NetworkError", "503", "504"]
    })


def aggressive_retry() -> RetryPolicy:
    return RetryPolicy({
        "max_retries": 2,
        "base_delay": 0.5,
        "max_delay": 10.0,
        "backoff_factor": 1.5,
        "jitter": True
    })