#!/usr/bin/env python3
"""
任务状态查询示例脚本
演示如何提交任务并轮询状态直到完成
"""

import requests
import time
from typing import Optional, Dict, Any


class TaskStatusChecker:
    """任务状态检查器"""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.session = requests.Session()

    def submit_task(self, task_data: Dict[str, Any]) -> str:
        """提交视频生成任务"""
        url = f"{self.base_url}/tasks/video/async"
        response = self.session.post(url, json=task_data)
        response.raise_for_status()

        result = response.json()
        print(f"✅ 任务已提交")
        print(f"   任务ID: {result['task_id']}")
        print(f"   优先级: {result['priority']}")
        print(f"   预估时长: {result.get('estimated_duration', 'N/A')} 秒")

        return result['task_id']

    def get_task_status(self, task_id: str) -> Dict[str, Any]:
        """获取任务状态"""
        url = f"{self.base_url}/tasks/status/{task_id}"
        response = self.session.get(url)
        response.raise_for_status()
        return response.json()

    def wait_for_completion(
        self,
        task_id: str,
        poll_interval: int = 5,
        timeout: int = 3600
    ) -> Optional[Dict[str, Any]]:
        """
        等待任务完成

        Args:
            task_id: 任务ID
            poll_interval: 轮询间隔(秒)
            timeout: 超时时间(秒)

        Returns:
            任务结果，如果超时则返回None
        """
        start_time = time.time()
        last_progress = -1

        print(f"\n⏳ 等待任务完成 (任务ID: {task_id})")
        print("=" * 60)

        while True:
            # 检查超时
            elapsed = time.time() - start_time
            if elapsed > timeout:
                print(f"\n❌ 超时: 任务执行超过 {timeout} 秒")
                return None

            try:
                # 获取状态
                status = self.get_task_status(task_id)

                # 显示进度（只在进度变化时）
                current_progress = status.get('progress', 0)
                if current_progress != last_progress:
                    progress_bar = self._create_progress_bar(current_progress)
                    print(f"\r{progress_bar} {current_progress:.1f}% | {status.get('message', '')}", end='', flush=True)
                    last_progress = current_progress

                # 检查任务状态
                task_status = status['status']

                if task_status == 'completed':
                    print(f"\n\n✅ 任务完成!")
                    print(f"   实际耗时: {status.get('actual_duration', 'N/A')} 秒")
                    return status.get('result')

                elif task_status == 'failed':
                    print(f"\n\n❌ 任务失败!")
                    print(f"   错误信息: {status.get('error', 'Unknown error')}")
                    return None

                elif task_status == 'cancelled':
                    print(f"\n\n⚠️  任务已被取消")
                    return None

                # 等待下一次轮询
                time.sleep(poll_interval)

            except requests.exceptions.RequestException as e:
                print(f"\n\n❌ 网络错误: {e}")
                return None

    def _create_progress_bar(self, progress: float, width: int = 40) -> str:
        """创建进度条"""
        filled = int(width * progress / 100)
        bar = '█' * filled + '░' * (width - filled)
        return f"[{bar}]"

    def cancel_task(self, task_id: str) -> bool:
        """取消任务"""
        url = f"{self.base_url}/tasks/cancel/{task_id}"
        try:
            response = self.session.delete(url)
            response.raise_for_status()
            print(f"✅ 任务 {task_id} 已取消")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ 取消任务失败: {e}")
            return False

    def get_task_history(self, limit: int = 10, status: Optional[str] = None) -> list:
        """获取任务历史"""
        url = f"{self.base_url}/tasks/history?limit={limit}"
        if status:
            url += f"&status={status}"

        response = self.session.get(url)
        response.raise_for_status()
        return response.json()


def main():
    """主函数：演示完整的任务提交和状态查询流程"""

    # 初始化检查器
    checker = TaskStatusChecker()

    # 准备任务数据
    task_data = {
        "theme_id": "产品宣传",
        "keywords_id": ["AI", "创新", "科技"],
        "target_duration_id": 60,
        "user_description_id": "一个展示AI技术的60秒宣传视频",
        "priority": "high",
        "config": {
            "quality": "high",
            "format": "mp4",
            "resolution": "1920x1080"
        }
    }

    try:
        # 1. 提交任务
        task_id = checker.submit_task(task_data)

        # 2. 等待任务完成
        result = checker.wait_for_completion(
            task_id=task_id,
            poll_interval=3,  # 每3秒查询一次
            timeout=1800      # 30分钟超时
        )

        # 3. 处理结果
        if result:
            print("\n📊 任务结果:")
            print(f"   输出路径: {result.get('output_path', 'N/A')}")
            if 'metadata' in result:
                print(f"   元数据: {result['metadata']}")
        else:
            print("\n⚠️  任务未成功完成")

        # 4. 查看最近的任务历史
        print("\n\n📜 最近完成的任务:")
        print("=" * 60)
        history = checker.get_task_history(limit=5, status="completed")
        for i, task in enumerate(history, 1):
            print(f"{i}. [{task['task_id']}] {task['status']} - {task['message']}")

    except KeyboardInterrupt:
        print("\n\n⚠️  用户中断")
        print(f"您可以稍后使用以下命令查询任务状态:")
        print(f"curl http://localhost:8000/tasks/status/{task_id}")

    except Exception as e:
        print(f"\n\n❌ 发生错误: {e}")


if __name__ == "__main__":
    main()
