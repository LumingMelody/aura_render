"""
分析面板 - 数据可视化和业务分析
"""
from typing import Dict, List, Any, Optional, Tuple
import asyncio
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import json
import statistics
from enum import Enum

from .performance_monitor import PerformanceMonitor, Metric
from database.database_manager import DatabaseManager
from cache.cache_manager import CacheManager


class ChartType(Enum):
    """图表类型"""
    LINE = "line"
    BAR = "bar"
    PIE = "pie"
    AREA = "area"
    SCATTER = "scatter"
    HEATMAP = "heatmap"


class TimeRange(Enum):
    """时间范围"""
    LAST_HOUR = "1h"
    LAST_6_HOURS = "6h"
    LAST_24_HOURS = "24h"
    LAST_7_DAYS = "7d"
    LAST_30_DAYS = "30d"


@dataclass
class ChartData:
    """图表数据"""
    title: str
    chart_type: ChartType
    data: List[Dict[str, Any]]
    labels: List[str]
    datasets: List[Dict[str, Any]]
    options: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class DashboardWidget:
    """面板组件"""
    id: str
    title: str
    widget_type: str
    size: Tuple[int, int]  # (width, height)
    position: Tuple[int, int]  # (x, y)
    data_source: str
    config: Dict[str, Any] = field(default_factory=dict)
    refresh_interval: int = 60  # 秒


@dataclass
class BusinessMetric:
    """业务指标"""
    name: str
    value: float
    unit: str
    change_percent: float
    trend: str  # up, down, stable
    description: str
    timestamp: datetime = field(default_factory=datetime.now)


class AnalyticsDashboard:
    """分析面板"""

    def __init__(self, performance_monitor: PerformanceMonitor,
                 database_manager: Optional[DatabaseManager] = None,
                 cache_manager: Optional[CacheManager] = None):
        self.performance_monitor = performance_monitor
        self.database_manager = database_manager
        self.cache_manager = cache_manager

        # 面板配置
        self.widgets: Dict[str, DashboardWidget] = {}
        self.dashboard_layouts: Dict[str, List[DashboardWidget]] = {}

        # 缓存
        self.chart_cache: Dict[str, ChartData] = {}
        self.cache_ttl = 300  # 5分钟

        # 初始化默认面板
        self._setup_default_dashboard()

    def _setup_default_dashboard(self):
        """设置默认面板"""
        # 系统概览面板
        self.dashboard_layouts['system_overview'] = [
            DashboardWidget(
                id='cpu_usage_chart',
                title='CPU使用率',
                widget_type='line_chart',
                size=(6, 4),
                position=(0, 0),
                data_source='system.cpu.usage_percent',
                config={'color': '#ff6b6b', 'unit': '%'}
            ),
            DashboardWidget(
                id='memory_usage_chart',
                title='内存使用率',
                widget_type='line_chart',
                size=(6, 4),
                position=(6, 0),
                data_source='system.memory.usage_percent',
                config={'color': '#4ecdc4', 'unit': '%'}
            ),
            DashboardWidget(
                id='response_time_chart',
                title='响应时间',
                widget_type='area_chart',
                size=(12, 4),
                position=(0, 4),
                data_source='app.response_time.p95_ms',
                config={'color': '#45b7d1', 'unit': 'ms'}
            ),
            DashboardWidget(
                id='error_rate_gauge',
                title='错误率',
                widget_type='gauge',
                size=(4, 4),
                position=(0, 8),
                data_source='app.errors.rate_percent',
                config={'max_value': 10, 'unit': '%'}
            ),
            DashboardWidget(
                id='request_count_metric',
                title='请求总数',
                widget_type='metric_card',
                size=(4, 4),
                position=(4, 8),
                data_source='app.requests.total',
                config={'icon': 'requests'}
            ),
            DashboardWidget(
                id='active_alerts',
                title='活跃告警',
                widget_type='alert_list',
                size=(4, 4),
                position=(8, 8),
                data_source='alerts.active',
                config={'max_items': 5}
            )
        ]

        # 业务分析面板
        self.dashboard_layouts['business_analytics'] = [
            DashboardWidget(
                id='user_activity_heatmap',
                title='用户活动热力图',
                widget_type='heatmap',
                size=(8, 6),
                position=(0, 0),
                data_source='user.activity.hourly',
                config={'color_scheme': 'blues'}
            ),
            DashboardWidget(
                id='popular_features_pie',
                title='热门功能使用分布',
                widget_type='pie_chart',
                size=(4, 6),
                position=(8, 0),
                data_source='features.usage',
                config={'show_legend': True}
            ),
            DashboardWidget(
                id='conversion_funnel',
                title='转化漏斗',
                widget_type='funnel_chart',
                size=(6, 6),
                position=(0, 6),
                data_source='user.conversion_funnel',
                config={'steps': ['访问', '注册', '使用', '付费']}
            ),
            DashboardWidget(
                id='revenue_trend',
                title='收入趋势',
                widget_type='line_chart',
                size=(6, 6),
                position=(6, 6),
                data_source='business.revenue.daily',
                config={'color': '#2ecc71', 'unit': '$'}
            )
        ]

    async def get_dashboard_data(self, dashboard_name: str = 'system_overview',
                               time_range: TimeRange = TimeRange.LAST_24_HOURS) -> Dict[str, Any]:
        """获取面板数据"""
        if dashboard_name not in self.dashboard_layouts:
            raise ValueError(f"Unknown dashboard: {dashboard_name}")

        widgets = self.dashboard_layouts[dashboard_name]
        dashboard_data = {
            'name': dashboard_name,
            'widgets': [],
            'timestamp': datetime.now().isoformat(),
            'time_range': time_range.value
        }

        # 并行获取所有组件的数据
        widget_tasks = []
        for widget in widgets:
            task = asyncio.create_task(
                self._get_widget_data(widget, time_range)
            )
            widget_tasks.append(task)

        widget_results = await asyncio.gather(*widget_tasks, return_exceptions=True)

        for widget, result in zip(widgets, widget_results):
            if isinstance(result, Exception):
                print(f"❌ Error loading widget {widget.id}: {result}")
                widget_data = {
                    'id': widget.id,
                    'title': widget.title,
                    'error': str(result)
                }
            else:
                widget_data = result

            dashboard_data['widgets'].append(widget_data)

        return dashboard_data

    async def _get_widget_data(self, widget: DashboardWidget,
                             time_range: TimeRange) -> Dict[str, Any]:
        """获取组件数据"""
        cache_key = f"widget:{widget.id}:{time_range.value}"

        # 检查缓存
        if self.cache_manager:
            cached_data = await self.cache_manager.get(cache_key)
            if cached_data:
                return json.loads(cached_data)

        # 根据组件类型获取数据
        if widget.widget_type == 'line_chart':
            chart_data = await self._create_line_chart(widget, time_range)
        elif widget.widget_type == 'area_chart':
            chart_data = await self._create_area_chart(widget, time_range)
        elif widget.widget_type == 'bar_chart':
            chart_data = await self._create_bar_chart(widget, time_range)
        elif widget.widget_type == 'pie_chart':
            chart_data = await self._create_pie_chart(widget, time_range)
        elif widget.widget_type == 'gauge':
            chart_data = await self._create_gauge(widget, time_range)
        elif widget.widget_type == 'metric_card':
            chart_data = await self._create_metric_card(widget, time_range)
        elif widget.widget_type == 'alert_list':
            chart_data = await self._create_alert_list(widget, time_range)
        elif widget.widget_type == 'heatmap':
            chart_data = await self._create_heatmap(widget, time_range)
        else:
            chart_data = {'error': f'Unknown widget type: {widget.widget_type}'}

        widget_data = {
            'id': widget.id,
            'title': widget.title,
            'type': widget.widget_type,
            'size': widget.size,
            'position': widget.position,
            'data': chart_data,
            'config': widget.config,
            'timestamp': datetime.now().isoformat()
        }

        # 缓存数据
        if self.cache_manager:
            await self.cache_manager.set(
                cache_key,
                json.dumps(widget_data, default=str),
                expire=self.cache_ttl
            )

        return widget_data

    async def _create_line_chart(self, widget: DashboardWidget,
                               time_range: TimeRange) -> Dict[str, Any]:
        """创建线形图"""
        metric_name = widget.data_source
        duration_minutes = self._get_duration_minutes(time_range)

        # 获取指标历史数据
        metrics = self.performance_monitor.metrics_collector.get_metric_history(
            metric_name, duration_minutes
        )

        if not metrics:
            return {'labels': [], 'datasets': []}

        # 准备数据
        labels = []
        values = []

        for metric in metrics:
            labels.append(metric.timestamp.strftime('%H:%M'))
            values.append(metric.value)

        return {
            'labels': labels,
            'datasets': [{
                'label': widget.title,
                'data': values,
                'borderColor': widget.config.get('color', '#007bff'),
                'backgroundColor': f"{widget.config.get('color', '#007bff')}20",
                'tension': 0.4
            }],
            'options': {
                'responsive': True,
                'scales': {
                    'y': {
                        'beginAtZero': True,
                        'title': {
                            'display': True,
                            'text': widget.config.get('unit', '')
                        }
                    }
                }
            }
        }

    async def _create_area_chart(self, widget: DashboardWidget,
                               time_range: TimeRange) -> Dict[str, Any]:
        """创建面积图"""
        line_data = await self._create_line_chart(widget, time_range)
        if line_data and 'datasets' in line_data:
            line_data['datasets'][0]['fill'] = True
        return line_data

    async def _create_bar_chart(self, widget: DashboardWidget,
                              time_range: TimeRange) -> Dict[str, Any]:
        """创建柱状图"""
        # 这里可以根据具体需求实现柱状图数据
        return {
            'labels': ['API-1', 'API-2', 'API-3', 'API-4'],
            'datasets': [{
                'label': widget.title,
                'data': [65, 59, 80, 81],
                'backgroundColor': [
                    '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'
                ]
            }]
        }

    async def _create_pie_chart(self, widget: DashboardWidget,
                              time_range: TimeRange) -> Dict[str, Any]:
        """创建饼图"""
        # 示例：功能使用分布
        return {
            'labels': ['视频生成', '图像处理', '音频合成', '文本分析'],
            'datasets': [{
                'data': [45, 25, 20, 10],
                'backgroundColor': [
                    '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'
                ]
            }]
        }

    async def _create_gauge(self, widget: DashboardWidget,
                          time_range: TimeRange) -> Dict[str, Any]:
        """创建仪表盘"""
        metric_name = widget.data_source
        latest_metric = self.performance_monitor.metrics_collector.get_latest_metric(metric_name)

        current_value = latest_metric.value if latest_metric else 0
        max_value = widget.config.get('max_value', 100)

        return {
            'value': current_value,
            'max': max_value,
            'unit': widget.config.get('unit', ''),
            'color': self._get_gauge_color(current_value, max_value),
            'ranges': [
                {'from': 0, 'to': max_value * 0.7, 'color': '#2ecc71'},
                {'from': max_value * 0.7, 'to': max_value * 0.9, 'color': '#f39c12'},
                {'from': max_value * 0.9, 'to': max_value, 'color': '#e74c3c'}
            ]
        }

    def _get_gauge_color(self, value: float, max_value: float) -> str:
        """获取仪表盘颜色"""
        ratio = value / max_value
        if ratio < 0.7:
            return '#2ecc71'  # 绿色
        elif ratio < 0.9:
            return '#f39c12'  # 橙色
        else:
            return '#e74c3c'  # 红色

    async def _create_metric_card(self, widget: DashboardWidget,
                                time_range: TimeRange) -> Dict[str, Any]:
        """创建指标卡片"""
        metric_name = widget.data_source
        latest_metric = self.performance_monitor.metrics_collector.get_latest_metric(metric_name)

        if not latest_metric:
            return {'value': 0, 'change': 0}

        current_value = latest_metric.value

        # 计算变化百分比
        duration_minutes = min(self._get_duration_minutes(time_range), 60)
        historical_metrics = self.performance_monitor.metrics_collector.get_metric_history(
            metric_name, duration_minutes
        )

        change_percent = 0
        if len(historical_metrics) > 1:
            old_value = historical_metrics[0].value
            if old_value != 0:
                change_percent = ((current_value - old_value) / old_value) * 100

        return {
            'value': self._format_metric_value(current_value),
            'change': round(change_percent, 1),
            'trend': 'up' if change_percent > 0 else 'down' if change_percent < 0 else 'stable',
            'unit': widget.config.get('unit', ''),
            'icon': widget.config.get('icon', 'metric')
        }

    def _format_metric_value(self, value: float) -> str:
        """格式化指标值"""
        if value >= 1_000_000:
            return f"{value / 1_000_000:.1f}M"
        elif value >= 1_000:
            return f"{value / 1_000:.1f}K"
        else:
            return f"{value:.0f}"

    async def _create_alert_list(self, widget: DashboardWidget,
                               time_range: TimeRange) -> Dict[str, Any]:
        """创建告警列表"""
        active_alerts = self.performance_monitor.alert_manager.get_active_alerts()
        max_items = widget.config.get('max_items', 10)

        alert_data = []
        for alert in active_alerts[:max_items]:
            alert_data.append({
                'id': alert.id,
                'message': alert.message,
                'level': alert.level.value,
                'timestamp': alert.timestamp.strftime('%H:%M:%S'),
                'metric': alert.metric_name,
                'value': alert.current_value
            })

        return {
            'alerts': alert_data,
            'total_count': len(active_alerts)
        }

    async def _create_heatmap(self, widget: DashboardWidget,
                            time_range: TimeRange) -> Dict[str, Any]:
        """创建热力图"""
        # 示例：24小时用户活动热力图
        hours = list(range(24))
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

        # 生成示例数据
        import random
        data = []
        for day_idx, day in enumerate(days):
            for hour in hours:
                intensity = random.randint(10, 100)
                data.append({
                    'x': hour,
                    'y': day_idx,
                    'v': intensity
                })

        return {
            'data': data,
            'xLabels': [f"{h:02d}:00" for h in hours],
            'yLabels': days,
            'colorScale': widget.config.get('color_scheme', 'blues')
        }

    def _get_duration_minutes(self, time_range: TimeRange) -> int:
        """获取时间范围对应的分钟数"""
        duration_map = {
            TimeRange.LAST_HOUR: 60,
            TimeRange.LAST_6_HOURS: 360,
            TimeRange.LAST_24_HOURS: 1440,
            TimeRange.LAST_7_DAYS: 10080,
            TimeRange.LAST_30_DAYS: 43200
        }
        return duration_map.get(time_range, 1440)

    async def get_business_metrics(self) -> List[BusinessMetric]:
        """获取业务指标"""
        metrics = []

        # 系统健康度
        health = self.performance_monitor.get_system_health()
        health_score = 100
        if health.overall_status == 'warning':
            health_score = 75
        elif health.overall_status == 'critical':
            health_score = 50

        metrics.append(BusinessMetric(
            name='系统健康度',
            value=health_score,
            unit='%',
            change_percent=0,  # 需要历史数据计算
            trend='stable',
            description='系统整体健康状况评分'
        ))

        # 性能指标
        dashboard_data = self.performance_monitor.get_dashboard_data()
        total_requests = dashboard_data['performance']['total_requests']
        uptime_hours = dashboard_data['performance']['uptime_hours']

        if uptime_hours > 0:
            requests_per_hour = total_requests / uptime_hours
            metrics.append(BusinessMetric(
                name='请求处理率',
                value=requests_per_hour,
                unit='req/h',
                change_percent=5.2,  # 示例数据
                trend='up',
                description='每小时处理的请求数量'
            ))

        # 可用性
        error_rate = health.error_rate
        availability = max(0, 100 - error_rate)
        metrics.append(BusinessMetric(
            name='系统可用性',
            value=availability,
            unit='%',
            change_percent=-0.1,
            trend='down',
            description='系统可用性百分比'
        ))

        return metrics

    async def export_dashboard_config(self, dashboard_name: str) -> Dict[str, Any]:
        """导出面板配置"""
        if dashboard_name not in self.dashboard_layouts:
            raise ValueError(f"Unknown dashboard: {dashboard_name}")

        widgets = self.dashboard_layouts[dashboard_name]
        config = {
            'name': dashboard_name,
            'version': '1.0',
            'created_at': datetime.now().isoformat(),
            'widgets': [
                {
                    'id': widget.id,
                    'title': widget.title,
                    'type': widget.widget_type,
                    'size': widget.size,
                    'position': widget.position,
                    'data_source': widget.data_source,
                    'config': widget.config,
                    'refresh_interval': widget.refresh_interval
                }
                for widget in widgets
            ]
        }
        return config

    async def import_dashboard_config(self, config: Dict[str, Any]) -> bool:
        """导入面板配置"""
        try:
            dashboard_name = config['name']
            widgets = []

            for widget_config in config['widgets']:
                widget = DashboardWidget(
                    id=widget_config['id'],
                    title=widget_config['title'],
                    widget_type=widget_config['type'],
                    size=tuple(widget_config['size']),
                    position=tuple(widget_config['position']),
                    data_source=widget_config['data_source'],
                    config=widget_config.get('config', {}),
                    refresh_interval=widget_config.get('refresh_interval', 60)
                )
                widgets.append(widget)

            self.dashboard_layouts[dashboard_name] = widgets
            print(f"✅ Dashboard imported: {dashboard_name}")
            return True

        except Exception as e:
            print(f"❌ Failed to import dashboard: {e}")
            return False

    async def get_performance_report(self, time_range: TimeRange = TimeRange.LAST_24_HOURS) -> Dict[str, Any]:
        """生成性能报告"""
        duration_minutes = self._get_duration_minutes(time_range)

        # 获取系统指标统计
        cpu_stats = self.performance_monitor.metrics_collector.calculate_metric_stats(
            'system.cpu.usage_percent', duration_minutes
        )
        memory_stats = self.performance_monitor.metrics_collector.calculate_metric_stats(
            'system.memory.usage_percent', duration_minutes
        )
        response_time_stats = self.performance_monitor.metrics_collector.calculate_metric_stats(
            'app.response_time.p95_ms', duration_minutes
        )

        # 告警统计
        alert_history = self.performance_monitor.alert_manager.get_alert_history(
            duration_minutes // 60
        )

        report = {
            'period': {
                'range': time_range.value,
                'start_time': (datetime.now() - timedelta(minutes=duration_minutes)).isoformat(),
                'end_time': datetime.now().isoformat()
            },
            'system_performance': {
                'cpu': cpu_stats,
                'memory': memory_stats,
                'response_time': response_time_stats
            },
            'alerts': {
                'total_alerts': len(alert_history),
                'critical_alerts': len([a for a in alert_history if a.level.value == 'critical']),
                'warning_alerts': len([a for a in alert_history if a.level.value == 'warning']),
                'most_frequent_alerts': self._get_most_frequent_alerts(alert_history)
            },
            'recommendations': self._generate_performance_recommendations(
                cpu_stats, memory_stats, response_time_stats, alert_history
            ),
            'generated_at': datetime.now().isoformat()
        }

        return report

    def _get_most_frequent_alerts(self, alert_history: List) -> List[Dict[str, Any]]:
        """获取最频繁的告警"""
        alert_counter = Counter(alert.message for alert in alert_history)
        return [
            {'message': message, 'count': count}
            for message, count in alert_counter.most_common(5)
        ]

    def _generate_performance_recommendations(self, cpu_stats: Dict, memory_stats: Dict,
                                           response_time_stats: Dict, alert_history: List) -> List[str]:
        """生成性能优化建议"""
        recommendations = []

        # CPU建议
        if cpu_stats and cpu_stats.get('mean', 0) > 70:
            recommendations.append("CPU平均使用率较高，建议优化CPU密集型任务或增加计算资源")

        # 内存建议
        if memory_stats and memory_stats.get('mean', 0) > 80:
            recommendations.append("内存使用率持续偏高，建议检查内存泄漏或增加内存容量")

        # 响应时间建议
        if response_time_stats and response_time_stats.get('p95', 0) > 1000:
            recommendations.append("95%响应时间超过1秒，建议优化查询性能或增加缓存")

        # 告警建议
        if len(alert_history) > 10:
            recommendations.append("告警频率较高，建议调整告警阈值或解决根本问题")

        if not recommendations:
            recommendations.append("系统运行良好，继续保持监控")

        return recommendations