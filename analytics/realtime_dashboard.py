"""
Real-time Analytics Dashboard

Advanced analytics dashboard for real-time monitoring and visualization
of Aura Render platform metrics.
"""

import asyncio
import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from collections import deque, defaultdict
import psutil
import logging

from fastapi import WebSocket, WebSocketDisconnect
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class RealtimeMetric:
    """Real-time metric data point"""
    timestamp: float
    name: str
    value: float
    unit: str
    tags: Dict[str, str]


@dataclass
class DashboardWidget:
    """Dashboard widget configuration"""
    id: str
    type: str  # line, bar, pie, gauge, heatmap, table
    title: str
    metrics: List[str]
    refresh_interval: int = 5  # seconds
    time_window: int = 3600  # seconds (1 hour)
    options: Dict[str, Any] = None


class MetricsAggregator:
    """Aggregates and processes metrics for dashboard visualization"""
    
    def __init__(self, window_size: int = 3600):
        self.window_size = window_size
        self.metrics_buffer = defaultdict(lambda: deque(maxlen=1000))
        self.aggregated_data = {}
        self.last_aggregation = time.time()
        
    def add_metric(self, metric: RealtimeMetric):
        """Add a new metric to the buffer"""
        key = f"{metric.name}:{json.dumps(metric.tags, sort_keys=True)}"
        self.metrics_buffer[key].append({
            'timestamp': metric.timestamp,
            'value': metric.value,
            'unit': metric.unit
        })
        
    def aggregate(self) -> Dict[str, Any]:
        """Aggregate metrics for dashboard display"""
        current_time = time.time()
        cutoff_time = current_time - self.window_size
        
        aggregated = {}
        
        for key, buffer in self.metrics_buffer.items():
            # Filter data within time window
            recent_data = [
                point for point in buffer 
                if point['timestamp'] > cutoff_time
            ]
            
            if not recent_data:
                continue
                
            values = [point['value'] for point in recent_data]
            timestamps = [point['timestamp'] for point in recent_data]
            
            # Calculate statistics
            aggregated[key] = {
                'current': values[-1] if values else 0,
                'min': min(values),
                'max': max(values),
                'avg': sum(values) / len(values),
                'p50': np.percentile(values, 50),
                'p95': np.percentile(values, 95),
                'p99': np.percentile(values, 99),
                'count': len(values),
                'rate': self._calculate_rate(timestamps, values),
                'trend': self._calculate_trend(timestamps, values),
                'data_points': list(zip(timestamps, values))[-100:]  # Last 100 points
            }
            
        self.aggregated_data = aggregated
        self.last_aggregation = current_time
        return aggregated
    
    def _calculate_rate(self, timestamps: List[float], values: List[float]) -> float:
        """Calculate rate of change"""
        if len(timestamps) < 2:
            return 0.0
            
        time_diff = timestamps[-1] - timestamps[0]
        if time_diff == 0:
            return 0.0
            
        value_diff = values[-1] - values[0]
        return value_diff / time_diff
    
    def _calculate_trend(self, timestamps: List[float], values: List[float]) -> str:
        """Calculate trend direction"""
        if len(values) < 3:
            return "stable"
            
        recent = values[-10:]  # Last 10 values
        older = values[-20:-10] if len(values) >= 20 else values[:-10]
        
        if not older:
            return "stable"
            
        recent_avg = sum(recent) / len(recent)
        older_avg = sum(older) / len(older)
        
        diff_percent = ((recent_avg - older_avg) / older_avg) * 100 if older_avg != 0 else 0
        
        if diff_percent > 5:
            return "increasing"
        elif diff_percent < -5:
            return "decreasing"
        else:
            return "stable"


class RealtimeDashboard:
    """Real-time analytics dashboard manager"""
    
    def __init__(self):
        self.aggregator = MetricsAggregator()
        self.connected_clients: List[WebSocket] = []
        self.widgets = self._initialize_widgets()
        self.metrics_collector = None
        self._running = False
        
    def _initialize_widgets(self) -> List[DashboardWidget]:
        """Initialize default dashboard widgets"""
        return [
            DashboardWidget(
                id="system_health",
                type="gauge",
                title="System Health",
                metrics=["system.health.score"],
                refresh_interval=5
            ),
            DashboardWidget(
                id="request_rate",
                type="line",
                title="Request Rate",
                metrics=["http.requests.rate"],
                refresh_interval=5,
                time_window=1800
            ),
            DashboardWidget(
                id="video_generation",
                type="bar",
                title="Video Generation Status",
                metrics=[
                    "video.generation.success",
                    "video.generation.failed",
                    "video.generation.pending"
                ],
                refresh_interval=10
            ),
            DashboardWidget(
                id="resource_usage",
                type="line",
                title="Resource Usage",
                metrics=[
                    "system.cpu.usage",
                    "system.memory.usage",
                    "system.disk.usage"
                ],
                refresh_interval=5,
                options={"stacked": False, "show_legend": True}
            ),
            DashboardWidget(
                id="response_time",
                type="heatmap",
                title="Response Time Distribution",
                metrics=["http.response.time"],
                refresh_interval=10,
                options={"buckets": [0, 100, 250, 500, 1000, 2000, 5000]}
            ),
            DashboardWidget(
                id="ai_optimization",
                type="pie",
                title="AI Optimization Distribution",
                metrics=[
                    "ai.optimization.basic",
                    "ai.optimization.standard",
                    "ai.optimization.premium"
                ],
                refresh_interval=30
            ),
            DashboardWidget(
                id="error_rate",
                type="line",
                title="Error Rate",
                metrics=["http.errors.rate"],
                refresh_interval=5,
                options={"color": "red", "show_alert_threshold": True}
            ),
            DashboardWidget(
                id="queue_status",
                type="table",
                title="Queue Status",
                metrics=[
                    "queue.video_generation.length",
                    "queue.optimization.length",
                    "queue.rendering.length"
                ],
                refresh_interval=10
            )
        ]
    
    async def start(self):
        """Start the dashboard background tasks"""
        self._running = True
        
        # Start metrics collection
        asyncio.create_task(self._collect_system_metrics())
        
        # Start dashboard update loop
        asyncio.create_task(self._update_dashboard_loop())
        
        logger.info("Real-time dashboard started")
    
    async def stop(self):
        """Stop the dashboard"""
        self._running = False
        
        # Close all WebSocket connections
        for client in self.connected_clients:
            await client.close()
        
        self.connected_clients.clear()
        logger.info("Real-time dashboard stopped")
    
    async def connect_client(self, websocket: WebSocket):
        """Connect a new dashboard client"""
        await websocket.accept()
        self.connected_clients.append(websocket)
        
        # Send initial dashboard configuration
        await websocket.send_json({
            "type": "init",
            "widgets": [asdict(w) for w in self.widgets],
            "timestamp": time.time()
        })
        
        # Send current metrics
        metrics = self.aggregator.aggregate()
        await websocket.send_json({
            "type": "metrics",
            "data": self._format_metrics_for_client(metrics),
            "timestamp": time.time()
        })
        
        logger.info(f"Dashboard client connected. Total clients: {len(self.connected_clients)}")
    
    async def disconnect_client(self, websocket: WebSocket):
        """Disconnect a dashboard client"""
        if websocket in self.connected_clients:
            self.connected_clients.remove(websocket)
        
        logger.info(f"Dashboard client disconnected. Total clients: {len(self.connected_clients)}")
    
    async def handle_client_message(self, websocket: WebSocket, message: Dict[str, Any]):
        """Handle messages from dashboard clients"""
        msg_type = message.get("type")
        
        if msg_type == "subscribe":
            # Client subscribing to specific metrics
            metrics = message.get("metrics", [])
            # Handle subscription logic
            
        elif msg_type == "query":
            # Client requesting specific data
            query = message.get("query")
            result = await self._process_query(query)
            await websocket.send_json({
                "type": "query_result",
                "data": result,
                "timestamp": time.time()
            })
            
        elif msg_type == "config":
            # Client updating widget configuration
            widget_id = message.get("widget_id")
            config = message.get("config")
            self._update_widget_config(widget_id, config)
    
    async def _collect_system_metrics(self):
        """Collect system metrics continuously"""
        while self._running:
            try:
                # CPU usage
                cpu_percent = psutil.cpu_percent(interval=1)
                self.aggregator.add_metric(RealtimeMetric(
                    timestamp=time.time(),
                    name="system.cpu.usage",
                    value=cpu_percent,
                    unit="percent",
                    tags={"type": "system"}
                ))
                
                # Memory usage
                memory = psutil.virtual_memory()
                self.aggregator.add_metric(RealtimeMetric(
                    timestamp=time.time(),
                    name="system.memory.usage",
                    value=memory.percent,
                    unit="percent",
                    tags={"type": "system"}
                ))
                
                # Disk usage
                disk = psutil.disk_usage('/')
                self.aggregator.add_metric(RealtimeMetric(
                    timestamp=time.time(),
                    name="system.disk.usage",
                    value=disk.percent,
                    unit="percent",
                    tags={"type": "system"}
                ))
                
                # Network I/O
                net_io = psutil.net_io_counters()
                self.aggregator.add_metric(RealtimeMetric(
                    timestamp=time.time(),
                    name="system.network.bytes_sent",
                    value=net_io.bytes_sent,
                    unit="bytes",
                    tags={"type": "system", "direction": "out"}
                ))
                
                self.aggregator.add_metric(RealtimeMetric(
                    timestamp=time.time(),
                    name="system.network.bytes_recv",
                    value=net_io.bytes_recv,
                    unit="bytes",
                    tags={"type": "system", "direction": "in"}
                ))
                
                # Process count
                process_count = len(psutil.pids())
                self.aggregator.add_metric(RealtimeMetric(
                    timestamp=time.time(),
                    name="system.process.count",
                    value=process_count,
                    unit="count",
                    tags={"type": "system"}
                ))
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(5)
    
    async def _update_dashboard_loop(self):
        """Update dashboard clients periodically"""
        while self._running:
            try:
                # Aggregate metrics
                metrics = self.aggregator.aggregate()
                
                if self.connected_clients:
                    # Format metrics for clients
                    formatted_metrics = self._format_metrics_for_client(metrics)
                    
                    # Send to all connected clients
                    disconnected_clients = []
                    
                    for client in self.connected_clients:
                        try:
                            await client.send_json({
                                "type": "metrics",
                                "data": formatted_metrics,
                                "timestamp": time.time()
                            })
                        except WebSocketDisconnect:
                            disconnected_clients.append(client)
                        except Exception as e:
                            logger.error(f"Error sending metrics to client: {e}")
                            disconnected_clients.append(client)
                    
                    # Remove disconnected clients
                    for client in disconnected_clients:
                        await self.disconnect_client(client)
                
                await asyncio.sleep(5)
                
            except Exception as e:
                logger.error(f"Error in dashboard update loop: {e}")
                await asyncio.sleep(5)
    
    def _format_metrics_for_client(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Format aggregated metrics for dashboard display"""
        formatted = {}
        
        for key, data in metrics.items():
            metric_name = key.split(':')[0]
            
            formatted[metric_name] = {
                'current': data['current'],
                'min': data['min'],
                'max': data['max'],
                'avg': data['avg'],
                'trend': data['trend'],
                'sparkline': data['data_points'][-20:] if 'data_points' in data else []
            }
        
        return formatted
    
    async def _process_query(self, query: Dict[str, Any]) -> Dict[str, Any]:
        """Process custom queries from dashboard clients"""
        query_type = query.get("type")
        
        if query_type == "time_range":
            # Query metrics for specific time range
            start_time = query.get("start")
            end_time = query.get("end")
            metrics = query.get("metrics", [])
            
            # Process time range query
            return {"status": "success", "data": {}}
            
        elif query_type == "aggregate":
            # Aggregate metrics with custom function
            metric = query.get("metric")
            function = query.get("function", "avg")
            group_by = query.get("group_by")
            
            # Process aggregation query
            return {"status": "success", "data": {}}
            
        else:
            return {"status": "error", "message": f"Unknown query type: {query_type}"}
    
    def _update_widget_config(self, widget_id: str, config: Dict[str, Any]):
        """Update widget configuration"""
        for widget in self.widgets:
            if widget.id == widget_id:
                if 'refresh_interval' in config:
                    widget.refresh_interval = config['refresh_interval']
                if 'time_window' in config:
                    widget.time_window = config['time_window']
                if 'options' in config:
                    widget.options = {**(widget.options or {}), **config['options']}
                break
    
    def add_custom_metric(self, name: str, value: float, tags: Dict[str, str] = None):
        """Add a custom metric to the dashboard"""
        self.aggregator.add_metric(RealtimeMetric(
            timestamp=time.time(),
            name=name,
            value=value,
            unit="count",
            tags=tags or {}
        ))
    
    def get_dashboard_stats(self) -> Dict[str, Any]:
        """Get dashboard statistics"""
        return {
            "connected_clients": len(self.connected_clients),
            "total_metrics": len(self.aggregator.metrics_buffer),
            "last_aggregation": self.aggregator.last_aggregation,
            "widgets_count": len(self.widgets),
            "uptime": time.time() - (self.aggregator.last_aggregation or time.time())
        }


class AnalyticsProcessor:
    """Process and analyze metrics for insights"""
    
    def __init__(self):
        self.anomaly_detector = AnomalyDetector()
        self.trend_analyzer = TrendAnalyzer()
        self.correlation_finder = CorrelationFinder()
        
    async def process_metrics(self, metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Process metrics to generate insights"""
        insights = {
            "anomalies": [],
            "trends": [],
            "correlations": [],
            "predictions": []
        }
        
        # Detect anomalies
        for metric_name, data in metrics.items():
            if 'data_points' in data and len(data['data_points']) > 10:
                anomalies = self.anomaly_detector.detect(data['data_points'])
                if anomalies:
                    insights["anomalies"].append({
                        "metric": metric_name,
                        "anomalies": anomalies
                    })
        
        # Analyze trends
        trends = self.trend_analyzer.analyze(metrics)
        insights["trends"] = trends
        
        # Find correlations
        correlations = self.correlation_finder.find(metrics)
        insights["correlations"] = correlations
        
        return insights


class AnomalyDetector:
    """Detect anomalies in metric data"""
    
    def detect(self, data_points: List[tuple]) -> List[Dict[str, Any]]:
        """Detect anomalies using statistical methods"""
        if len(data_points) < 10:
            return []
            
        values = [point[1] for point in data_points]
        timestamps = [point[0] for point in data_points]
        
        # Calculate statistics
        mean = np.mean(values)
        std = np.std(values)
        
        anomalies = []
        
        # Z-score based detection
        for i, value in enumerate(values):
            z_score = abs((value - mean) / std) if std > 0 else 0
            
            if z_score > 3:  # 3 standard deviations
                anomalies.append({
                    "timestamp": timestamps[i],
                    "value": value,
                    "z_score": z_score,
                    "severity": "high" if z_score > 4 else "medium"
                })
        
        return anomalies


class TrendAnalyzer:
    """Analyze trends in metrics"""
    
    def analyze(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Analyze trends across metrics"""
        trends = []
        
        for metric_name, data in metrics.items():
            if 'trend' in data:
                trend_info = {
                    "metric": metric_name,
                    "direction": data['trend'],
                    "rate": data.get('rate', 0),
                    "confidence": self._calculate_confidence(data)
                }
                
                # Add trend classification
                if data['trend'] == "increasing" and data.get('rate', 0) > 0.1:
                    trend_info["alert"] = "rapid_increase"
                elif data['trend'] == "decreasing" and data.get('rate', 0) < -0.1:
                    trend_info["alert"] = "rapid_decrease"
                
                trends.append(trend_info)
        
        return trends
    
    def _calculate_confidence(self, data: Dict[str, Any]) -> float:
        """Calculate confidence in trend detection"""
        if 'count' not in data:
            return 0.0
            
        # More data points = higher confidence
        count_factor = min(data['count'] / 100, 1.0)
        
        # Lower variance = higher confidence
        if 'max' in data and 'min' in data and 'avg' in data:
            range_val = data['max'] - data['min']
            variance_factor = 1.0 - min(range_val / (data['avg'] + 0.001), 1.0)
        else:
            variance_factor = 0.5
        
        return (count_factor + variance_factor) / 2


class CorrelationFinder:
    """Find correlations between metrics"""
    
    def find(self, metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find correlations between different metrics"""
        correlations = []
        
        metric_names = list(metrics.keys())
        
        for i in range(len(metric_names)):
            for j in range(i + 1, len(metric_names)):
                metric1 = metrics[metric_names[i]]
                metric2 = metrics[metric_names[j]]
                
                if 'data_points' in metric1 and 'data_points' in metric2:
                    correlation = self._calculate_correlation(
                        metric1['data_points'],
                        metric2['data_points']
                    )
                    
                    if abs(correlation) > 0.7:  # Strong correlation
                        correlations.append({
                            "metric1": metric_names[i],
                            "metric2": metric_names[j],
                            "correlation": correlation,
                            "strength": "strong" if abs(correlation) > 0.9 else "moderate"
                        })
        
        return correlations
    
    def _calculate_correlation(self, data1: List[tuple], data2: List[tuple]) -> float:
        """Calculate Pearson correlation coefficient"""
        # Align data points by timestamp
        times1 = {point[0]: point[1] for point in data1}
        times2 = {point[0]: point[1] for point in data2}
        
        common_times = set(times1.keys()) & set(times2.keys())
        
        if len(common_times) < 3:
            return 0.0
        
        values1 = [times1[t] for t in sorted(common_times)]
        values2 = [times2[t] for t in sorted(common_times)]
        
        # Calculate correlation
        correlation_matrix = np.corrcoef(values1, values2)
        return correlation_matrix[0, 1] if not np.isnan(correlation_matrix[0, 1]) else 0.0


# Global dashboard instance
dashboard = RealtimeDashboard()


async def start_dashboard():
    """Start the real-time dashboard"""
    await dashboard.start()


async def stop_dashboard():
    """Stop the real-time dashboard"""
    await dashboard.stop()