"""
AutonomyX Vision AI — Metrics and Observability
"""
from collections import defaultdict
from datetime import datetime
import threading
import time
import uuid

class MetricsCollector:
    """Collect and expose Prometheus metrics."""
    
    def __init__(self):
        self._counters: dict[str, int] = defaultdict(int)
        self._histograms: dict[str, list[float]] = defaultdict(list)
        self._gauges: dict[str, float] = {}
        self._lock = threading.Lock()
        self._request_times: list[dict] = []  # Keep last 1000 requests
    
    def increment(self, name: str, labels: dict = None, value: int = 1):
        """Increment a counter."""
        with self._lock:
            key = self._make_key(name, labels)
            self._counters[key] += value
    
    def observe(self, name: str, value: float, labels: dict = None):
        """Observe a histogram value."""
        with self._lock:
            key = self._make_key(name, labels)
            self._histograms[key].append(value)
            # Keep only last 1000 values
            if len(self._histograms[key]) > 1000:
                self._histograms[key] = self._histograms[key][-1000:]
    
    def gauge(self, name: str, value: float, labels: dict = None):
        """Set a gauge value."""
        with self._lock:
            key = self._make_key(name, labels)
            self._gauges[key] = value
    
    def _make_key(self, name: str, labels: dict = None) -> str:
        if not labels:
            return name
        label_str = ",".join(f'{k}="{v}"' for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"
    
    def record_request(self, endpoint: str, status: int, duration: float, api_key: str = None):
        """Record a request for metrics."""
        with self._lock:
            self._request_times.append({
                "endpoint": endpoint,
                "status": status,
                "duration": duration,
                "timestamp": datetime.utcnow().isoformat()
            })
            # Keep last 1000
            if len(self._request_times) > 1000:
                self._request_times = self._request_times[-1000:]
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format."""
        lines = []
        
        with self._lock:
            # Counters
            for key, value in self._counters.items():
                lines.append(f"# TYPE {key} counter")
                lines.append(f"{key} {value}")
            
            # Histograms
            for key, values in self._histograms.items():
                if values:
                    sorted_values = sorted(values)
                    n = len(sorted_values)
                    lines.append(f"# TYPE {key} histogram")
                    
                    # Calculate percentiles
                    p50 = sorted_values[int(n * 0.50)] if n > 0 else 0
                    p95 = sorted_values[int(n * 0.95)] if n > 0 else 0
                    p99 = sorted_values[int(n * 0.99)] if n > 0 else 0
                    
                    lines.append(f'{key}_sum {sum(values)}')
                    lines.append(f'{key}_count {n}')
                    lines.append(f'{key}_bucket{{le="{p50}"}} {int(n * 0.50)}')
                    lines.append(f'{key}_bucket{{le="{p95}"}} {int(n * 0.95)}')
                    lines.append(f'{key}_bucket{{le="{p99}"}} {int(n * 0.99)}')
                    lines.append(f'{key}_bucket{{le="+Inf"}} {n}')
            
            # Gauges
            for key, value in self._gauges.items():
                lines.append(f"# TYPE {key} gauge")
                lines.append(f"{key} {value}")
        
        return "\n".join(lines)
    
    def get_stats(self) -> dict:
        """Get basic stats."""
        with self._lock:
            total_requests = len(self._request_times)
            if total_requests == 0:
                return {"total_requests": 0, "error_rate": 0}
            
            errors = sum(1 for r in self._request_times if r["status"] >= 400)
            durations = [r["duration"] for r in self._request_times]
            
            return {
                "total_requests": total_requests,
                "error_rate": errors / total_requests,
                "avg_duration_ms": sum(durations) / total_requests * 1000,
                "p50_duration_ms": sorted(durations)[int(total_requests * 0.50)] * 1000 if durations else 0,
                "p95_duration_ms": sorted(durations)[int(total_requests * 0.95)] * 1000 if durations else 0,
                "p99_duration_ms": sorted(durations)[int(total_requests * 0.99)] * 1000 if durations else 0,
            }

# Global metrics collector
metrics = MetricsCollector()

def generate_request_id() -> str:
    """Generate a unique request ID."""
    return f"req_{uuid.uuid4().hex[:16]}"

def timingMiddleware(endpoint: str):
    """Decorator for timing endpoints."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            start = time.time()
            try:
                result = func(*args, **kwargs)
                status = 200
                return result
            except Exception as e:
                status = 500
                raise
            finally:
                duration = time.time() - start
                metrics.increment(f"vision_api_requests_total", {"endpoint": endpoint, "status": status})
                metrics.observe(f"vision_api_request_duration_seconds", duration, {"endpoint": endpoint})
        return wrapper
    return decorator