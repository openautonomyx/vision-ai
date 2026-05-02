"""
Unit tests for AutonomyX Vision AI - Metrics Module

Run with: pytest tests/test_metrics.py -v
"""

import pytest
import time
from unittest.mock import patch, MagicMock


class TestMetricsModule:
    """Tests for metrics.py module."""
    
    def test_metrics_increment_counter(self):
        """Test incrementing a counter."""
        from metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.increment("test_counter")
        
        assert collector._counters["test_counter"] == 1
    
    def test_metrics_increment_with_value(self):
        """Test incrementing counter with custom value."""
        from metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.increment("test_counter", value=5)
        
        assert collector._counters["test_counter"] == 5
    
    def test_metrics_increment_with_labels(self):
        """Test incrementing with labels."""
        from metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.increment("test_counter", {"endpoint": "/health", "status": "200"})
        
        key = collector._make_key("test_counter", {"endpoint": "/health", "status": "200"})
        assert collector._counters[key] == 1
    
    def test_metrics_observe_histogram(self):
        """Test observing histogram value."""
        from metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.observe("test_histogram", 0.5)
        
        assert 0.5 in collector._histograms["test_histogram"]
    
    def test_metrics_observe_multiple_values(self):
        """Test observing multiple values."""
        from metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.observe("test_histogram", 0.1)
        collector.observe("test_histogram", 0.2)
        collector.observe("test_histogram", 0.3)
        
        assert len(collector._histograms["test_histogram"]) == 3
    
    def test_metrics_gauge(self):
        """Test setting gauge value."""
        from metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.gauge("test_gauge", 42.0)
        
        assert collector._gauges["test_gauge"] == 42.0
    
    def test_metrics_record_request(self):
        """Test recording request."""
        from metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.record_request("/health", 200, 0.5)
        
        assert len(collector._request_times) == 1
        req = collector._request_times[0]
        assert req["endpoint"] == "/health"
        assert req["status"] == 200
    
    def test_metrics_get_stats_empty(self):
        """Test stats when no requests."""
        from metrics import MetricsCollector
        
        collector = MetricsCollector()
        stats = collector.get_stats()
        
        assert stats["total_requests"] == 0
    
    def test_metrics_get_stats_with_requests(self):
        """Test stats with recorded requests."""
        from metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.record_request("/health", 200, 0.5)
        collector.record_request("/health", 200, 0.3)
        collector.record_request("/health", 200, 0.7)
        
        stats = collector.get_stats()
        
        assert stats["total_requests"] == 3
        assert 0 <= stats["error_rate"] <= 1
    
    def test_generate_request_id(self):
        """Test request ID generation."""
        from metrics import generate_request_id
        
        req_id = generate_request_id()
        
        assert req_id.startswith("req_")
        assert len(req_id) > 4
    
    def test_generate_request_id_unique(self):
        """Test unique request IDs."""
        from metrics import generate_request_id
        
        ids = set()
        for _ in range(100):
            ids.add(generate_request_id())
        
        # All should be unique
        assert len(ids) == 100
    
    def test_metrics_format_counters(self):
        """Test Prometheus format output."""
        from metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.increment("test_counter")
        collector.increment("test_counter")
        
        output = collector.get_metrics()
        
        assert "# TYPE test_counter counter" in output
        assert "test_counter 2" in output
    
    def test_metrics_format_histogram(self):
        """Test histogram output format."""
        from metrics import MetricsCollector
        
        collector = MetricsCollector()
        collector.observe("test_histogram", 0.5)
        
        output = collector.get_metrics()
        
        assert "# TYPE test_histogram histogram" in output


if __name__ == "__main__":
    pytest.main([__file__, "-v"])