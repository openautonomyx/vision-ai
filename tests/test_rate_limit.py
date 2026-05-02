"""
Unit tests for AutonomyX Vision AI - Rate Limiting Module

Run with: pytest tests/test_rate_limit.py -v
"""

import pytest
import time
from unittest.mock import patch, MagicMock


class TestRateLimitModule:
    """Tests for rate_limit.py module."""
    
    def test_rate_limiter_initializes_with_full_tokens(self):
        """Test rate limiter starts with full tokens."""
        from rate_limit import RateLimiter
        
        limiter = RateLimiter()
        
        # Initially should have full tokens when bucket is created
        remaining = limiter.get_remaining("test_key", rate_limit=100)
        assert remaining == 100
    
    def test_rate_limiter_check_allows_first_request(self):
        """Test first request is allowed."""
        from rate_limit import RateLimiter
        
        limiter = RateLimiter()
        
        # First request should be allowed
        result = limiter.check("test_key", rate_limit=100)
        assert result is True
    
    def test_rate_limiter_check_consumes_tokens(self):
        """Test check consumes tokens."""
        from rate_limit import RateLimiter
        
        limiter = RateLimiter()
        
        # Make request
        limiter.check("test_key", rate_limit=100)
        
        # Remaining should be 99
        remaining = limiter.get_remaining("test_key", rate_limit=100)
        assert remaining == 99
    
    def test_rate_limiter_rejects_when_exhausted(self):
        """Test request rejected when tokens exhausted."""
        from rate_limit import RateLimiter
        
        limiter = RateLimiter()
        
        # Exhaust tokens with 100 requests
        for _ in range(100):
            limiter.check("exhaust_key", rate_limit=100)
        
        # Next request should be rejected
        result = limiter.check("exhaust_key", rate_limit=100)
        assert result is False
    
    def test_rate_limiter_reset(self):
        """Test resetting rate limit."""
        from rate_limit import RateLimiter
        
        limiter = RateLimiter()
        
        # Exhaust tokens
        for _ in range(100):
            limiter.check("reset_key", rate_limit=100)
        
        # Should be exhausted
        assert limiter.check("reset_key", rate_limit=100) is False
        
        # Reset
        limiter.reset("reset_key")
        
        # Should have full tokens again
        remaining = limiter.get_remaining("reset_key", rate_limit=100)
        assert remaining == 100
    
    def test_rate_limiter_different_keys_independent(self):
        """Test different keys have independent limits."""
        from rate_limit import RateLimiter
        
        limiter = RateLimiter()
        
        # Exhaust one key
        for _ in range(100):
            limiter.check("key_a", rate_limit=100)
        
        # Other key should still work
        result = limiter.check("key_b", rate_limit=100)
        assert result is True
    
    def test_rate_limiter_custom_rate_limit(self):
        """Test different rate limits per key."""
        from rate_limit import RateLimiter
        
        limiter = RateLimiter()
        
        # Key with custom limit
        result = limiter.check("custom_key", rate_limit=50)
        assert result is True
        
        remaining = limiter.get_remaining("custom_key", rate_limit=50)
        assert remaining == 49
    
    def test_rate_limiter_cost_parameter(self):
        """Test custom cost parameter."""
        from rate_limit import RateLimiter
        
        limiter = RateLimiter()
        
        # Request with cost of 5
        result = limiter.check("cost_key", rate_limit=10, cost=5)
        assert result is True
        
        remaining = limiter.get_remaining("cost_key", rate_limit=10)
        assert remaining == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])