"""
AutonomyX Vision AI — Rate Limiting
"""
from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional
import threading

class RateLimiter:
    """Token bucket rate limiter per API key."""
    
    def __init__(self):
        self._buckets: dict[str, dict] = defaultdict(lambda: {
            "tokens": 0,
            "last_refill": datetime.utcnow()
        })
        self._lock = threading.Lock()
    
    def _refill(self, key: str, rate_limit: int):
        """Refill tokens based on time elapsed."""
        now = datetime.utcnow()
        bucket = self._buckets[key]
        last = bucket["last_refill"]
        
        # Calculate tokens to add (1 token per second / rate_limit * 60)
        seconds = (now - last).total_seconds()
        tokens_to_add = int(seconds * (rate_limit / 60))
        
        bucket["tokens"] = min(rate_limit, bucket["tokens"] + tokens_to_add)
        bucket["last_refill"] = now
    
    def check(self, key: str, rate_limit: int, cost: int = 1) -> bool:
        """Check if request is allowed and consume tokens."""
        with self._lock:
            self._refill(key, rate_limit)
            bucket = self._buckets[key]
            
            if bucket["tokens"] >= cost:
                bucket["tokens"] -= cost
                return True
            return False
    
    def get_remaining(self, key: str, rate_limit: int) -> int:
        """Get remaining tokens."""
        with self._lock:
            self._refill(key, rate_limit)
            return self._buckets[key]["tokens"]
    
    def reset(self, key: str):
        """Reset rate limit for a key."""
        with self._lock:
            if key in self._buckets:
                del self._buckets[key]

# Global rate limiter instance
rate_limiter = RateLimiter()