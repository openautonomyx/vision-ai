"""
Unit tests for AutonomyX Vision AI - Authentication Module

Run with: pytest tests/test_auth.py -v
"""

import pytest
from datetime import datetime, timedelta
from unittest.mock import patch, MagicMock


class TestAuthModule:
    """Tests for auth.py module."""
    
    def test_create_api_key(self):
        """Test API key creation with valid data."""
        from auth import APIKeyCreate, create_api_key
        
        data = APIKeyCreate(name="test-key", role="developer", rate_limit=100)
        key, response = create_api_key(data)
        
        assert key.startswith("ax_")
        assert response.name == "test-key"
        assert response.role == "developer"
        assert response.rate_limit == 100
    
    def test_create_api_key_with_expiration(self):
        """Test API key creation with expiration."""
        from auth import APIKeyCreate, create_api_key
        
        data = APIKeyCreate(name="temp-key", role="developer", rate_limit=50, expires_in_days=7)
        key, response = create_api_key(data)
        
        assert key.startswith("ax_")
        assert response.expires_at is not None
    
    def test_verify_valid_api_key(self):
        """Test verification of valid API key."""
        from auth import APIKeyCreate, create_api_key, verify_api_key
        
        data = APIKeyCreate(name="test-key", role="developer", rate_limit=100)
        key, _ = create_api_key(data)
        
        result = verify_api_key(key)
        assert result is not None
        assert result["name"] == "test-key"
        assert result["role"] == "developer"
    
    def test_verify_invalid_api_key(self):
        """Test verification of invalid API key."""
        from auth import verify_api_key
        
        result = verify_api_key("invalid_key_123")
        assert result is None
    
    def test_verify_expired_api_key(self):
        """Test verification of expired API key."""
        from auth import APIKeyCreate, create_api_key, verify_api_key, api_keys_store
        
        # Create key that's already expired
        old_expiry = (datetime.utcnow() - timedelta(days=1)).isoformat()
        api_keys_store["expired_key"] = {
            "name": "expired",
            "role": "developer",
            "rate_limit": 100,
            "created_at": datetime.utcnow().isoformat(),
            "expires_at": old_expiry,
            "last_used": None,
            "is_active": True
        }
        
        result = verify_api_key("expired_key")
        assert result is None
    
    def test_revoke_api_key(self):
        """Test revoking an API key."""
        from auth import APIKeyCreate, create_api_key, verify_api_key, revoke_api_key
        
        data = APIKeyCreate(name="revokable-key", role="developer", rate_limit=100)
        key, _ = create_api_key(data)
        
        # Verify key works before revocation
        assert verify_api_key(key) is not None
        
        # Revoke the key
        result = revoke_api_key(key)
        assert result is True
        
        # Verify key no longer works after revocation
        assert verify_api_key(key) is None
    
    def test_list_api_keys(self):
        """Test listing all API keys."""
        from auth import APIKeyCreate, create_api_key, list_api_keys
        from auth import api_keys_store
        
        # Clear existing keys
        api_keys_store.clear()
        
        # Create a few keys
        create_api_key(APIKeyCreate(name="key1", role="developer", rate_limit=100))
        create_api_key(APIKeyCreate(name="key2", role="admin", rate_limit=200))
        
        keys = list_api_keys()
        
        assert len(keys) == 2
        key_names = [k.name for k in keys]
        assert "key1" in key_names
        assert "key2" in key_names
    
    def test_api_key_with_admin_role(self):
        """Test API key with admin role."""
        from auth import APIKeyCreate, create_api_key, verify_api_key
        
        data = APIKeyCreate(name="admin-key", role="admin", rate_limit=1000)
        key, response = create_api_key(data)
        
        result = verify_api_key(key)
        assert result["role"] == "admin"
    
    def test_api_key_with_viewer_role(self):
        """Test API key with viewer role."""
        from auth import APIKeyCreate, create_api_key, verify_api_key
        
        data = APIKeyCreate(name="viewer-key", role="viewer", rate_limit=50)
        key, response = create_api_key(data)
        
        result = verify_api_key(key)
        assert result["role"] == "viewer"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])