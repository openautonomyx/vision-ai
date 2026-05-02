"""
AutonomyX Vision AI — Authentication and API Keys

NOTE: This implementation uses in-memory storage. For production, use Redis or a database.
The current implementation stores raw keys for Phase 1 simplicity - update to hashed keys for production.
"""
import secrets
import hashlib
from datetime import datetime, timedelta
from typing import Optional
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

# In-memory store for API keys (use Redis or database for production)
# Format: {api_key: {name, created_at, expires_at, role, rate_limit, last_used}}
api_keys_store: dict = {}

ADMIN_BOOTSTRAP_NOTE = "For first-time setup, set ADMIN_BOOTSTRAP_TOKEN environment variable"

class APIKeyCreate(BaseModel):
    name: str
    role: str = "developer"  # admin, developer, viewer, agent
    rate_limit: int = 100  # requests per minute
    expires_in_days: Optional[int] = 90  # None for never

class APIKeyResponse(BaseModel):
    key: str
    name: str
    role: str
    rate_limit: int
    created_at: str
    expires_at: Optional[str]

class APIKeyInfo(BaseModel):
    name: str
    role: str
    rate_limit: int
    created_at: str
    expires_at: Optional[str]
    last_used: Optional[str]
    is_active: bool

# API key header
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def _hash_key(key: str) -> str:
    """Create a secure hash of the API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()

def create_api_key(data: APIKeyCreate) -> tuple[str, APIKeyResponse]:
    """Create a new API key."""
    key = f"ax_{secrets.token_urlsafe(32)}"
    
    now = datetime.utcnow()
    expires_at = None
    if data.expires_in_days:
        expires_at = (now + timedelta(days=data.expires_in_days)).isoformat()
    
    api_keys_store[key] = {
        "name": data.name,
        "role": data.role,
        "rate_limit": data.rate_limit,
        "created_at": now.isoformat(),
        "expires_at": expires_at,
        "last_used": None,
        "is_active": True
    }
    
    response = APIKeyResponse(
        key=key,
        name=data.name,
        role=data.role,
        rate_limit=data.rate_limit,
        created_at=now.isoformat(),
        expires_at=expires_at
    )
    
    return key, response

def get_api_key_info(key: str) -> Optional[APIKeyInfo]:
    """Get info about an API key without exposing the key."""
    if key not in api_keys_store:
        return None
    
    data = api_keys_store[key]
    return APIKeyInfo(
        name=data["name"],
        role=data["role"],
        rate_limit=data["rate_limit"],
        created_at=data["created_at"],
        expires_at=data["last_used"],
        last_used=data.get("last_used"),
        is_active=data["is_active"]
    )

def verify_api_key(key: str) -> Optional[dict]:
    """Verify an API key and return its metadata if valid."""
    if not key:
        return None
    
    # Check if key exists
    if key not in api_keys_store:
        return None
    
    data = api_keys_store[key]
    
    # Check if key is active
    if not data.get("is_active", False):
        return None
    
    # Check expiration
    if data.get("expires_at"):
        expires_at = datetime.fromisoformat(data["expires_at"])
        if datetime.utcnow() > expires_at:
            return None
    
    # Update last used
    data["last_used"] = datetime.utcnow().isoformat()
    
    return data

def revoke_api_key(key: str) -> bool:
    """Revoke an API key."""
    if key in api_keys_store:
        api_keys_store[key]["is_active"] = False
        return True
    return False

def list_api_keys() -> list[APIKeyInfo]:
    """List all API keys (without exposing the keys)."""
    result = []
    for key, data in api_keys_store.items():
        result.append(APIKeyInfo(
            name=data["name"],
            role=data["role"],
            rate_limit=data["rate_limit"],
            created_at=data["created_at"],
            expires_at=data["expires_at"],
            last_used=data.get("last_used"),
            is_active=data["is_active"]
        ))
    return result