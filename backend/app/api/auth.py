"""
=============================================================
 COMPUTE BAZAAR — Security & Authentication (Weeks 3-8)
=============================================================
 Implements:
   - Secret API key checks for agents
   - JWT token creation and verification
   - FastAPI Security dependency injections
=============================================================
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader, HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt
from sqlalchemy.orm import Session

from backend.app.db.models import Agent, SessionLocal

# Auth configuration — raise at startup when in production with unset key
_SECRET_KEY_ENV = os.getenv("SECRET_KEY")
_DEBUG = os.getenv("DEBUG", "true").lower() in ("1", "true", "yes")
_HARDCODED_FALLBACK = "computebazaar_super_secret_key_12345"

if not _SECRET_KEY_ENV and not _DEBUG:
    raise ValueError(
        "SECRET_KEY environment variable is not set and DEBUG=false. "
        "Set a strong, random SECRET_KEY before deploying to production."
    )

SECRET_KEY: str = _SECRET_KEY_ENV or _HARDCODED_FALLBACK
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 1440  # 24 hours

# Security schemas
bearer_scheme = HTTPBearer(auto_error=False)
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


# ── Dependency: Get Database Session ─────────────────────────
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# ── Token utilities ──────────────────────────────────────────
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None) -> str:
    """Generate a JWT token for the authenticated agent."""
    to_encode = data.copy()
    now_utc = datetime.now(timezone.utc)
    if expires_delta:
        expire = now_utc + expires_delta
    else:
        expire = now_utc + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# ── Security Dependency: Authenticate Agent ──────────────────
def get_current_agent(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    db: Session = Depends(get_db)
) -> Agent:
    """
    Authenticate agent via Bearer JWT token or X-API-Key header.
    Throws HTTP 401 if authentication fails.
    """
    agent_name: Optional[str] = None
    
    # Method 1: JWT token check
    if credentials:
        try:
            token = credentials.credentials
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            agent_name = payload.get("sub")
            if agent_name is None:
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Invalid authentication token payload",
                    headers={"WWW-Authenticate": "Bearer"},
                )
        except JWTError:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Could not validate credentials",
                headers={"WWW-Authenticate": "Bearer"},
            )
            
        # Retrieve agent by name
        agent = db.query(Agent).filter(Agent.name == agent_name, Agent.is_active == True).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Agent not found or inactive",
            )
        return agent

    # Method 2: API Key check
    if api_key:
        agent = db.query(Agent).filter(Agent.api_key == api_key, Agent.is_active == True).first()
        if not agent:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or inactive API Key",
            )
        return agent

    # No credentials provided
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Not authenticated. Provide Bearer Token or X-API-Key header.",
        headers={"WWW-Authenticate": "Bearer"},
    )
