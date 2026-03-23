from fastapi import HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from db import get_user_by_api_key, get_monthly_usage

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)

TIER_LIMITS = {
    "free":         50,
    "starter":      500,
    "professional": 2000,
    "enterprise":   999999,
}

def get_current_user(api_key: str = Security(API_KEY_HEADER)):
    user = get_user_by_api_key(api_key)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid API key.")
    return user

def check_limit(api_key: str = Security(API_KEY_HEADER)):
    user = get_current_user(api_key)
    limit = TIER_LIMITS.get(user.get("tier", "free"), 50)
    usage = get_monthly_usage(user["id"])
    if usage >= limit:
        raise HTTPException(
            status_code=429,
            detail=f"Monthly limit of {limit} docs reached. Upgrade your plan."
        )
    return user
