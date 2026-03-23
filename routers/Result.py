import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, HTTPException, Security
from fastapi.security.api_key import APIKeyHeader
from auth import get_current_user
from db import get_job

router = APIRouter()
API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)

@router.get("/results/{job_id}")
def get_results(job_id: str, api_key: str = Security(API_KEY_HEADER)):
    user = get_current_user(api_key)
    job  = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job["user_id"] != user["id"]:
        raise HTTPException(403, "Access denied.")
    return {
        "job_id":        job["id"],
        "status":        job["status"],
        "doc_type":      job.get("doc_type"),
        "result":        job.get("result"),
        "error_message": job.get("error_message"),
    }
