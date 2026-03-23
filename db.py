from supabase import create_client
from config import SUPABASE_URL, SUPABASE_KEY
import uuid
from datetime import datetime

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

def get_user_by_api_key(api_key: str):
    res = supabase.table("users").select("*").eq("api_key", api_key).single().execute()
    return res.data

def get_monthly_usage(user_id: str) -> int:
    month = datetime.utcnow().strftime("%Y-%m")
    res = supabase.table("usage_logs") \
        .select("id", count="exact") \
        .eq("user_id", user_id) \
        .eq("month", month) \
        .execute()
    return res.count or 0

def create_job(user_id: str, doc_type: str, webhook_url: str = None) -> dict:
    job = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "status": "processing",
        "doc_type": doc_type,
        "webhook_url": webhook_url,
        "created_at": datetime.utcnow().isoformat(),
    }
    supabase.table("jobs").insert(job).execute()
    return job

def save_result(job_id: str, result: dict):
    supabase.table("jobs").update({
        "status": "completed",
        "result": result,
        "completed_at": datetime.utcnow().isoformat(),
    }).eq("id", job_id).execute()

def save_error(job_id: str, error: str):
    supabase.table("jobs").update({
        "status": "failed",
        "error_message": error,
    }).eq("id", job_id).execute()

def get_job(job_id: str):
    res = supabase.table("jobs").select("*").eq("id", job_id).single().execute()
    return res.data

def log_usage(user_id: str, job_id: str):
    supabase.table("usage_logs").insert({
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "job_id": job_id,
        "month": datetime.utcnow().strftime("%Y-%m"),
    }).execute()
