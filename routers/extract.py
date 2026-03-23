import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import APIRouter, UploadFile, File, Form, Depends, BackgroundTasks, HTTPException
from auth import check_limit
from db import create_job, save_result, save_error, log_usage
from ocr import extract_text
from classifier import classify
from llm import extract_structured
from typing import Optional
import httpx

router = APIRouter()

ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/tiff", "application/pdf"}

def process(job_id: str, file_bytes: bytes, content_type: str, doc_type: str, webhook_url: str):
    try:
        raw_text = extract_text(file_bytes, content_type)
        resolved_type = classify(raw_text) if doc_type == "auto" else doc_type
        result = extract_structured(raw_text, resolved_type)
        save_result(job_id, result)
        if webhook_url:
            try:
                httpx.post(webhook_url, json={"job_id": job_id, "result": result}, timeout=10)
            except Exception:
                pass
    except Exception as e:
        save_error(job_id, str(e))

@router.post("/extract", status_code=202)
async def extract(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    doc_type: str = Form("auto"),
    webhook_url: Optional[str] = Form(None),
    user=Depends(check_limit),
):
    if file.content_type not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")

    file_bytes = await file.read()
    job = create_job(user["id"], doc_type, webhook_url)
    log_usage(user["id"], job["id"])
    background_tasks.add_task(
        process, job["id"], file_bytes, file.content_type, doc_type, webhook_url
    )
    return {"job_id": job["id"], "status": "processing"}
