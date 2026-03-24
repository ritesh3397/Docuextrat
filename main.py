
import os, sys, uuid, json, re, io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from supabase import create_client
from datetime import datetime
from PIL import Image
from pdf2image import convert_from_bytes
import pytesseract
import httpx
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")

app = FastAPI(
    title="DocuExtract AI Engine",
    description="""
## 📄 Extract structured data from any document instantly!

**Supported Documents:**
- 🧾 Invoices
- 🧾 Receipts
- 🪪 ID Cards

**How to use:**
1. Click **Authorize** button → Enter your API Key
2. POST `/v1/extract` → Upload document
3. GET `/v1/results/{job_id}` → Get extracted JSON

**Test API Key:** `test-api-key-123`
    """,
    version="1.0.0",
    openapi_tags=[
        {"name": "Health", "description": "✅ Check if API is running"},
        {"name": "Extract", "description": "📤 Upload document for AI extraction"},
        {"name": "Results", "description": "📥 Get extracted JSON data"},
    ]
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)
TIER_LIMITS = {"free": 50, "starter": 500, "professional": 2000, "enterprise": 999999}
ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/tiff", "application/pdf"}

def get_supabase():
    return create_client(
        os.environ.get("SUPABASE_URL"),
        os.environ.get("SUPABASE_KEY")
    )

def get_user_by_api_key(api_key):
    supabase = get_supabase()
    res = supabase.table("users").select("*").eq("api_key", api_key).single().execute()
    return res.data

def get_monthly_usage(user_id):
    supabase = get_supabase()
    month = datetime.utcnow().strftime("%Y-%m")
    res = supabase.table("usage_logs").select("id", count="exact").eq("user_id", user_id).eq("month", month).execute()
    return res.count or 0

def create_job(user_id, doc_type, webhook_url=None):
    supabase = get_supabase()
    job = {
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "status": "processing",
        "doc_type": doc_type,
        "webhook_url": webhook_url,
        "created_at": datetime.utcnow().isoformat()
    }
    supabase.table("jobs").insert(job).execute()
    return job

def save_result(job_id, result):
    supabase = get_supabase()
    supabase.table("jobs").update({
        "status": "completed",
        "result": result,
        "completed_at": datetime.utcnow().isoformat()
    }).eq("id", job_id).execute()

def save_error(job_id, error):
    supabase = get_supabase()
    supabase.table("jobs").update({
        "status": "failed",
        "error_message": error
    }).eq("id", job_id).execute()

def get_job(job_id):
    supabase = get_supabase()
    res = supabase.table("jobs").select("*").eq("id", job_id).single().execute()
    return res.data

def log_usage(user_id, job_id):
    supabase = get_supabase()
    supabase.table("usage_logs").insert({
        "id": str(uuid.uuid4()),
        "user_id": user_id,
        "job_id": job_id,
        "month": datetime.utcnow().strftime("%Y-%m")
    }).execute()

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
        raise HTTPException(status_code=429, detail=f"Monthly limit of {limit} docs reached.")
    return user

def extract_text(file_bytes, content_type):
    if content_type == "application/pdf":
        pages = convert_from_bytes(file_bytes)
        return "\n".join(pytesseract.image_to_string(p, config="--oem 3 --psm 6") for p in pages)
    image = Image.open(io.BytesIO(file_bytes))
    return pytesseract.image_to_string(image, config="--oem 3 --psm 6")

KEYWORDS = {
    "invoice": ["invoice", "bill to", "due date", "subtotal", "vendor", "gst"],
    "receipt": ["receipt", "thank you", "cashier", "change", "payment received"],
    "id_card": ["date of birth", "dob", "passport", "nationality", "expiry"],
}

def classify(raw_text):
    text = raw_text.lower()
    scores = {doc: sum(1 for kw in kws if kw in text) for doc, kws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "invoice"

PROMPTS = {
    "invoice": "Extract vendor_name, customer_name, date (YYYY-MM-DD), invoice_number, currency, subtotal, tax_amount, total_amount, line_items (array of {description,quantity,unit_price,total}), confidence_score (0-1). Use null for missing. JSON only.",
    "receipt": "Extract merchant_name, date (YYYY-MM-DD), currency, total_amount, tax_amount, payment_method, line_items (array of {description,quantity,unit_price,total}), confidence_score (0-1). Use null for missing. JSON only.",
    "id_card": "Extract full_name, date_of_birth (YYYY-MM-DD), id_number, expiry_date (YYYY-MM-DD), nationality, confidence_score (0-1). Use null for missing. JSON only.",
}

def extract_structured(raw_text, doc_type):
    prompt = PROMPTS.get(doc_type, PROMPTS["invoice"])
    response = httpx.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {os.environ.get('GROQ_API_KEY')}"},
        json={
            "model": "llama-3.1-8b-instant",
            "messages": [{"role": "user", "content": f"{prompt}\n\nOCR TEXT:\n{raw_text[:4000]}"}],
            "temperature": 0.1
        },
        timeout=30,
    )
    response.raise_for_status()
    raw = response.json()["choices"][0]["message"]["content"]
    result = json.loads(re.sub(r"```(?:json)?|```", "", raw).strip())
    result["doc_type"] = doc_type
    return result

def process(job_id, file_bytes, content_type, doc_type, webhook_url):
    try:
        raw_text = extract_text(file_bytes, content_type)
        resolved = classify(raw_text) if doc_type == "auto" else doc_type
        result = extract_structured(raw_text, resolved)
        save_result(job_id, result)
        if webhook_url:
            try:
                httpx.post(webhook_url, json={"job_id": job_id, "result": result}, timeout=10)
            except Exception:
                pass
    except Exception as e:
        save_error(job_id, str(e))

@app.get("/v1/health", tags=["Health"])
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.post("/v1/extract", status_code=202, tags=["Extract"])
async def extract(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Upload invoice, receipt or ID card (JPG/PNG/PDF)"),
    doc_type: str = Form("auto", description="auto | invoice | receipt | id_card"),
    webhook_url: str = Form(None, description="Optional callback URL"),
    user=Depends(check_limit),
):
    if file.content_type not in ALLOWED:
        raise HTTPException(400, f"Unsupported file type: {file.content_type}")
    file_bytes = await file.read()
    job = create_job(user["id"], doc_type, webhook_url)
    log_usage(user["id"], job["id"])
    background_tasks.add_task(process, job["id"], file_bytes, file.content_type, doc_type, webhook_url)
    return {"job_id": job["id"], "status": "processing"}

@app.get("/v1/results/{job_id}", tags=["Results"])
def get_results(job_id: str, api_key: str = Security(API_KEY_HEADER)):
    user = get_current_user(api_key)
    job = get_job(job_id)
    if not job:
        raise HTTPException(404, "Job not found.")
    if job["user_id"] != user["id"]:
        raise HTTPException(403, "Access denied.")
    return {
        "job_id": job["id"],
        "status": job["status"],
        "doc_type": job.get("doc_type"),
        "result": job.get("result"),
        "error_message": job.get("error_message")
    }
from fastapi.responses import FileResponse

@app.get("/")
def serve_ui():
    return FileResponse("index.html")
