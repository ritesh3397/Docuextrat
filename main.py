import os, sys, uuid, json, re, io
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks, Security, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from fastapi.responses import HTMLResponse
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
    description="Extract structured JSON from invoices, receipts & ID documents.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url=None,
)

app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=True)
TIER_LIMITS = {"free": 50, "starter": 500, "professional": 2000, "enterprise": 999999}
ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/tiff", "application/pdf"}

def get_supabase():
    return create_client(os.environ.get("SUPABASE_URL"), os.environ.get("SUPABASE_KEY"))

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
    job = {"id": str(uuid.uuid4()), "user_id": user_id, "status": "processing",
           "doc_type": doc_type, "webhook_url": webhook_url, "created_at": datetime.utcnow().isoformat()}
    supabase.table("jobs").insert(job).execute()
    return job

def save_result(job_id, result):
    supabase = get_supabase()
    supabase.table("jobs").update({"status": "completed", "result": result,
        "completed_at": datetime.utcnow().isoformat()}).eq("id", job_id).execute()

def save_error(job_id, error):
    supabase = get_supabase()
    supabase.table("jobs").update({"status": "failed", "error_message": error}).eq("id", job_id).execute()

def get_job(job_id):
    supabase = get_supabase()
    res = supabase.table("jobs").select("*").eq("id", job_id).single().execute()
    return res.data

def log_usage(user_id, job_id):
    supabase = get_supabase()
    supabase.table("usage_logs").insert({"id": str(uuid.uuid4()), "user_id": user_id,
        "job_id": job_id, "month": datetime.utcnow().strftime("%Y-%m")}).execute()

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
        json={"model": "llama-3.1-8b-instant",
              "messages": [{"role": "user", "content": f"{prompt}\n\nOCR TEXT:\n{raw_text[:4000]}"}],
              "temperature": 0.1},
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

# ── UI ────────────────────────────────────────────────────────────────────────
UI_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>DocuExtract — AI Document Parser</title>
<link href="https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@400;500&display=swap" rel="stylesheet">
<style>
*,*::before,*::after{box-sizing:border-box;margin:0;padding:0}
:root{--bg:#0a0e12;--surface:#111820;--border:#1e2a35;--teal:#00d4b8;--teal-dim:rgba(0,212,184,0.12);--text:#e8f0f5;--muted:#5a7080;--error:#ff4d6d}
body{font-family:'Syne',sans-serif;background:var(--bg);color:var(--text);min-height:100vh;display:flex;flex-direction:column;align-items:center;padding:40px 20px;background-image:radial-gradient(ellipse 60% 40% at 50% -10%,rgba(0,212,184,0.08) 0%,transparent 70%)}
header{text-align:center;margin-bottom:48px;animation:fadeDown 0.6s ease both}
.logo{display:inline-flex;align-items:center;gap:10px;margin-bottom:16px}
.logo-icon{width:40px;height:40px;background:var(--teal);border-radius:10px;display:grid;place-items:center;font-size:20px}
.logo-text{font-size:22px;font-weight:800;letter-spacing:-0.5px}
.logo-text span{color:var(--teal)}
h1{font-size:clamp(28px,5vw,48px);font-weight:800;line-height:1.1;letter-spacing:-1px;margin-bottom:12px}
h1 em{font-style:normal;color:var(--teal)}
.subtitle{color:var(--muted);font-size:15px}
.card{width:100%;max-width:620px;background:var(--surface);border:1px solid var(--border);border-radius:20px;padding:32px;animation:fadeUp 0.6s ease 0.1s both}
.field{margin-bottom:20px}
label{display:block;font-size:11px;font-weight:600;text-transform:uppercase;letter-spacing:1px;color:var(--muted);margin-bottom:8px}
input[type=text],select{width:100%;background:var(--bg);border:1px solid var(--border);border-radius:10px;padding:12px 16px;color:var(--text);font-family:'DM Mono',monospace;font-size:13px;outline:none;transition:border-color 0.2s}
input[type=text]:focus,select:focus{border-color:var(--teal)}
select option{background:var(--surface)}
.dropzone{border:2px dashed var(--border);border-radius:14px;padding:40px 20px;text-align:center;cursor:pointer;transition:all 0.2s;position:relative;margin-bottom:20px}
.dropzone:hover,.dropzone.drag{border-color:var(--teal);background:var(--teal-dim)}
.dropzone input{position:absolute;inset:0;opacity:0;cursor:pointer;width:100%;height:100%}
.drop-icon{font-size:36px;margin-bottom:12px;display:block}
.drop-title{font-size:16px;font-weight:700;margin-bottom:6px}
.drop-sub{font-size:12px;color:var(--muted)}
.file-name{margin-top:12px;font-family:'DM Mono',monospace;font-size:12px;color:var(--teal);display:none}
.formats{display:flex;gap:8px;flex-wrap:wrap;justify-content:center;margin-top:12px}
.tag{font-size:11px;font-weight:600;padding:3px 10px;background:var(--border);border-radius:6px;color:var(--muted);font-family:'DM Mono',monospace}
button{width:100%;padding:16px;background:var(--teal);color:#0a0e12;border:none;border-radius:12px;font-family:'Syne',sans-serif;font-size:15px;font-weight:700;cursor:pointer;transition:all 0.2s}
button:hover{opacity:0.9;transform:translateY(-1px)}
button:disabled{opacity:0.4;cursor:not-allowed;transform:none}
.status{margin-top:20px;padding:14px 18px;border-radius:10px;font-size:13px;font-weight:600;display:none;align-items:center;gap:10px}
.status.loading{display:flex;background:rgba(0,212,184,0.08);border:1px solid rgba(0,212,184,0.2);color:var(--teal)}
.status.error{display:flex;background:rgba(255,77,109,0.08);border:1px solid rgba(255,77,109,0.2);color:var(--error)}
.spinner{width:16px;height:16px;border:2px solid rgba(0,212,184,0.3);border-top-color:var(--teal);border-radius:50%;animation:spin 0.8s linear infinite;flex-shrink:0}
.result-card{width:100%;max-width:620px;margin-top:20px;background:var(--surface);border:1px solid var(--border);border-radius:20px;overflow:hidden;display:none;animation:fadeUp 0.4s ease both}
.result-header{display:flex;align-items:center;justify-content:space-between;padding:16px 24px;border-bottom:1px solid var(--border)}
.result-title{font-size:13px;font-weight:700;text-transform:uppercase;letter-spacing:1px;color:var(--teal)}
.copy-btn{width:auto;padding:7px 16px;font-size:12px;background:rgba(0,212,184,0.12);color:var(--teal);border:1px solid rgba(0,212,184,0.3);border-radius:8px}
.copy-btn:hover{background:rgba(0,212,184,0.2);transform:none}
.result-body{padding:24px;overflow-x:auto}
pre{font-family:'DM Mono',monospace;font-size:12px;line-height:1.7;color:#a8c0cc;white-space:pre-wrap;word-break:break-all}
.confidence{display:inline-flex;align-items:center;gap:6px;background:rgba(0,212,184,0.12);border:1px solid rgba(0,212,184,0.2);border-radius:20px;padding:4px 12px;font-size:12px;font-weight:600;color:var(--teal);margin-top:16px}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes fadeDown{from{opacity:0;transform:translateY(-20px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeUp{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
</style>
</head>
<body>
<header>
  <div class="logo"><div class="logo-icon">📄</div><div class="logo-text">Docu<span>Extract</span></div></div>
  <h1>Upload any doc.<br><em>Get clean JSON.</em></h1>
  <p class="subtitle">Invoice • Receipt • ID Card → Structured data in seconds</p>
</header>
<div class="card">
  <div class="field">
    <label>API Key</label>
    <input type="text" id="apiKey" placeholder="Enter your API key" value="test-api-key-123"/>
  </div>
  <div class="field">
    <label>Document Type</label>
    <select id="docType">
      <option value="auto">🔍 Auto Detect</option>
      <option value="invoice">🧾 Invoice</option>
      <option value="receipt">🛒 Receipt</option>
      <option value="id_card">🪪 ID Card</option>
    </select>
  </div>
  <div class="dropzone" id="dropzone">
    <input type="file" id="fileInput" accept=".jpg,.jpeg,.png,.webp,.tiff,.pdf"/>
    <span class="drop-icon">📂</span>
    <div class="drop-title">Drop your file here</div>
    <div class="drop-sub">or click to browse</div>
    <div class="formats"><span class="tag">JPG</span><span class="tag">PNG</span><span class="tag">PDF</span><span class="tag">WEBP</span><span class="tag">TIFF</span></div>
    <div class="file-name" id="fileName"></div>
  </div>
  <button id="extractBtn" onclick="extractDoc()" disabled>Extract JSON →</button>
  <div class="status loading" id="statusLoading"><div class="spinner"></div><span id="statusText">Processing...</span></div>
  <div class="status error" id="statusError"><span>⚠️</span><span id="errorText">Something went wrong.</span></div>
</div>
<div class="result-card" id="resultCard">
  <div class="result-header">
    <div class="result-title">✅ Extracted Data</div>
    <button class="copy-btn" onclick="copyJSON()">Copy JSON</button>
  </div>
  <div class="result-body">
    <pre id="resultJson"></pre>
    <div class="confidence" id="confidenceBadge" style="display:none"></div>
  </div>
</div>
<script>
const API_BASE="";
const fileInput=document.getElementById('fileInput');
const dropzone=document.getElementById('dropzone');
const extractBtn=document.getElementById('extractBtn');
fileInput.addEventListener('change',()=>{if(fileInput.files[0]){document.getElementById('fileName').textContent='📎 '+fileInput.files[0].name;document.getElementById('fileName').style.display='block';extractBtn.disabled=false;}});
dropzone.addEventListener('dragover',e=>{e.preventDefault();dropzone.classList.add('drag');});
dropzone.addEventListener('dragleave',()=>dropzone.classList.remove('drag'));
dropzone.addEventListener('drop',e=>{e.preventDefault();dropzone.classList.remove('drag');fileInput.files=e.dataTransfer.files;fileInput.dispatchEvent(new Event('change'));});
async function extractDoc(){
  const apiKey=document.getElementById('apiKey').value.trim();
  const docType=document.getElementById('docType').value;
  const file=fileInput.files[0];
  if(!apiKey||!file)return;
  showLoading('Uploading document...');
  document.getElementById('resultCard').style.display='none';
  extractBtn.disabled=true;
  try{
    const fd=new FormData();
    fd.append('file',file);fd.append('doc_type',docType);
    const r=await fetch('/v1/extract',{method:'POST',headers:{'X-API-Key':apiKey},body:fd});
    if(!r.ok){const e=await r.json();throw new Error(e.detail||'Upload failed');}
    const {job_id}=await r.json();
    showLoading('AI extracting data...');
    const result=await pollResult(job_id,apiKey);
    showResult(result);
  }catch(e){showError(e.message);}
  finally{extractBtn.disabled=false;}
}
async function pollResult(jobId,apiKey,attempts=0){
  if(attempts>20)throw new Error('Timeout — try again');
  await new Promise(r=>setTimeout(r,2000));
  const r=await fetch('/v1/results/'+jobId,{headers:{'X-API-Key':apiKey}});
  if(!r.ok)throw new Error('Failed to fetch result');
  const data=await r.json();
  if(data.status==='completed')return data;
  if(data.status==='failed')throw new Error(data.error_message||'Failed');
  showLoading('Processing... ('+( attempts+1)+')');
  return pollResult(jobId,apiKey,attempts+1);
}
function showResult(data){
  hideLoading();
  const result=data.result||{};
  document.getElementById('resultJson').textContent=JSON.stringify(result,null,2);
  const score=result.confidence_score;
  const cb=document.getElementById('confidenceBadge');
  if(score!==undefined){cb.textContent='⚡ Confidence: '+Math.round(score*100)+'%';cb.style.display='inline-flex';}
  document.getElementById('resultCard').style.display='block';
}
function copyJSON(){
  navigator.clipboard.writeText(document.getElementById('resultJson').textContent).then(()=>{
    const b=document.querySelector('.copy-btn');b.textContent='Copied! ✓';setTimeout(()=>b.textContent='Copy JSON',2000);
  });
}
function showLoading(msg){document.getElementById('statusLoading').style.display='flex';document.getElementById('statusText').textContent=msg;document.getElementById('statusError').style.display='none';}
function hideLoading(){document.getElementById('statusLoading').style.display='none';}
function showError(msg){hideLoading();document.getElementById('statusError').style.display='flex';document.getElementById('errorText').textContent=msg;}
</script>
</body>
</html>"""

# ── Routes ────────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse, include_in_schema=False)
def serve_ui():
    return HTMLResponse(content=UI_HTML)

@app.get("/v1/health", tags=["Health"])
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}

@app.post("/v1/extract", status_code=202, tags=["Extract"])
async def extract(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(..., description="Upload invoice, receipt or ID card (JPG/PNG/PDF)"),
    doc_type: str = Form("auto", description="auto | invoice | receipt | id_card"),
    webhook_url: str = Form(None),
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
    return {"job_id": job["id"], "status": job["status"], "doc_type": job.get("doc_type"),
            "result": job.get("result"), "error_message": job.get("error_message")}
