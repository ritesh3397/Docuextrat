from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import extract, results, health

app = FastAPI(
    title="DocuExtract AI Engine",
    description="Extract structured JSON from invoices, receipts & ID documents.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, prefix="/v1", tags=["Health"])
app.include_router(extract.router, prefix="/v1", tags=["Extract"])
app.include_router(results.router, prefix="/v1", tags=["Results"])
