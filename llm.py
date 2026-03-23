import json
import re
import httpx
from config import GROQ_API_KEY

GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
MODEL    = "llama-3.1-8b-instant"

PROMPTS = {
    "invoice": """
Extract these fields from the invoice OCR text as JSON:
vendor_name, vendor_address, customer_name, date (YYYY-MM-DD),
invoice_number, currency (3-letter), subtotal (float), tax_amount (float),
total_amount (float), line_items (array of {description, quantity, unit_price, total}),
confidence_score (float 0-1).
Use null for missing fields. Respond ONLY with valid JSON.
""",
    "receipt": """
Extract these fields from the receipt OCR text as JSON:
merchant_name, date (YYYY-MM-DD), currency (3-letter), total_amount (float),
tax_amount (float), payment_method, line_items (array of {description, quantity, unit_price, total}),
confidence_score (float 0-1).
Use null for missing fields. Respond ONLY with valid JSON.
""",
    "id_card": """
Extract these fields from the ID document OCR text as JSON:
full_name, date_of_birth (YYYY-MM-DD), id_number, expiry_date (YYYY-MM-DD),
nationality, confidence_score (float 0-1).
Use null for missing fields. Respond ONLY with valid JSON.
""",
}

def extract_structured(raw_text: str, doc_type: str) -> dict:
    system_prompt = PROMPTS.get(doc_type, PROMPTS["invoice"])
    user_message  = f"{system_prompt}\n\nOCR TEXT:\n{raw_text[:4000]}"

    response = httpx.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {GROQ_API_KEY}"},
        json={
            "model": MODEL,
            "messages": [{"role": "user", "content": user_message}],
            "temperature": 0.1,
        },
        timeout=30,
    )
    response.raise_for_status()

    raw = response.json()["choices"][0]["message"]["content"]
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()
    result = json.loads(cleaned)
    result["doc_type"] = doc_type
    return result
