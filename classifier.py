KEYWORDS = {
    "invoice":  ["invoice", "bill to", "due date", "subtotal", "vendor", "gst", "tax invoice"],
    "receipt":  ["receipt", "thank you", "cashier", "change", "payment received", "pos"],
    "id_card":  ["date of birth", "dob", "passport", "nationality", "expiry", "id no", "licence"],
}

def classify(raw_text: str) -> str:
    text = raw_text.lower()
    scores = {
        doc: sum(1 for kw in kws if kw in text)
        for doc, kws in KEYWORDS.items()
    }
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "invoice"
