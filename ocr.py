import pytesseract
from PIL import Image
import io
from pdf2image import convert_from_bytes

def extract_text(file_bytes: bytes, content_type: str) -> str:
    if content_type == "application/pdf":
        pages = convert_from_bytes(file_bytes)
        all_text = []
        for page in pages:
            text = pytesseract.image_to_string(page, config="--oem 3 --psm 6")
            all_text.append(text)
        return "\n".join(all_text)
    else:
        image = Image.open(io.BytesIO(file_bytes))
        return pytesseract.image_to_string(image, config="--oem 3 --psm 6")
