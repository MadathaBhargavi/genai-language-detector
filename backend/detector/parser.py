"""
Document parsing module for GenAI Language Detector.
Extracts clean plain text from uploaded files: TXT, PDF, DOCX.
"""

import io
from typing import Tuple

def extract_text_from_file(filename: str, content_bytes: bytes) -> Tuple[bool, str, str]:
    """
    Extracts text from file bytes based on file extension.
    Returns: (success: bool, extracted_text: str, error_message: str)
    """
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    
    if ext == "txt":
        try:
            # Try UTF-8 first, fallback to latin-1
            try:
                text = content_bytes.decode("utf-8")
            except UnicodeDecodeError:
                text = content_bytes.decode("latin-1")
            return True, text.strip(), ""
        except Exception as e:
            return False, "", f"Failed to read TXT file: {str(e)}"
            
    elif ext == "pdf":
        try:
            from pypdf import PdfReader
            stream = io.BytesIO(content_bytes)
            reader = PdfReader(stream)
            extracted_pages = []
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    extracted_pages.append(page_text.strip())
            full_text = "\n\n".join(extracted_pages).strip()
            if not full_text:
                return False, "", "Could not extract text from this PDF (it may contain scanned images only)."
            return True, full_text, ""
        except Exception as e:
            return False, "", f"Failed to parse PDF: {str(e)}"

    elif ext in ["docx", "doc"]:
        try:
            import docx
            stream = io.BytesIO(content_bytes)
            doc = docx.Document(stream)
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = "\n\n".join(paragraphs).strip()
            if not full_text:
                return False, "", "No text found in DOCX file."
            return True, full_text, ""
        except Exception as e:
            return False, "", f"Failed to parse Word document: {str(e)}"

    else:
        return False, "", f"Unsupported file type (.{ext}). Supported formats: TXT, PDF, DOCX."
