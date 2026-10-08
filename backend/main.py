"""
FastAPI application for GenAI Language Detector.
Provides REST API endpoints for text analysis, document extraction,
sample corpora, and static frontend hosting.
"""

import os
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional

from backend.detector.classifier import detector_engine
from backend.detector.parser import extract_text_from_file

app = FastAPI(
    title="GenAI Language Detector API",
    description="Production-grade stylometric and ML-powered AI text detection engine.",
    version="1.0.0"
)

# Enable CORS for external frontend or local dev servers
cors_origins_env = os.environ.get("ALLOWED_ORIGINS", "*")
origins = ["*"] if cors_origins_env == "*" else [orig.strip() for orig in cors_origins_env.split(",") if orig.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="The written text to analyze.")

class AnalyzeResponse(BaseModel):
    valid: bool
    result: str
    ai_probability: float
    human_probability: float
    confidence: str
    word_count: int
    sentence_count: int
    average_sentence_length: float
    writing_analysis: dict
    sentence_breakdown: list
    signals: list
    disclaimer: str
    warning: Optional[str] = None

@app.get("/api/health")
def health_check():
    """Health check endpoint providing model and service status."""
    return {
        "status": "healthy",
        "service": "GenAI Language Detector",
        "version": "1.0.0",
        "model_loaded": detector_engine.model_payload is not None
    }

@app.post("/api/analyze")
def analyze_text(payload: AnalyzeRequest):
    """
    Analyze written text and estimate authorship probabilities,
    writing metrics, and sentence-level signals.
    """
    raw_text = payload.text
    if not raw_text or not raw_text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty. Please enter text to analyze.")
        
    result = detector_engine.analyze(raw_text)
    if not result.get("valid", True) and "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])
        
    return result

@app.post("/api/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload and extract text from TXT, PDF, or DOCX documents.
    """
    filename = file.filename or "uploaded_file"
    max_size = 10 * 1024 * 1024  # 10MB limit
    
    content = await file.read()
    if len(content) > max_size:
        raise HTTPException(status_code=400, detail="File size exceeds maximum allowed limit (10MB).")
        
    success, extracted_text, err_msg = extract_text_from_file(filename, content)
    if not success:
        raise HTTPException(status_code=400, detail=err_msg)
        
    words = extracted_text.split()
    return {
        "success": True,
        "filename": filename,
        "text": extracted_text,
        "word_count": len(words),
        "character_count": len(extracted_text)
    }

@app.get("/api/samples")
def get_sample_texts():
    """Provide realistic sample texts for immediate testing."""
    return {
        "samples": [
            {
                "id": "human_sample",
                "label": "Human Written (Authentic Story)",
                "category": "Human",
                "text": "I honestly didn't expect much when I first set up the backyard smoker last Saturday. My brother kept saying the temperature would fluctuate wildly because of the sudden chilly wind coming off the ridge, and for the first two hours he was totally right. I had to stand out there in an old flannel shirt, poking at the damp oak chunks with a pair of rusty grill tongs, wondering why I didn't just order barbecue takeout like a normal person. Around hour four, though, the thin blue smoke settled down. The crust on the brisket turned into that dark mahogany bark everyone obsesses over. When we finally carved into it on our picnic table, nobody said a word for ten straight minutes—we were just eating like wolves. Best Saturday afternoon all autumn."
            },
            {
                "id": "ai_sample",
                "label": "AI Generated (ChatGPT / LLM Style)",
                "category": "AI",
                "text": "Artificial intelligence has fundamentally transformed the modern technological landscape, reshaping how industries operate and innovate. In today's fast-paced digital era, machine learning algorithms play a pivotal role in optimizing operational workflows, enhancing predictive accuracy, and fostering data-driven decision-making. Furthermore, it is important to note that the seamless integration of neural networks across sectors such as healthcare and finance underscores the vital importance of robust governance frameworks. As organizations continue to navigate this dynamic paradigm, striking a harmonious balance between rapid innovation and ethical considerations remains paramount. In conclusion, the future of artificial intelligence holds immense promise, offering a rich tapestry of opportunities for sustainable societal advancement."
            },
            {
                "id": "mixed_sample",
                "label": "Mixed / AI-Assisted (Hybrid Writing)",
                "category": "Mixed",
                "text": "Last Tuesday, our engineering squad debated how to handle sudden spikes in our checkout webhook. I was arguing that we should stick with simple database row locking because nobody wanted to introduce another message broker before our November release. However, integrating a distributed pub-sub architecture ensures that messages are processed asynchronously without overwhelming database connection pools. Moreover, it is crucial to recognize that implementing exponential backoff strategies mitigates transient network failures. We ended up writing a quick Redis proof-of-concept over lukewarm coffee."
            }
        ]
    }

# Mount static frontend directory if it exists
frontend_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "frontend")
if os.path.exists(frontend_path):
    app.mount("/", StaticFiles(directory=frontend_path, html=True), name="frontend")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    uvicorn.run("backend.main:app", host=host, port=port)
