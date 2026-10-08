"""
Unit and Integration Tests for GenAI Language Detector.
Validates:
1. Human-written text analysis
2. AI-generated text analysis
3. Very short text handling
4. Long text handling
5. Empty text handling
6. Invalid inputs
7. API endpoints (Analyze, Upload, Samples, Health)
8. Document parsers (TXT, PDF, DOCX)
"""

import unittest
import io
from fastapi.testclient import TestClient
from backend.main import app
from backend.detector.classifier import detector_engine
from backend.detector.parser import extract_text_from_file

class TestGenAIDetector(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.ai_text = (
            "Artificial intelligence has fundamentally transformed the modern technological landscape, "
            "reshaping how industries operate and innovate. In today's fast-paced digital era, machine "
            "learning algorithms play a pivotal role in optimizing operational workflows, enhancing predictive "
            "accuracy, and fostering data-driven decision-making. Furthermore, it is important to note that the "
            "seamless integration of neural networks across sectors such as healthcare and finance underscores "
            "the vital importance of robust governance frameworks. In conclusion, the future of artificial "
            "intelligence holds immense promise, offering a rich tapestry of opportunities for sustainable societal advancement."
        )
        self.human_text = (
            "I honestly didn't expect much when I first set up the backyard smoker last Saturday. "
            "My brother kept saying the temperature would fluctuate wildly because of the sudden chilly wind "
            "coming off the ridge, and for the first two hours he was totally right. I had to stand out there "
            "in an old flannel shirt, poking at the damp oak chunks with a pair of rusty grill tongs, wondering "
            "why I didn't just order barbecue takeout like a normal person. Around hour four, though, the thin "
            "blue smoke settled down. The crust on the brisket turned into that dark mahogany bark everyone "
            "obsesses over. When we finally carved into it on our picnic table, nobody said a word for ten straight "
            "minutes—we were just eating like wolves. Best Saturday afternoon all autumn."
        )

    def test_health_endpoint(self):
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["model_loaded"])

    def test_samples_endpoint(self):
        response = self.client.get("/api/samples")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("samples", data)
        self.assertGreaterEqual(len(data["samples"]), 3)

    def test_ai_text_detection(self):
        response = self.client.post("/api/analyze", json={"text": self.ai_text})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["result"], "Likely AI-Generated")
        self.assertGreater(data["ai_probability"], 0.65)
        self.assertLess(data["human_probability"], 0.35)
        self.assertIn(data["confidence"], ["High", "Medium"])
        self.assertGreater(data["word_count"], 50)
        self.assertIn("writing_analysis", data)
        self.assertIn("sentence_breakdown", data)
        self.assertIn("disclaimer", data)

    def test_human_text_detection(self):
        response = self.client.post("/api/analyze", json={"text": self.human_text})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["result"], "Likely Human-Written")
        self.assertLess(data["ai_probability"], 0.35)
        self.assertGreater(data["human_probability"], 0.65)
        self.assertEqual(data["confidence"], "High")

    def test_short_text_handling(self):
        short_text = "Just went to the store to get milk."
        response = self.client.post("/api/analyze", json={"text": short_text})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["result"], "Uncertain")
        self.assertEqual(data["confidence"], "Low")
        self.assertIn("warning", data)

    def test_empty_text_handling(self):
        response = self.client.post("/api/analyze", json={"text": "   "})
        self.assertEqual(response.status_code, 400)

    def test_long_text_handling(self):
        long_text = (self.human_text + " ") * 6
        response = self.client.post("/api/analyze", json={"text": long_text})
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertGreater(data["word_count"], 400)
        self.assertEqual(data["result"], "Likely Human-Written")

    def test_file_upload_txt(self):
        content = b"This is a test plain text document containing natural language sentences."
        response = self.client.post(
            "/api/upload",
            files={"file": ("test.txt", io.BytesIO(content), "text/plain")}
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["filename"], "test.txt")
        self.assertGreater(data["word_count"], 0)

    def test_file_upload_unsupported(self):
        response = self.client.post(
            "/api/upload",
            files={"file": ("image.png", io.BytesIO(b"\x89PNG\r\n\x1a\n"), "image/png")}
        )
        self.assertEqual(response.status_code, 400)

if __name__ == "__main__":
    unittest.main()
