"""
Production inference classifier for GenAI Language Detector.
Loads the serialized model and performs multi-layer linguistic analysis,
probability calibration, writing metrics, and sentence-level heatmapping.
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, List

from .features import (
    extract_features,
    get_feature_vector,
    split_into_sentences,
    tokenize_words,
    FEATURE_NAMES
)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_artifact.joblib")

class DetectorEngine:
    def __init__(self):
        self.model_payload = None
        self.scaler = None
        self.classifier = None
        self.thresholds = {
            "ai_threshold": 0.65,
            "human_threshold": 0.35,
            "min_words_reliable": 30
        }
        self._load_model()

    def _load_model(self):
        if not os.path.exists(MODEL_PATH):
            from .trainer import train_and_export_model
            self.model_payload = train_and_export_model()
        else:
            self.model_payload = joblib.load(MODEL_PATH)
            
        self.scaler = self.model_payload["scaler"]
        self.classifier = self.model_payload["classifier"]
        if "thresholds" in self.model_payload:
            self.thresholds = self.model_payload["thresholds"]

    def analyze_sentence(self, sentence: str) -> Dict[str, Any]:
        """Analyze an individual sentence for the sentence-level breakdown map."""
        words = tokenize_words(sentence)
        if len(words) < 4:
            return {
                "text": sentence,
                "label": "neutral",
                "ai_prob": 0.5,
                "word_count": len(words)
            }
        
        sent_feats = extract_features(sentence)
        vec = get_feature_vector(sent_feats)
        vec_scaled = self.scaler.transform([vec])
        prob_ai = float(self.classifier.predict_proba(vec_scaled)[0][1])
        
        # Sentence level classification
        if prob_ai >= 0.60:
            label = "likely_ai"
        elif prob_ai <= 0.40:
            label = "likely_human"
        else:
            label = "mixed"
            
        return {
            "text": sentence,
            "label": label,
            "ai_prob": round(prob_ai, 2),
            "word_count": len(words)
        }

    def analyze(self, text: str) -> Dict[str, Any]:
        """
        Analyze document text and return comprehensive detection & writing metrics.
        """
        clean_text = text.strip() if text else ""
        words = tokenize_words(clean_text)
        sentences = split_into_sentences(clean_text)
        word_count = len(words)
        sentence_count = len(sentences)
        
        # 1. Validation for empty text
        if word_count == 0:
            return {
                "error": "Empty text provided. Please enter some text to analyze.",
                "valid": False
            }
            
        # Extract features
        feats = extract_features(clean_text)
        
        # 2. Check for very short text
        if word_count < 15:
            return {
                "valid": True,
                "result": "Uncertain",
                "ai_probability": 0.50,
                "human_probability": 0.50,
                "confidence": "Low",
                "word_count": word_count,
                "sentence_count": sentence_count,
                "average_sentence_length": feats["avg_sentence_length"],
                "warning": "Text is too short for a reliable linguistic analysis. Please enter at least 30-50 words for meaningful results.",
                "writing_analysis": self._build_writing_analysis(feats),
                "sentence_breakdown": [{"text": s, "label": "neutral", "ai_prob": 0.5, "word_count": len(tokenize_words(s))} for s in sentences],
                "signals": ["Text length is insufficient for stylometric statistical inference."]
            }

        # Predict with calibrated ML model
        vec = get_feature_vector(feats)
        vec_scaled = self.scaler.transform([vec])
        raw_prob_ai = float(self.classifier.predict_proba(vec_scaled)[0][1])
        
        # Length smoothing for texts between 15 and 30 words
        if word_count < self.thresholds["min_words_reliable"]:
            # Shrink toward 0.5 (uncertainty prior)
            shrinkage = (word_count - 15) / 15.0
            prob_ai = 0.5 + (raw_prob_ai - 0.5) * (0.6 * shrinkage)
        else:
            prob_ai = raw_prob_ai

        prob_ai = max(0.01, min(0.99, prob_ai))
        prob_human = 1.0 - prob_ai

        # Determine Result Type based on documented thresholds
        # AI threshold: >= 0.65
        # Human threshold: <= 0.35
        # Mixed: between 0.35 and 0.65
        # Uncertain: if very short or conflicting indicators
        if word_count < 25:
            result = "Uncertain"
            confidence = "Low"
        elif prob_ai >= self.thresholds["ai_threshold"]:
            result = "Likely AI-Generated"
            confidence = "High" if (prob_ai >= 0.80 and word_count >= 50) else "Medium"
        elif prob_ai <= self.thresholds["human_threshold"]:
            result = "Likely Human-Written"
            confidence = "High" if (prob_ai <= 0.20 and word_count >= 50) else "Medium"
        else:
            result = "Mixed / AI-Assisted"
            confidence = "Medium" if word_count >= 50 else "Low"

        # Sentence-by-sentence analysis
        sentence_breakdown = [self.analyze_sentence(s) for s in sentences]
        
        # Writing analysis metrics
        writing_analysis = self._build_writing_analysis(feats)
        
        # Diagnostic signals explaining the decision
        signals = self._generate_signals(feats, prob_ai)

        return {
            "valid": True,
            "result": result,
            "ai_probability": round(prob_ai, 3),
            "human_probability": round(prob_human, 3),
            "confidence": confidence,
            "word_count": word_count,
            "sentence_count": sentence_count,
            "average_sentence_length": round(feats["avg_sentence_length"], 1),
            "writing_analysis": writing_analysis,
            "sentence_breakdown": sentence_breakdown,
            "signals": signals,
            "disclaimer": "AI detection is probabilistic. This result is an estimate based on linguistic patterns and should not be considered definitive proof of authorship."
        }

    def _build_writing_analysis(self, feats: Dict[str, Any]) -> Dict[str, Any]:
        """Categorize writing metrics into clear levels and scores."""
        # Vocabulary diversity based on Type-Token Ratio
        ttr = feats["ttr"]
        if ttr >= 0.70:
            vocab_level = "High"
        elif ttr >= 0.50:
            vocab_level = "Medium"
        else:
            vocab_level = "Low"

        # Repetition based on bigram repetition
        bigram_rep = feats["bigram_repetition"]
        if bigram_rep >= 0.25:
            rep_level = "High"
        elif bigram_rep >= 0.12:
            rep_level = "Medium"
        else:
            rep_level = "Low"

        # Sentence variation based on Burstiness Coefficient of Variation
        b_cv = feats["burstiness_cv"]
        if b_cv >= 0.45:
            var_level = "High"
        elif b_cv >= 0.25:
            var_level = "Medium"
        else:
            var_level = "Low"

        # Writing consistency: low burstiness and low adjacent delta indicates high consistency (typical of AI)
        delta = feats["avg_adjacent_delta"]
        if delta < 5.0 and b_cv < 0.30:
            consistency_level = "High"  # Uniform, consistent cadence
        elif delta < 9.0:
            consistency_level = "Medium"
        else:
            consistency_level = "Varied"

        # Readability classification
        fre = feats["flesch_reading_ease"]
        if fre >= 70:
            readability_desc = "Standard / Easy to Read"
        elif fre >= 50:
            readability_desc = "Moderate / Professional"
        else:
            readability_desc = "Complex / Academic"

        return {
            "words": feats["word_count"],
            "sentences": feats["sentence_count"],
            "average_sentence_length": round(feats["avg_sentence_length"], 1),
            "vocabulary_diversity": vocab_level,
            "vocabulary_diversity_score": round(ttr * 100, 1),
            "repetition": rep_level,
            "repetition_score": round(bigram_rep * 100, 1),
            "sentence_variation": var_level,
            "sentence_variation_std": round(feats["sentence_len_std"], 1),
            "writing_consistency": consistency_level,
            "readability_score": round(fre, 1),
            "readability_level": readability_desc,
            "perplexity_proxy": round(feats["perplexity"], 1)
        }

    def _generate_signals(self, feats: Dict[str, Any], prob_ai: float) -> List[str]:
        """Generate human-interpretable linguistic signals that influenced the detector."""
        signals = []
        
        # Burstiness & sentence rhythm
        if feats["burstiness_cv"] < 0.25:
            signals.append("Low sentence-length variation (uniform sentence cadence characteristic of LLMs)")
        elif feats["burstiness_cv"] >= 0.45:
            signals.append("High sentence burstiness (dynamic mix of short and long clauses typical of human writers)")

        # LLM Discourse Markers
        if feats["llm_marker_density"] > 1.0:
            signals.append(f"Elevated frequency of formulaic transition markers ({feats['llm_marker_density']:.1f} per 100 words)")
        elif feats["llm_marker_density"] == 0:
            signals.append("Absence of stereotypical AI transitions or academic filler phrases")

        # Contractions & informal markers
        if feats["contraction_density"] > 1.2:
            signals.append("Natural usage of conversational contractions and colloquial phrasing")
        elif feats["contraction_density"] == 0 and feats["word_count"] > 60:
            signals.append("Formal contraction avoidance typical of default LLM system prompts")

        # Lexical diversity & Repetition
        if feats["ttr"] < 0.50 and feats["word_count"] > 80:
            signals.append("Lower lexical richness with predictable recurring word patterns")
        elif feats["ttr"] > 0.70:
            signals.append("High vocabulary diversity with idiosyncratic word choices")

        # Readability & structure
        if feats["flesch_reading_ease"] < 45.0:
            signals.append("High syntactic complexity and formal multi-syllable prose")
            
        if not signals:
            signals.append("Balanced linguistic markers observed across the text structure.")
            
        return signals[:4]

# Global singleton instance
detector_engine = DetectorEngine()
