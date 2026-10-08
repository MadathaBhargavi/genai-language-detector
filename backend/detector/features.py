"""
Feature extraction pipeline for GenAI Language Detector.
Extracts linguistic, stylometric, syntactic, and perplexity-proxy features
from input text to differentiate between human and AI-generated text.
"""

import re
import math
from collections import Counter
from typing import Dict, List, Any, Tuple
import numpy as np

# Common discourse markers and buzzwords heavily over-indexed in LLM outputs
LLM_MARKERS = {
    "delve", "delving", "tapestry", "crucial", "crucially", "testament", "beacon",
    "pivotal", "underscores", "interplay", "navigating", "foster", "fostering",
    "holistic", "seamlessly", "paramount", "nuanced", "multifaceted", "cornerstone",
    "moreover", "furthermore", "in conclusion", "it is important to note",
    "it is worth noting", "to summarize", "in summary", "overall", "firstly",
    "secondly", "lastly", "consequently", "subsequently", "nevertheless", "harness",
    "harnessing", "unravel", "demystify", "embark", "testament to", "rich tapestry",
    "vital role", "a testament", "dynamic landscape", "ever-evolving"
}

# Standard contractions common in human informal and semi-formal writing
CONTRACTIONS = {
    "don't", "can't", "won't", "didn't", "i'm", "i've", "it's", "you're", "they're",
    "we're", "couldn't", "shouldn't", "wouldn't", "isn't", "aren't", "wasn't",
    "weren't", "haven't", "hasn't", "hadn't", "that's", "there's", "what's", "let's"
}

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences cleanly using regex preserving sentence boundaries."""
    raw_sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    sentences = [s.strip() for s in raw_sentences if s.strip() and len(s.strip()) > 3]
    return sentences if sentences else [text.strip()]

def tokenize_words(text: str) -> List[str]:
    """Extract lowercased word tokens."""
    return re.findall(r"\b[a-zA-Z']+\b", text.lower())

def calculate_char_entropy(text: str) -> float:
    """Calculate Shannon entropy of character distribution."""
    if not text:
        return 0.0
    counts = Counter(text)
    total = len(text)
    entropy = -sum((cnt / total) * math.log2(cnt / total) for cnt in counts.values())
    return entropy

def calculate_word_entropy(words: List[str]) -> float:
    """Calculate Shannon entropy of word distribution."""
    if not words:
        return 0.0
    counts = Counter(words)
    total = len(words)
    entropy = -sum((cnt / total) * math.log2(cnt / total) for cnt in counts.values())
    return entropy

def calculate_syllables(word: str) -> int:
    """Heuristic count of syllables in an English word."""
    word = word.lower()
    if len(word) <= 3:
        return 1
    word = re.sub(r'(?:[^laeiouy]|ed|es|e)$', '', word)
    word = re.sub(r'^y', '', word)
    matches = re.findall(r'[aeiouy]{1,2}', word)
    return max(1, len(matches))

def calculate_burstiness(sentence_lengths: List[int]) -> Tuple[float, float]:
    """
    Burstiness measures variation in sentence lengths.
    Human writing has high burstiness (very short sentences mixed with long clauses).
    AI writing has low burstiness (uniform sentence lengths).
    Returns (std_dev, coefficient_of_variation).
    """
    if len(sentence_lengths) < 2:
        return 0.0, 0.0
    mean = np.mean(sentence_lengths)
    std = float(np.std(sentence_lengths))
    cv = (std / mean) if mean > 0 else 0.0
    return std, cv

def calculate_perplexity_proxy(text: str, words: List[str]) -> float:
    """
    Calculates a n-gram statistical language model cross-entropy proxy.
    AI text exhibits high predictability (lower cross-entropy / perplexity proxy),
    while human text exhibits higher surprise and idiosyncratic phrasing.
    """
    if len(words) < 5:
        return 50.0

    # Bigram and trigram frequency distribution
    bigrams = [(words[i], words[i+1]) for i in range(len(words)-1)]
    trigrams = [(words[i], words[i+1], words[i+2]) for i in range(len(words)-2)]
    
    bi_counts = Counter(bigrams)
    tri_counts = Counter(trigrams)
    word_counts = Counter(words)
    
    total_bigrams = len(bigrams)
    
    # Calculate transition log-likelihood proxy
    log_probs = []
    for w1, w2 in bigrams:
        count_w1 = word_counts[w1]
        count_w1_w2 = bi_counts[(w1, w2)]
        # Add-1 Laplace smoothing
        prob = (count_w1_w2 + 0.1) / (count_w1 + 0.1 * len(word_counts))
        log_probs.append(math.log(prob))
        
    avg_log_prob = np.mean(log_probs) if log_probs else -5.0
    # Perplexity proxy = exp(-avg_log_prob)
    perplexity = float(np.exp(-avg_log_prob))
    return min(perplexity, 500.0)

def extract_features(text: str) -> Dict[str, Any]:
    """
    Extract comprehensive linguistic, stylometric, and statistical features.
    Returns both feature vector for ML and human-readable metrics.
    """
    sentences = split_into_sentences(text)
    words = tokenize_words(text)
    
    total_chars = len(text)
    total_words = len(words)
    total_sentences = max(1, len(sentences))
    
    # Word lengths
    word_lengths = [len(w) for w in words] if words else [0]
    avg_word_length = float(np.mean(word_lengths)) if words else 0.0
    
    # Sentence lengths in words
    sentence_word_counts = [len(tokenize_words(s)) for s in sentences]
    avg_sentence_length = float(np.mean(sentence_word_counts)) if sentence_word_counts else 0.0
    
    # Burstiness
    sentence_len_std, burstiness_cv = calculate_burstiness(sentence_word_counts)
    
    # Lexical Diversity (Type-Token Ratio & Root TTR)
    unique_words = set(words)
    ttr = (len(unique_words) / total_words) if total_words > 0 else 0.0
    root_ttr = (len(unique_words) / math.sqrt(total_words)) if total_words > 0 else 0.0
    
    # Hapax Legomena (words occurring exactly once)
    word_counts = Counter(words)
    hapax_count = sum(1 for c in word_counts.values() if c == 1)
    hapax_ratio = (hapax_count / len(unique_words)) if unique_words else 0.0
    
    # Repetition: n-gram repetition rates
    bigrams = [f"{words[i]}_{words[i+1]}" for i in range(len(words)-1)] if len(words) >= 2 else []
    unique_bigrams = set(bigrams)
    bigram_repetition = (1.0 - (len(unique_bigrams) / len(bigrams))) if bigrams else 0.0
    
    trigrams = [f"{words[i]}_{words[i+1]}_{words[i+2]}" for i in range(len(words)-2)] if len(words) >= 3 else []
    unique_trigrams = set(trigrams)
    trigram_repetition = (1.0 - (len(unique_trigrams) / len(trigrams))) if trigrams else 0.0
    
    # LLM Discourse Markers density per 100 words
    llm_hits = sum(1 for w in words if w in LLM_MARKERS)
    # Check multi-word markers
    text_lower = text.lower()
    for marker in LLM_MARKERS:
        if " " in marker and marker in text_lower:
            llm_hits += 2
    llm_marker_density = (llm_hits / max(1, total_words)) * 100.0
    
    # Contractions density (Humans use significantly more contractions in natural writing)
    contraction_hits = sum(1 for w in words if w in CONTRACTIONS)
    contraction_density = (contraction_hits / max(1, total_words)) * 100.0
    
    # Entropies
    char_entropy = calculate_char_entropy(text)
    word_entropy = calculate_word_entropy(words)
    
    # Perplexity proxy
    perplexity = calculate_perplexity_proxy(text, words)
    
    # Punctuation analysis
    punct_chars = re.findall(r'[,;:!?"\-\(\)]', text)
    punct_density = (len(punct_chars) / max(1, total_chars)) * 100.0
    comma_density = (text.count(',') / max(1, total_sentences))
    semicolon_count = text.count(';')
    exclamation_count = text.count('!')
    
    # Readability: Flesch Reading Ease & Grade Level
    syllables = sum(calculate_syllables(w) for w in words)
    avg_syllables_per_word = (syllables / total_words) if total_words > 0 else 1.0
    
    # Flesch Reading Ease = 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
    flesch_reading_ease = 206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
    flesch_reading_ease = max(0.0, min(100.0, flesch_reading_ease))
    
    # Sentence length variance between adjacent sentences (Delta burstiness)
    if len(sentence_word_counts) >= 2:
        adjacent_diffs = [abs(sentence_word_counts[i] - sentence_word_counts[i-1]) for i in range(1, len(sentence_word_counts))]
        avg_adjacent_delta = float(np.mean(adjacent_diffs))
    else:
        avg_adjacent_delta = 0.0

    feature_dict = {
        # Core numerical features for ML model (ordered vector)
        "avg_sentence_length": avg_sentence_length,
        "sentence_len_std": sentence_len_std,
        "burstiness_cv": burstiness_cv,
        "avg_adjacent_delta": avg_adjacent_delta,
        "ttr": ttr,
        "root_ttr": root_ttr,
        "hapax_ratio": hapax_ratio,
        "bigram_repetition": bigram_repetition,
        "trigram_repetition": trigram_repetition,
        "llm_marker_density": llm_marker_density,
        "contraction_density": contraction_density,
        "char_entropy": char_entropy,
        "word_entropy": word_entropy,
        "perplexity": perplexity,
        "avg_word_length": avg_word_length,
        "punct_density": punct_density,
        "comma_density": comma_density,
        "flesch_reading_ease": flesch_reading_ease,
        
        # Metadata / writing statistics for UI presentation
        "word_count": total_words,
        "sentence_count": total_sentences,
        "character_count": total_chars,
        "syllable_count": syllables,
    }
    
    return feature_dict

FEATURE_NAMES = [
    "avg_sentence_length",
    "sentence_len_std",
    "burstiness_cv",
    "avg_adjacent_delta",
    "ttr",
    "root_ttr",
    "hapax_ratio",
    "bigram_repetition",
    "trigram_repetition",
    "llm_marker_density",
    "contraction_density",
    "char_entropy",
    "word_entropy",
    "perplexity",
    "avg_word_length",
    "punct_density",
    "comma_density",
    "flesch_reading_ease"
]

def get_feature_vector(features: Dict[str, Any]) -> List[float]:
    """Extract ordered numeric feature vector for scikit-learn models."""
    return [float(features[name]) for name in FEATURE_NAMES]
