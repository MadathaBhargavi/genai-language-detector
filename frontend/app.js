/**
 * GenAI Language Detector - Frontend Application Logic
 * Modern, robust, production-oriented client script.
 */

// API Configuration
const API_BASE = ""; // Relative path when served via FastAPI

// DOM Element Selectors
const textInput = document.getElementById("text-input");
const btnAnalyze = document.getElementById("btn-analyze");
const btnClear = document.getElementById("btn-clear");
const analyzeSpinner = document.getElementById("analyze-spinner");
const btnText = btnAnalyze.querySelector(".btn-text");
const wordCountBadge = document.getElementById("word-count-badge");
const charCountBadge = document.getElementById("char-count-badge");
const readingTimeBadge = document.getElementById("reading-time-badge");
const validationAlert = document.getElementById("validation-alert");

// File Upload Elements
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("file-input");
const btnBrowseFile = document.getElementById("btn-browse-file");
const uploadStatus = document.getElementById("upload-status");

// Results Elements
const resultsPlaceholder = document.getElementById("results-placeholder");
const resultsContent = document.getElementById("results-content");
const resultBanner = document.getElementById("result-banner");
const resultHeadline = document.getElementById("result-headline");
const confidenceBadge = document.getElementById("confidence-badge");
const aiProbNum = document.getElementById("ai-prob-num");
const humanProbNum = document.getElementById("human-prob-num");
const meterAiBar = document.getElementById("meter-ai-bar");
const meterHumanBar = document.getElementById("meter-human-bar");

// Writing Analysis Stats
const statWords = document.getElementById("stat-words");
const statSentences = document.getElementById("stat-sentences");
const statAvgLen = document.getElementById("stat-avg-len");
const statVocab = document.getElementById("stat-vocab");
const statVocabScore = document.getElementById("stat-vocab-score");
const statRepetition = document.getElementById("stat-repetition");
const statRepetitionScore = document.getElementById("stat-repetition-score");
const statBurstiness = document.getElementById("stat-burstiness");
const statBurstinessSub = document.getElementById("stat-burstiness-sub");
const statConsistency = document.getElementById("stat-consistency");
const statReadability = document.getElementById("stat-readability");
const statReadabilityScore = document.getElementById("stat-readability-score");

// Sentence Heatmap & Signals
const sentenceMapContainer = document.getElementById("sentence-map-container");
const signalsList = document.getElementById("signals-list");

// Action Buttons
const btnCopyReport = document.getElementById("btn-copy-report");
const btnExportJson = document.getElementById("btn-export-json");
const btnResetView = document.getElementById("btn-reset-view");
const themeToggle = document.getElementById("theme-toggle");

// Current analysis response cache for exports
let currentAnalysisData = null;

// =====================================================================
// Theme Management
// =====================================================================
function initTheme() {
  const savedTheme = localStorage.getItem("genai_detector_theme") || "dark";
  document.documentElement.setAttribute("data-theme", savedTheme);
}

themeToggle.addEventListener("click", () => {
  const current = document.documentElement.getAttribute("data-theme") || "dark";
  const next = current === "dark" ? "light" : "dark";
  document.documentElement.setAttribute("data-theme", next);
  localStorage.setItem("genai_detector_theme", next);
});

// =====================================================================
// Live Text Metrics
// =====================================================================
function countWords(str) {
  const matches = str.trim().match(/\b[a-zA-Z0-9']+\b/g);
  return matches ? matches.length : 0;
}

function updateLiveMetrics() {
  const text = textInput.value;
  const chars = text.length;
  const words = countWords(text);

  charCountBadge.textContent = chars.toLocaleString();
  wordCountBadge.textContent = words.toLocaleString();

  // Approx 200 words per minute reading speed
  const seconds = Math.ceil((words / 200) * 60);
  if (seconds < 60) {
    readingTimeBadge.textContent = `${seconds} sec read`;
  } else {
    readingTimeBadge.textContent = `${Math.ceil(seconds / 60)} min read`;
  }

  // Button state management
  const hasText = text.trim().length > 0;
  btnAnalyze.disabled = !hasText;
  btnClear.disabled = !hasText;

  // Clear validation alert if user continues typing
  if (validationAlert.classList.contains("hidden") === false && words >= 25) {
    hideAlert();
  }
}

textInput.addEventListener("input", updateLiveMetrics);

// Clear Button
btnClear.addEventListener("click", () => {
  textInput.value = "";
  updateLiveMetrics();
  hideAlert();
  textInput.focus();
});

// =====================================================================
// Sample Text Loader
// =====================================================================
const sampleButtons = document.querySelectorAll(".btn-sample");

let sampleCache = null;

async function loadSampleText(type) {
  try {
    if (!sampleCache) {
      const res = await fetch(`${API_BASE}/api/samples`);
      if (res.ok) {
        const data = await res.json();
        sampleCache = data.samples;
      }
    }

    if (sampleCache) {
      const match = sampleCache.find(s => s.id.startsWith(type));
      if (match) {
        textInput.value = match.text;
        updateLiveMetrics();
        hideAlert();
        textInput.scrollTop = 0;
        return;
      }
    }
  } catch (err) {
    console.warn("Could not fetch samples from API, using fallback", err);
  }

  // Static fallback if API is temporarily unreachable
  const fallbacks = {
    human: "I honestly didn't expect much when I first set up the backyard smoker last Saturday. My brother kept saying the temperature would fluctuate wildly because of the sudden chilly wind coming off the ridge, and for the first two hours he was totally right. I had to stand out there in an old flannel shirt, poking at the damp oak chunks with a pair of rusty grill tongs, wondering why I didn't just order barbecue takeout like a normal person. Around hour four, though, the thin blue smoke settled down. The crust on the brisket turned into that dark mahogany bark everyone obsesses over. When we finally carved into it on our picnic table, nobody said a word for ten straight minutes—we were just eating like wolves.",
    ai: "Artificial intelligence has fundamentally transformed the modern technological landscape, reshaping how industries operate and innovate. In today's fast-paced digital era, machine learning algorithms play a pivotal role in optimizing operational workflows, enhancing predictive accuracy, and fostering data-driven decision-making. Furthermore, it is important to note that the seamless integration of neural networks across sectors such as healthcare and finance underscores the vital importance of robust governance frameworks. In conclusion, the future of artificial intelligence holds immense promise, offering a rich tapestry of opportunities for sustainable societal advancement.",
    mixed: "Last Tuesday, our engineering squad debated how to handle sudden spikes in our checkout webhook. I was arguing that we should stick with simple database row locking because nobody wanted to introduce another message broker before our November release. However, integrating a distributed pub-sub architecture ensures that messages are processed asynchronously without overwhelming database connection pools. Moreover, it is crucial to recognize that implementing exponential backoff strategies mitigates transient network failures. We ended up writing a quick Redis proof-of-concept over lukewarm coffee."
  };

  if (fallbacks[type]) {
    textInput.value = fallbacks[type];
    updateLiveMetrics();
    hideAlert();
  }
}

sampleButtons.forEach(btn => {
  btn.addEventListener("click", () => {
    const type = btn.getAttribute("data-sample");
    loadSampleText(type);
  });
});

// =====================================================================
// Document Upload (TXT, PDF, DOCX)
// =====================================================================
btnBrowseFile.addEventListener("click", () => fileInput.click());
dropzone.addEventListener("click", (e) => {
  if (e.target !== btnBrowseFile) fileInput.click();
});

// Drag and drop event listeners
["dragenter", "dragover"].forEach(eventName => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.add("drag-active");
  });
});

["dragleave", "drop"].forEach(eventName => {
  dropzone.addEventListener(eventName, (e) => {
    e.preventDefault();
    e.stopPropagation();
    dropzone.classList.remove("drag-active");
  });
});

dropzone.addEventListener("drop", (e) => {
  const dt = e.dataTransfer;
  const files = dt.files;
  if (files && files.length > 0) {
    handleFileUpload(files[0]);
  }
});

fileInput.addEventListener("change", (e) => {
  if (e.target.files && e.target.files.length > 0) {
    handleFileUpload(e.target.files[0]);
  }
});

async function handleFileUpload(file) {
  showUploadStatus(`Uploading & extracting ${file.name}...`, "info");
  const formData = new FormData();
  formData.append("file", file);

  try {
    const response = await fetch(`${API_BASE}/api/upload`, {
      method: "POST",
      body: formData
    });

    const data = await response.json();
    if (!response.ok) {
      throw new Error(data.detail || "Failed to parse document");
    }

    textInput.value = data.text;
    updateLiveMetrics();
    showUploadStatus(`✓ Extracted ${data.word_count} words from ${file.name}`, "success");
    setTimeout(() => {
      uploadStatus.classList.add("hidden");
    }, 4000);
  } catch (err) {
    showUploadStatus(`✕ ${err.message}`, "error");
  }
}

function showUploadStatus(msg, status) {
  uploadStatus.textContent = msg;
  uploadStatus.className = `upload-status ${status}`;
  uploadStatus.classList.remove("hidden");
}

// =====================================================================
// Analyze Action & API Request
// =====================================================================
btnAnalyze.addEventListener("click", async () => {
  const text = textInput.value.trim();
  if (!text) {
    showAlert("Please enter some text to analyze.");
    return;
  }

  const words = countWords(text);
  if (words < 15) {
    showAlert("Text is very short. For the most accurate analysis, enter at least 30-50 words.");
  }

  setAnalyzingState(true);

  try {
    const response = await fetch(`${API_BASE}/api/analyze`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({ text })
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.detail || "Analysis request failed.");
    }

    currentAnalysisData = data;
    renderResults(data);

    if (data.warning) {
      showAlert(data.warning);
    } else {
      hideAlert();
    }
  } catch (err) {
    showAlert(`Error: ${err.message}`);
  } finally {
    setAnalyzingState(false);
  }
});

function setAnalyzingState(isAnalyzing) {
  btnAnalyze.disabled = isAnalyzing;
  btnClear.disabled = isAnalyzing;
  if (isAnalyzing) {
    btnText.textContent = "Analyzing...";
    analyzeSpinner.classList.remove("hidden");
  } else {
    btnText.textContent = "Analyze Text";
    analyzeSpinner.classList.add("hidden");
  }
}

function showAlert(message) {
  validationAlert.textContent = message;
  validationAlert.classList.remove("hidden");
}

function hideAlert() {
  validationAlert.classList.add("hidden");
}

// =====================================================================
// Render Detection Results
// =====================================================================
function renderResults(data) {
  // Hide placeholder, reveal content
  resultsPlaceholder.classList.add("hidden");
  resultsContent.classList.remove("hidden");

  // Result headline & styling class
  const result = data.result;
  resultHeadline.textContent = result;

  // Banner status class
  resultBanner.className = "result-banner";
  let statusClass = "status-uncertain";
  if (result.includes("AI-Generated")) {
    statusClass = "status-ai";
  } else if (result.includes("Human-Written")) {
    statusClass = "status-human";
  } else if (result.includes("Mixed") || result.includes("Assisted")) {
    statusClass = "status-mixed";
  }
  resultBanner.classList.add(statusClass);

  // Confidence Badge
  const conf = data.confidence || "Medium";
  confidenceBadge.textContent = `${conf} Confidence`;
  confidenceBadge.className = `confidence-badge ${conf.toLowerCase()}`;

  // Probabilities
  const aiPct = Math.round(data.ai_probability * 100);
  const humanPct = Math.round(data.human_probability * 100);

  aiProbNum.textContent = `${aiPct}%`;
  humanProbNum.textContent = `${humanPct}%`;

  // Dual Meter Bar animation
  meterAiBar.style.width = `${aiPct}%`;
  meterHumanBar.style.width = `${humanPct}%`;

  // Writing Analysis Cards
  const wa = data.writing_analysis || {};
  statWords.textContent = (wa.words || data.word_count || 0).toLocaleString();
  statSentences.textContent = (wa.sentences || data.sentence_count || 0).toLocaleString();
  statAvgLen.textContent = (wa.average_sentence_length || data.average_sentence_length || 0).toFixed(1);

  statVocab.textContent = wa.vocabulary_diversity || "Medium";
  statVocabScore.textContent = `TTR ${wa.vocabulary_diversity_score || 0}%`;

  statRepetition.textContent = wa.repetition || "Low";
  statRepetitionScore.textContent = `2-gram ${wa.repetition_score || 0}%`;

  statBurstiness.textContent = wa.sentence_variation || "Medium";
  statBurstinessSub.textContent = `Std dev ${wa.sentence_variation_std || 0}`;

  statConsistency.textContent = wa.writing_consistency || "Medium";
  statReadability.textContent = wa.readability_level || "Moderate";
  statReadabilityScore.textContent = `Flesch score ${wa.readability_score || 0}`;

  // Sentence-Level Heatmap
  renderSentenceMap(data.sentence_breakdown || []);

  // Diagnostic Signals
  renderSignals(data.signals || []);

  // Smooth scroll to results on mobile viewports
  if (window.innerWidth <= 1024) {
    resultsContent.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

function renderSentenceMap(sentences) {
  sentenceMapContainer.innerHTML = "";

  if (!sentences || sentences.length === 0) {
    sentenceMapContainer.innerHTML = "<p class='text-muted'>No sentence segments available.</p>";
    return;
  }

  sentences.forEach((s) => {
    const span = document.createElement("span");
    span.className = `sentence-token ${s.label || "neutral"}`;
    span.textContent = s.text + " ";
    const aiPct = Math.round((s.ai_prob || 0.5) * 100);
    span.title = `Sentence AI likelihood: ${aiPct}% (${s.word_count} words)`;
    
    // Click inspection
    span.addEventListener("click", () => {
      showAlert(`Selected sentence (${s.word_count} words): ${aiPct}% estimated AI probability.`);
    });

    sentenceMapContainer.appendChild(span);
  });
}

function renderSignals(signals) {
  signalsList.innerHTML = "";
  if (!signals || signals.length === 0) {
    const li = document.createElement("li");
    li.textContent = "Standard linguistic patterns observed across text features.";
    signalsList.appendChild(li);
    return;
  }

  signals.forEach(sig => {
    const li = document.createElement("li");
    li.textContent = sig;
    signalsList.appendChild(li);
  });
}

// =====================================================================
// Report Export & Reset Handlers
// =====================================================================
btnCopyReport.addEventListener("click", () => {
  if (!currentAnalysisData) return;
  const d = currentAnalysisData;
  const wa = d.writing_analysis || {};

  const report = `=== GenAI Language Detector Report ===
Detection Result: ${d.result}
AI Probability: ${Math.round(d.ai_probability * 100)}%
Human Probability: ${Math.round(d.human_probability * 100)}%
Confidence: ${d.confidence}

Writing Analysis:
- Words: ${d.word_count}
- Sentences: ${d.sentence_count}
- Average Sentence Length: ${d.average_sentence_length} words
- Vocabulary Diversity: ${wa.vocabulary_diversity} (${wa.vocabulary_diversity_score}%)
- Repetition: ${wa.repetition} (${wa.repetition_score}%)
- Sentence Variation: ${wa.sentence_variation}
- Writing Consistency: ${wa.writing_consistency}
- Readability: ${wa.readability_level} (${wa.readability_score})

Key Signals:
${(d.signals || []).map(s => `* ${s}`).join("\n")}

Disclaimer: ${d.disclaimer}
`;

  navigator.clipboard.writeText(report).then(() => {
    const originalText = btnCopyReport.textContent;
    btnCopyReport.textContent = "✓ Report Copied!";
    setTimeout(() => {
      btnCopyReport.textContent = originalText;
    }, 2500);
  }).catch(() => {
    alert("Could not automatically copy to clipboard. Please copy manually.");
  });
});

btnExportJson.addEventListener("click", () => {
  if (!currentAnalysisData) return;
  const blob = new Blob([JSON.stringify(currentAnalysisData, null, 2)], { type: "application/json" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `genai-detection-report-${Date.now()}.json`;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
});

btnResetView.addEventListener("click", () => {
  textInput.value = "";
  updateLiveMetrics();
  resultsContent.classList.add("hidden");
  resultsPlaceholder.classList.remove("hidden");
  hideAlert();
  textInput.focus();
  window.scrollTo({ top: 0, behavior: "smooth" });
});

// Initialization
document.addEventListener("DOMContentLoaded", () => {
  initTheme();
  updateLiveMetrics();
});
