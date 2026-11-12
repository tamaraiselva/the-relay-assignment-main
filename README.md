# The Relay — Research Paper Assistant & Rate-Limit Benchmark

A production-minded research-paper Q&A assistant built over a fixed library of three seminal AI papers (*Attention Is All You Need*, *Moshi*, and *Dynamic Chunking*). The system features custom BM25 retrieval, streaming answer generation with Groq (`openai/gpt-oss-120b`), citation verification, automated fallback abstentions, a calibrated stand-in provider for stress/concurrency testing, and a Streamlit dashboard.

---

## Table of Contents
1. [Project Overview](#1-project-overview)
2. [Code Architecture & Repository Structure](#2-code-architecture--repository-structure)
3. [Quick Start & Deployment](#3-quick-start--deployment)
4. [Write-Up & Technical Report](#4-write-up--technical-report)
   - [What Was Built & Design Decisions](#41-what-was-built--why)
   - [Limits Found in Real Service vs. Stand-In Calibration](#42-the-real-service-and-stand-in-calibration)
   - [Benchmark & Latency Results](#43-benchmark--latency-results)
   - [Where the Time Goes (Latency Breakdown)](#44-where-the-time-goes)
   - [When Limits Are Hit & Failure Modes](#45-when-a-limit-is-hit)
   - [Retrieval Settings: Quality vs. Cost](#46-retrieval-quality-vs-cost)
   - [Naive Retriever Failure Cases](#47-how-this-library-misleads-a-naive-retriever)
   - [Retry Policy](#48-retry-policy)
   - [User Experience Under Rate Limits](#49-what-the-person-sees)
   - [Feedback on Vaani](#410-feedback-on-vaani)
   - [AI Assistant Disclosure](#411-how-i-worked-with-the-assistant)
5. [Results Log (`results.csv`)](#5-results-log-resultscsv)

---

## 1. Project Overview

- **Core Goal**: Answer questions strictly grounded in three research papers, enforcing citations (e.g., `[S1]`) and refusing to guess when evidence is absent or insufficient.
- **Retrieval Engine**: From-scratch BM25 search engine with stopword elimination, frequency-based relevance scoring, and configurable chunking strategies (`small_k3`, `medium_k4`, `large_k5_rerank`).
- **LLM Pipeline**: Streaming responses via Groq API using `openai/gpt-oss-120b`, logging time-to-first-token (TTFT) and total latency.
- **Stand-In Testing Harness**: A local calibrated provider mimicking Groq's TTFT and token generation rate to stress-test rate limits, concurrency limits, and token budgets without depleting API quotas.
- **Interactive UI**: Streamlit web app allowing users to test queries, inspect BM25-ranked passages, evaluate answers, and view aggregated benchmark metrics.

---

## 2. Code Architecture & Repository Structure

```text
├── app.py                      # Interactive Streamlit dashboard & inspection UI
├── requirements.txt            # Python dependencies (Streamlit, Groq, PyMuPDF, etc.)
├── .python-version             # Pinned to Python 3.11.9 for stable Linux container builds
├── .streamlit/
│   └── config.toml             # UI theme & headless server configuration
├── src/
│   └── relay/
│       ├── retrieval.py        # Custom BM25 implementation & corpus loader
│       ├── answer.py           # Core pipeline: retrieval, prompt crafting, streaming, citations
│       └── providers/
│           └── standin.py      # Simulated Groq provider for rate-limit and concurrency load testing
├── scripts/
│   ├── extract_papers.py       # Extracts page-level text from raw PDFs using PyMuPDF
│   ├── chunk_pages.py          # Splits extracted text into configured chunk sizes with overlap
│   ├── load_test_standin.py    # Multi-threaded load tester simulating concurrency, burst, and overload
│   └── sumarize_results.py     # Aggregates results.csv into percentile latency & outcome tables
├── data/
│   ├── papers/                 # Source PDF papers (Attention, Moshi, Dynamic Chunking)
│   └── processed/              # Pre-processed JSONL chunk files for BM25 retrieval
├── results.csv                 # Complete benchmark run log (150+ recorded executions)
├── question_log.csv            # Questions queried and evaluation logs
├── rate_limit_headers.jsonl    # Captured HTTP response rate-limit headers from Groq
└── REPORT.md                   # Full assignment technical report
```

### Key Modules
- **`src/relay/retrieval.py`**: Tokenizes queries and text passages using regex matching, filters standard English stopwords, and calculates BM25 scores ($k_1=1.5, b=0.75$). Loads pre-chunked indices from `data/processed/`.
- **`src/relay/answer.py`**: Formats the retrieved context with unique source tags, enforces a strict anti-hallucination system prompt, streams the model's completion, tracks latency metrics, and logs every execution to `results.csv`.
- **`src/relay/providers/standin.py`**: A thread-safe stand-in that simulates Groq's latency distribution ($\text{TTFT} \approx 806\text{ ms}$, $\approx 775\text{ tokens/s}$), tracks sliding-window RPM and daily request caps, and triggers realistic 429 errors when thresholds are breached.
- **`app.py`**: Streamlit dashboard with a sidebar configuration panel, interactive question box, ranked evidence inspector with BM25 score chips, real-time token/latency metrics, and manual quality grading interface (`correct`, `wrong`, `refused_correctly`).

---

## 3. Quick Start & Deployment

### Local Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/tamaraiselva/the-relay-assignment-main.git
   cd the-relay-assignment-main
   ```
2. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # Windows:
   .venv\Scripts\activate
   # Linux/macOS:
   source .venv/bin/activate
   ```
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure Environment Variables**:
   Create a `.env` file in the project root:
   ```env
   GROQ_API_KEY=your_groq_api_key_here
   GROQ_MODEL=openai/gpt-oss-120b
   ```
5. **Launch the Dashboard**:
   ```bash
   streamlit run app.py
   ```

### Deploying to Render
1. Create a new **Web Service** connected to your repository.
2. Configure settings:
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `streamlit run app.py --server.port $PORT --server.address 0.0.0.0`
3. In the **Environment** tab, set:
   - `GROQ_API_KEY`: `gsk_...`
   - `GROQ_MODEL`: `openai/gpt-oss-120b`

---

## 4. Write-Up & Technical Report

*(Extracted from [`REPORT.md`](REPORT.md))*

### 4.1 What Was Built & Why
A Python research-paper assistant was developed for a fixed library of three papers (*Attention Is All You Need*, *Moshi*, and *Dynamic Chunking*). The user selects one of three retrieval strategies and submits a question. BM25 ranks the candidate passages, and the selected excerpts are dispatched to Groq `openai/gpt-oss-120b`. The model streams a cited answer or abstains when evidence is insufficient. The pipeline records request metrics and user-assigned quality grades in `results.csv`.

**Core Design Decision**: Explicit refusal is prioritized over guessing. While this leaves unsupported questions unanswered, it prevents hallucinations and unsupported claims in a research context.

---

### 4.2 The Real Service and Stand-In Calibration

#### Table 1: Limits Found in Real Service
| Limit | Value Found | How Found | Confidence |
| :--- | :---: | :--- | :---: |
| **Requests per minute (RPM)** | 30 | Groq account Limits page | High |
| **Tokens per minute (TPM)** | 8,000 | Groq Limits page & response header | High |
| **Requests per day (RPD)** | 1,000 | Groq Limits page (`x-ratelimit-limit-requests` header) | High |
| **Tokens per day (TPD)** | 200,000 | Groq account Limits page | High |
| **In-Flight Concurrency** | Not measured | Concurrency was not tested against live API | Low |
| **Response on Limit Breach** | HTTP 429 Too Many Requests | Groq official documentation | Medium |
| **Recovery Window** | `Retry-After` header | Documented by Groq | Low |

#### Stand-In Behavior & Calibration
The stand-in is an in-process provider calibrated against real Groq measurements:
- **Baseline Time-to-First-Token (TTFT)**: 806 ms
- **Simulated Output Speed**: 775 tokens/sec
- **Calibration Reference**: Derived from a baseline Groq request ($\text{TTFT} = 806.3\text{ ms}$, total time $= 1,218.2\text{ ms}$, 319 output tokens). Under steady simulated load, mean total time was 1,199.2 ms.

---

### 4.3 Benchmark & Latency Results

#### Table 2a: Latency by Scenario
| Scenario | Service | Requests Sent | TTFT p50 / p95 / p99 (ms) | Total Time p50 / p95 / p99 (ms) |
| :--- | :---: | :---: | :---: | :---: |
| `steady` | Stand-in | 8 | 806.5 / 806.6 / 806.7 | 1198.8 / 1201.1 / 1201.1 |
| `burst` | Stand-in | 10 | 1106.3 / 1406.8 / 1406.9 | 1500.7 / 1801.8 / 1802.1 |
| `sustained_overload` | Stand-in | 8 | 956.4 / 1091.2 / 1103.2 | 956.4 / 1091.2 / 1103.2 |
| `sustained_overload_8_waves` | Stand-in | 64 | 956.4 / 1106.8 / 1107.0 | 956.4 / 1106.9 / 1107.0 |
| `rpm_halved` | Stand-in | 32 | 1.6 / 1.9 / 1.9 | 1.6 / 1.9 / 1.9 |
| `daily_budget_exhausted` | Stand-in | 8 | 1.6 / 1.9 / 1.9 | 1.6 / 1.9 / 1.9 |
| `interactive_demo` | Real Groq | 3 | 931.0 / 1121.7 / 1138.7 | 1251.3 / 1526.2 / 1550.6 |
| `smoke_test` | Real Groq | 24 | 1350.3 / 1648.6 / 1750.5 | 1582.0 / 2067.8 / 2206.0 |

#### Table 2b: Outcomes by Scenario
| Scenario | Succeeded (%) | Rate-Limited (%) | Errors / Dropped (%) | Tokens / Answer | Retries |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `steady` | 100.0% | 0.0% | 0.0% | 1,293.0 | 0.0 |
| `burst` | 100.0% | 0.0% | 0.0% | 1,293.0 | 0.0 |
| `sustained_overload` | 37.5% | 0.0% | 62.5% | 2.0 | 0.0 |
| `sustained_overload_8_waves` | 37.5% | 0.0% | 62.5% | 2.0 | 0.0 |
| `rpm_halved` | 93.8% | 6.2% | 0.0% | 2.0 | 0.0 |
| `daily_budget_exhausted` | 75.0% | 25.0% | 0.0% | 2.0 | 0.0 |
| `interactive_demo` | 100.0% | 0.0% | 0.0% | 2,205.0 | 0.0 |
| `smoke_test` | 91.7% | 0.0% | 8.3% | 2,206.1 | 0.0 |

---

### 4.4 Where the Time Goes
- Slowest observed real request: `d8c6ab46-891b-4cd9-8fd0-efdd1956c4cc` (`small_k3`, `smoke_test`).
  - **TTFT**: 1,606.3 ms
  - **Total Pipeline Latency**: 2,242.2 ms
  - **Stream Consumption**: 635.9 ms elapsed while reading output tokens.
- **Latency Attribution**: The timer starts before BM25 retrieval. In total TTFT (e.g. ~1,337 ms), retrieval, prompt assembly, TLS handshake, Groq queuing, and first token generation are combined.

---

### 4.5 When a Limit is Hit
- Under real Groq limits (30 RPM, 8k TPM, 1k RPD, 200k TPD), the client handles HTTP 429 without unhandled crashes.
- Stand-in load tests:
  - Halving RPM produced 6.2% rate-limited requests (2/32).
  - Daily budget exhaustion produced 25.0% rate-limited requests (2/8).
  - Overload waves dropped 62.5% of requests due to concurrency capping rather than queueing indefinitely.

---

### 4.6 Retrieval: Quality vs. Cost

#### Table 3: Performance Across Chunking Settings
| Setting | Chunk Size / Overlap / Fetched | Correct & Cited | Wrong | Refused Correctly | Mean Tokens | TTFT p50 (ms) | Total Time p50 (ms) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| `small_k3` | 180w / 30w / 3 passages | 20.0% | 60.0% | 20.0% | 1,343.0 | 1,363.1 | 1,551.9 |
| **`medium_k4`** | **320w / 50w / 4 passages** | **80.0%** | **20.0%** | **0.0%** | **2,362.6** | **1,428.4** | **1,855.4** |
| `large_k5_rerank` | 500w / 80w / 5 passages | 20.0% | 80.0% | 0.0% | 3,812.4 | 1,382.4 | 1,785.2 |

**Recommendation**: `medium_k4` is the recommended default setting. It provided the highest correct-and-cited rate (80.0%) while maintaining a moderate token footprint (~2.3k tokens). Under tight rate-limit conditions, users can downshift to `small_k3` to conserve tokens.

---

### 4.7 How This Library Misleads a Naive Retriever
1. **False Keyword Overlap on Out-of-Scope Questions**: Asking *"What is an LLM?"* triggers keyword matches across Transformer texts, but the paper lacks an explicit definition. A naive system hallucinates; our system correctly triggers an abstention.
2. **Unsupported Claims in Related Passages**: Asking about the Mimi codec in Moshi retrieves valid codec passages, but the model asserted a teacher-model claim not in the evidence.
3. **Pipeline Fragility on Supported Questions**: Asking *"How does Moshi model the user's and assistant's speech streams to support full-duplex dialogue?"* produced a citation fallback even though the paper contained the answer, highlighting the sensitivity of citation matching.

---

### 4.8 Retry Policy
- `max_retries=0` was enforced to prevent compounding rate-limit thundering herds.
- Under RPM limits, immediate retries are guaranteed to fail and consume network resources. Recommended future policy: at most 1 retry, strictly governed by the HTTP `Retry-After` header.

---

### 4.9 What the Person Sees
- When daily/minute limits are reached, the system captures HTTP 429 and displays a clear message:
  > *"Today's request limit has been reached, so your question was not answered. Please try again after the limit resets. Request ID: [UUID]."*

---

### 4.10 Feedback on Vaani
- Review of [vaani.dev](https://www.vaani.dev/):
  - **Strengths**: Low latency full-duplex voice interaction, natural conversational turn-taking, and the "Teacher AI" flow providing human-in-the-loop review before deployment.
  - **Suggestion**: The site advertises low latency figures (e.g., p90). Providing a technical breakdown of measurement conditions (audio buffer size, TTFT vs. full synthesis, network variance) and an interactive live web demo would enhance trust.

---

### 4.11 How I Worked with the Assistant
- Used an AI coding assistant to accelerate development.
- **Key prompts**: Understanding existing pipeline mechanics, debugging concurrent stand-in load tests, and auditing citation verification logic.
- **Human oversight**: Overruled assistant advice when grading an out-of-scope question (corrected assistant's recommendation from `correct` to `refused_correctly`), and rejected automated batch evaluations in favor of manual single-query inspection.

---

## 5. Results Log (`results.csv`)

The complete history of 150+ benchmark runs is persisted in [`results.csv`](results.csv). Each row contains:
- `request_id`: Unique UUID per execution
- `scenario`: Test scenario (`steady`, `burst`, `sustained_overload`, `rpm_halved`, `daily_budget_exhausted`, `interactive_demo`, `smoke_test`)
- `source`: Backend source (`real` Groq API or `stand-in`)
- `start_time`: ISO-8601 UTC timestamp
- `first_token_ms`: Time to first token
- `total_ms`: End-to-end execution latency
- `status`: Execution outcome (`ok`, `error`, `rate_limited`)
- `retries`: Number of retries attempted
- `tokens_in` & `tokens_out`: Token consumption metrics
- `retrieval_setting`: Strategy used (`small_k3`, `medium_k4`, `large_k5_rerank`, `standin`)
- `answer_grade`: Manual human grade (`correct`, `wrong`, `refused_correctly`)
