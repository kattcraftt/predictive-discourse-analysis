# Predictive Discourse Analysis: Enhancing Customer Behavior Forecasting in Digital Markets

A hybrid, discourse-driven machine learning framework integrating **PRISMA 2020**, **RoBERTa**, **BERTopic**, and **Random Forest** to model consumer sentiment, extract discourse patterns, classify behavior, and project longitudinal market shifts (2027–2030).

---

## Overview

Modern digital consumer discourse is distributed across heterogeneous platforms—from long-form video commentary to decentralized microblogging and technical forums. Understanding how this unstructured text translates into measurable customer actions is critical for proactive market planning.

This research and software framework delivers an end-to-end, multi-stage pipeline designed to:
- **Harvest organic user discourse** at scale from YouTube, Bluesky, and HackerNews via public APIs.
- **Enforce rigorous methodological filtering** using a 4-stage systematic screening protocol grounded in the **PRISMA 2020** framework.
- **Disentangle multi-dimensional NLP signals** by pairing Transformer-based sentiment classification (`RoBERTa`) with contextual topic modeling (`BERTopic` over `all-MiniLM-L6-v2`).
- **Predict granular behavioral classes** (`Purchase Intent`, `Passive Browsing`, `Churn Risk`) using an optimized **Random Forest** classifier.
- **Forecast macro market trends** by projecting behavioral distribution trajectories through 2030.

---

## Practical Use Cases

| Domain | Application Scenario |
|---|---|
| **E-Commerce & Retail** | Detect early churn indicators and shifting purchase intent from unstructured post-purchase reviews and product forum discussions. |
| **Market Intelligence** | Track organic product discussions across emerging and niche networks (Bluesky, HackerNews) without relying on paid survey bias. |
| **Customer Experience (CX)** | Identify top thematic drivers (via BERTopic clusters) linked directly to high churn risk to prioritize product roadmaps and fixes. |
| **Longitudinal Forecasting** | Model platform-level shifts in customer engagement styles to allocate retention budgets ahead of multi-year market transitions. |

---

## How It Works

```text
┌────────────────────────────────────────────────────────┐
│               DATA COLLECTION LAYER                    │
│   YouTube API v3   │   Bluesky AT   │  HackerNews API  │
│     (Primary)      │  (Secondary)   │    (Tertiary)    │
└───────────────────────────┬────────────────────────────┘
                            │ Raw Records (n = 12,940)
                            ▼
┌────────────────────────────────────────────────────────┐
│             PRISMA 2020 SYSTEMATIC FILTER              │
│   1. Identification   2. Screening                     │
│   3. Eligibility      4. Included Dataset (n = 5,694)  │
└───────────────────────────┬────────────────────────────┘
                            │ Curated Text
             ┌──────────────┴──────────────┐
             ▼                             ▼
┌─────────────────────────┐   ┌──────────────────────────┐
│   RoBERTa (Sentiment)   │   │  BERTopic (Topic Model)  │
│  neg / neu / pos probs  │   │  44 semantic clusters    │
└────────────┬────────────┘   └────────────┬─────────────┘
             └──────────────┬──────────────┘
                            │ Unified Feature Matrix
                            ▼
┌────────────────────────────────────────────────────────┐
│            RANDOM FOREST CLASSIFIER (200 Trees)        │
│        Accuracy: 95.08%   |   Weighted AUC: 0.9945     │
└───────────────────────────┬────────────────────────────┘
                            │
             ┌──────────────┴──────────────┐
             ▼                             ▼
┌─────────────────────────┐   ┌──────────────────────────┐
│  BEHAVIORAL INFERENCE   │   │  2027–2030 PROJECTIONS   │
│  • Purchase Intent      │   │  Purchase Intent  ↘      │
│  • Passive Browsing     │   │  Passive Browsing ↘      │
│  • Churn Risk           │   │  Churn Risk       ↗      │
└─────────────────────────┘   └──────────────────────────┘
```

The pipeline operates across four coordinated phases:

1. **Systematic Ingestion & Cleaning:** Raw data is captured across APIs into a structured PostgreSQL database. The PRISMA filter screens records against automated inclusion/exclusion criteria (removing bots, irrelevant keywords, non-English texts, and short/spam content), yielding a reliable 44.0% retention rate of high-signal discourse.
2. **Dual-Channel NLP Feature Engineering:** Text is analyzed along two axes:
   - **Valence/Affect:** Fine-grained sentiment probability vectors (`negative`, `neutral`, `positive`) via `cardiffnlp/twitter-roberta-base-sentiment-latest`.
   - **Thematic Context:** Dense semantic clustering via Sentence-Transformers, UMAP dimensionality reduction, and HDBSCAN topic isolation via BERTopic.
3. **Behavioral Classification:** The engineered features (probabilities, topic distributions, and platform interaction metadata) are fed into a balanced Random Forest model to classify records into `purchase_intent`, `passive_browsing`, or `churn_risk`.
4. **Longitudinal Trend Modeling:** Historic and cross-sectional behavioral frequencies are extrapolated using linear trend regression to establish market behavior forecasts through 2030.

---

## Experimental Findings

### Model Evaluation

Evaluated via 5-fold cross-validation and an independent 20% holdout test set:

| Metric | Holdout Test Score | 5-Fold Cross-Validation |
|---|---|---|
| **Accuracy** | **0.9508** | **0.9464 ± 0.0057** |
| **Weighted F1** | **0.9508** | — |
| **Weighted AUC-ROC** | **0.9945** | — |
| **Analyzed Corpus** | 5,694 curated records (from 12,940 raw) | — |
| **Identified Topics** | 44 distinct semantic clusters | — |

### Behavioral Projections (2027–2030)

| Behavioral Class | Baseline Distribution | 2030 Projected Share | Multi-Year Trend |
|---|---|---|---|
| **Passive Browsing** | 49.70% | 42.92% | Contracting (-6.78%) |
| **Purchase Intent** | 29.50% | 26.10% | Contracting (-3.40%) |
| **Churn Risk** | 20.80% | 30.99% | **Expanding (+10.19%)** |

> **Key Insight:** Consumer discourse demonstrates a projected ~10% surge in churn indicators by 2030, highlighting an urgent shift from exploratory browsing toward heightened customer sensitivity and product attrition.

---

## Repository Structure

```text
predictive-discourse-analysis/
├── collectors/                   # API scrapers and ingestion scripts
│   ├── youtube_collector.py      # YouTube Data API v3 integration
│   ├── bluesky_collector.py      # AT Protocol firehose/search client
│   └── hackernews_collector.py   # Algolia search API client
├── output_csv/                   # Pipeline artifacts and evaluation tables
│   ├── prisma/                   # Stage-by-stage inclusion/exclusion logs
│   ├── nlp/                      # Extracted feature vectors and topic models
│   ├── random_forest/            # Model weights, ROC curves, confusion matrices
│   └── comparison_viz/           # Feature importance & NLP comparative plots
├── config.py                     # Central platform, collection, and storage config
├── database.py                   # PostgreSQL connection pooling and ORM schema
├── main.py                       # Unified collection orchestrator
├── prisma_keywords.py            # Systematic inclusion/exclusion rules
├── prisma_filter.py              # 4-stage PRISMA 2020 automated screening
├── nlp_config.py                 # Hyperparameters for RoBERTa and BERTopic
├── nlp_extraction.py             # Parallel NLP extraction pipeline
├── rf_config.py                  # Random Forest hyperparameters & cross-val settings
├── random_forest.py              # Model training, validation, and artifact export
├── projection_analysis.py        # 2027–2030 behavioral forecasting engine
├── nlp_comparison_viz.py         # Visual comparison generators
├── reset_database.py             # Database management and schema cleanup utility
├── requirements.txt              # Pinned Python package dependencies
└── .env.example                  # Environment variable template
```

---

## Prerequisites

- **Python:** 3.10 or higher
- **Database:** PostgreSQL 14 or higher
- **Hardware:** Minimum 8 GB RAM (16 GB RAM or a CUDA-compatible GPU recommended for BERTopic and RoBERTa embeddings)
- **API Credentials:**
  - YouTube Data API v3 key (Google Cloud Console)
  - Bluesky account handle and app password (bsky.app)
  - HackerNews (public Algolia endpoint; no token required)

---

## Getting Started

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/kattcraftt/predictive-discourse-analysis.git
cd predictive-discourse-analysis

# Create and activate virtual environment
python -m venv .venv
# On macOS/Linux:
source .venv/bin/activate  
# On Windows: 
# .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

```ini
# YouTube Data API v3
YOUTUBE_API_KEY=your_youtube_api_key

# Bluesky
BLUESKY_HANDLE=yourhandle.bsky.social
BLUESKY_PASSWORD=your_app_password

# PostgreSQL Configuration
DB_HOST=localhost
DB_PORT=5432
DB_NAME=customer_behavior_db
DB_USER=your_db_user
DB_PASSWORD=your_db_password
```

Create the target database (tables are auto-generated on first run):

```bash
createdb customer_behavior_db
```

---

## Execution Pipeline

Execute the pipeline sequentially. Each module reads clean inputs from PostgreSQL and writes transformed outputs back to the database and `output_csv/`:

### Step 1: Ingest Multi-Source Text
```bash
python main.py
```
Gathers public commentary across YouTube, Bluesky, and HackerNews into `raw_combined_dataset`.

### Step 2: Run Systematic PRISMA Screening
```bash
python prisma_filter.py
```
Executes four-stage filtering based on language, content length, spam heuristics, and domain relevance.

### Step 3: Extract Discourse & Sentiment Features
```bash
python nlp_extraction.py
```
Runs RoBERTa inference for valence estimation and BERTopic for 44-topic clustering.

### Step 4: Train & Validate the Classifier
```bash
python random_forest.py
```
Trains the Random Forest model, runs 5-fold cross-validation, and saves metrics, confusion matrices, and ROC plots to `output_csv/random_forest/`.

### Step 5: Generate Market Projections (2027–2030)
```bash
python projection_analysis.py
```
Forecasts behavioral trajectories through 2030 and outputs tabular summaries.

### Step 6: Generate Analytical Visualizations
```bash
python nlp_comparison_viz.py
```
Creates comparative visual artifacts illustrating sentiment distributions, topic-behavior heatmaps, and relative NLP feature contributions.

---

## Database Management

To inspect dataset status, clear specific staging tables, or truncate tables without writing manual queries, use the interactive helper:

```bash
python reset_database.py
```

---

## PRISMA 2020 Filtering Framework

| Stage | Criteria Applied | Record Retention |
|---|---|---|
| **1. Identification** | Cross-platform multi-query API retrieval (YouTube, Bluesky, HackerNews) | 12,940 (100.0%) |
| **2. Screening** | Duplicate text detection, non-English exclusion, length normalization (>30 chars) | 12,690 (98.1%) |
| **3. Eligibility** | Domain keyword compliance and automated spam/bot heuristic filtering | 5,694 (44.0%) |
| **4. Included** | Full semantic convergence for downstream RoBERTa and BERTopic modeling | 5,694 (44.0%) |

---

## Hyperparameter Specifications

- **RoBERTa:** `cardiffnlp/twitter-roberta-base-sentiment-latest`, max token length: 512, batch size: 32.
- **BERTopic:** Embeddings: `all-MiniLM-L6-v2`, UMAP: `n_neighbors=15, n_components=5, metric='cosine'`, HDBSCAN: `min_cluster_size=10, method='eom'`.
- **Random Forest:** `n_estimators=200`, `max_depth=None`, `min_samples_split=5`, `min_samples_leaf=2`, `class_weight='balanced'`, `cv=5`.

---

## License

This project is licensed under the [MIT License](LICENSE).
