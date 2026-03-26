# LLM-Driven Adaptive Learning Systems
## Designing LLM-Driven Adaptive Learning Systems for Personalised Curriculum Development in Higher Education

**MSc Web and Data Science — University of Koblenz — 2025-26**  
**Team 4:** Sanjay Selvam Umadevi · Muhammed Sabik Kunnummal · Shirisha Shivakumar · Alisha Kabeer

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Repository Structure](#repository-structure)
3. [Requirements](#requirements)
4. [Installation](#installation)
5. [Dataset Setup](#dataset-setup)
6. [API Keys Configuration](#api-keys-configuration)
7. [How to Run](#how-to-run)
8. [Pipeline Walkthrough](#pipeline-walkthrough)
9. [Expected Outputs](#expected-outputs)
10. [Reproducibility Notes](#reproducibility-notes)
11. [Troubleshooting](#troubleshooting)

---

## Project Overview

This project implements an end-to-end pipeline that integrates machine learning analytics with LLM-driven personalised curriculum generation. Using the Open University Learning Analytics Dataset (OULAD), the system:

- Predicts student dropout risk using XGBoost (85% accuracy, ROC-AUC 0.94)
- Detects skill gaps via Z-score analysis across 173,912 assessments
- Constructs an LLM-integrated Course-Concept Knowledge Graph using GPT-4o-mini
- Generates personalised 6-week curricula using GPT-4o-mini and LLaMA-3.1-8B
- Evaluates curriculum quality using an LLM-as-Judge framework (9 metrics)

**Key result:** 43% quality improvement over naive baseline (5.4/10 → 7.7/10)

---

## Repository Structure

```
project/
│
├── Final_Code_reviewed.ipynb     ← Main notebook (run this)
├── README.md                     ← This file
│
├── data/
│   └── archive/                  ← Place all 7 OULAD CSV files here
│       ├── assessments.csv
│       ├── courses.csv
│       ├── studentAssessment.csv
│       ├── studentInfo.csv
│       ├── studentRegistration.csv
│       ├── studentVle.csv
│       └── vle.csv
│
└── outputs/                      ← Generated graphs and plots saved here
    ├── llm_knowledge_graph.png
    ├── student_gap_graph.png
    └── ...
```

---

## Requirements

### Python Version
```
Python 3.10 or higher
```

### Required Libraries

All libraries are installed in **Cell 1** of the notebook. You can also install them manually:

```bash
pip install networkx
pip install google-genai
pip install openai
pip install groq
pip install shap
pip install huggingface_hub
pip install xgboost
pip install pandas
pip install numpy
pip install matplotlib
pip install seaborn
pip install scikit-learn
```

### Full requirements summary

| Library | Version | Purpose |
|---|---|---|
| pandas | ≥ 1.5 | Data loading and merging |
| numpy | ≥ 1.23 | Numerical computation |
| matplotlib | ≥ 3.6 | Visualisation |
| seaborn | ≥ 0.12 | Statistical plots |
| scikit-learn | ≥ 1.2 | Train/test split, metrics, label encoding |
| xgboost | ≥ 1.7 | Pass/fail classifier + disengagement classifier |
| shap | ≥ 0.43 | Feature importance explanation |
| networkx | ≥ 3.0 | Course-Concept Knowledge Graph construction |
| openai | ≥ 1.0 | GPT-4o-mini API calls (graph + curriculum + judge) |
| huggingface_hub | ≥ 0.20 | LLaMA-3.1-8B inference via Cerebras |
| google-genai | any | Optional Google AI support |
| groq | any | Optional Groq support |

---

## Installation

### Step 1 — Clone or unzip the project

```bash
# If using git:
git clone <repository-url>
cd project/
```

### Step 2 — (Optional) Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows
```

### Step 3 — Install dependencies

Either run **Cell 1** inside the notebook, or install manually:

```bash
pip install networkx google-genai openai groq shap huggingface_hub xgboost pandas numpy matplotlib seaborn scikit-learn
```

### Step 4 — Launch Jupyter

```bash
jupyter notebook
# or
jupyter lab
```

Open `Final_Code_reviewed.ipynb`.

---

## Dataset Setup

The OULAD dataset is **not included** in this repository due to file size. You must download it separately.

### Download Instructions

1. Go to: **[https://analyse.kmi.open.ac.uk/open_dataset](https://analyse.kmi.open.ac.uk/open-dataset)**
2. Click **Download Dataset**
3. Extract the ZIP file
4. You will get 7 CSV files

### Place the files

Copy all 7 CSV files into the `data/archive/` folder:

```
data/
└── archive/
    ├── assessments.csv
    ├── courses.csv
    ├── studentAssessment.csv
    ├── studentInfo.csv
    ├── studentRegistration.csv
    ├── studentVle.csv
    └── vle.csv
```

### Update the path in the notebook

In **Cell 4**, update `BASE_PATH` to match where you placed your data:

```python
BASE_PATH = "../data/archive"   # ← change this if needed
```

**Examples:**
```python
BASE_PATH = "data/archive"                    # if running from project root
BASE_PATH = "/Users/yourname/data/archive"    # absolute path (Mac/Linux)
BASE_PATH = "C:/Users/yourname/data/archive"  # absolute path (Windows)
```

### Dataset statistics (for verification)

After loading in Cell 4, you should see:

| File | Rows (approx.) |
|---|---|
| studentInfo.csv | 32,593 |
| studentAssessment.csv | 173,912 |
| assessments.csv | 206 |
| studentVle.csv | ~10 million |
| vle.csv | ~6,364 |
| studentRegistration.csv | ~32,593 |
| courses.csv | 22 |

---

## API Keys Configuration

This project requires **two API keys**. You must replace the placeholder keys in the notebook before running.

> ⚠️ **Important:** The API keys currently in the notebook are the team's own keys. Replace them with your own keys before running to avoid hitting rate limits.

---

### API Key 1 — OpenAI (GPT-4o-mini)

Used for:
- LLM-integrated Course-Concept Knowledge Graph (Cell 53)
- Curriculum generation — Advanced OpenAI condition (Cell 71–73)
- LLM-as-Judge evaluation (Cell 79)

**How to get your key:**
1. Go to https://platform.openai.com/api-keys
2. Click **Create new secret key**
3. Copy the key (starts with `sk-`)

**Where to put it — update in TWO places:**

**Cell 53:**
```python
client = openai.OpenAI(api_key="YOUR_OPENAI_API_KEY_HERE")
```

**Cell 71:**
```python
client = OpenAI(api_key="YOUR_OPENAI_API_KEY_HERE")
```

> **Note:** GPT-4o-mini API calls cost approximately $0.002–$0.01 per curriculum generation. Running the full multi-student comparison (Cell 79) for 5 students across 3 conditions costs approximately $0.05–$0.15 total.

---

### API Key 2 — HuggingFace (LLaMA-3.1-8B via Cerebras)

Used for:
- Curriculum generation — Advanced HuggingFace condition (Cell 71, 75)

**How to get your key:**
1. Go to https://huggingface.co/settings/tokens
2. Click **New token**
3. Select **Read** access
4. Copy the token (starts with `hf_`)

**Where to put it — Cell 71:**
```python
hf_client = InferenceClient(
    api_key="YOUR_HUGGINGFACE_TOKEN_HERE"
)
```

> **Note:** HuggingFace inference via Cerebras is **free** for LLaMA-3.1-8B. No billing required.

---

## How to Run

### Run the full pipeline (recommended)

Open `Final_Code_reviewed.ipynb` and run all cells in order from top to bottom using:

**Kernel → Restart & Run All**

Total runtime: approximately **5–15 minutes** depending on internet speed for API calls.

---

### Cell-by-cell walkthrough

| Cells | Section | What happens |
|---|---|---|
| 1 | Installation | Installs all required libraries |
| 2 | Imports | Loads all Python libraries |
| 3–4 | Data Loading | Reads all 7 OULAD CSV files |
| 5–15 | Preprocessing | Merges CSVs, engineers 13 features |
| 16–18 | EDA | Plots distribution of final results |
| 19–26 | XGBoost | Trains pass/fail classifier, plots ROC + confusion matrix + feature importance |
| 27–30 | Dropout Risk | Computes dropout_risk = 1 − P(pass), plots distribution |
| 31–32 | SHAP | Runs SHAP analysis on trained XGBoost model |
| 33–35 | Disengagement | Trains secondary disengagement classifier |
| 36–50 | Z-Score Gap Detection | Detects skill gaps, computes severity, classifies TMA/CMA |
| 51–60 | Knowledge Graph | Builds LLM-integrated Course-Concept Graph using GPT-4o-mini |
| 61–69 | Prompt Builder | Builds advanced prompt with 5 adaptive rules |
| 70–75 | Curriculum Generation | Generates 6-week curricula (OpenAI + HuggingFace) |
| 76–77 | Single Student Output | Prints curriculum for one selected student |
| 78–79 | LLM Judge | Runs multi-student comparison and scoring across 9 metrics |

---

### Running for a specific student (Cell 65 & 77)

When Cell 65 runs, you will be prompted:

```
Enter 1 to provide Student ID manually or 2 for random selection:
```

- Enter **1** to type a specific student ID (e.g. `28400`)
- Enter **2** to select a random student from the dataset

---

## Pipeline Walkthrough

```
┌─────────────────────────────────────────────────────────┐
│                     ML LAYER                            │
│                                                         │
│  OULAD CSVs → Feature Engineering → XGBoost Classifier │
│       ↓                                    ↓            │
│  Z-Score Skill Gap Detection          Dropout Risk Score│
│  (17,704 students flagged)            (dropout_risk =   │
│  weakness_type: TMA or CMA             1 − P(pass))     │
│  weak_concepts: [concept list]                          │
└───────────────────┬─────────────────────────────────────┘
                    │ student profile
                    ↓
┌─────────────────────────────────────────────────────────┐
│                    LLM LAYER                            │
│                                                         │
│  Course-Concept Knowledge Graph (GPT-4o-mini)           │
│       ↓                                                 │
│  Advanced Prompt Builder (5 mandatory adaptive rules)   │
│       ↓                    ↓                            │
│  GPT-4o-mini           LLaMA-3.1-8B                    │
│  (3000 tokens)         (1000 tokens, free)              │
│       ↓                    ↓                            │
│       └────────┬───────────┘                            │
│                ↓                                        │
│  LLM-as-Judge (GPT-4o-mini, temp=0, 9 metrics)         │
│  Scores: 5.4/10 → 6.7/10 → 7.7/10                     │
└─────────────────────────────────────────────────────────┘
```

---

## Expected Outputs

After running all cells, you should see the following outputs:

### Plots and visualisations
- Distribution of Final Results (bar chart)
- XGBoost Confusion Matrix
- ROC Curve (AUC = 0.94)
- Feature Importance Ranking (horizontal bar chart)
- Dropout Risk Score Distribution (histogram with threshold at 0.6)
- SHAP Beeswarm Plot
- Distribution of Skill Gap Severity
- LLM-Generated Course-Concept Knowledge Graph (`llm_knowledge_graph.png`)
- Student-specific Gap Graph (`student_<id>_gap_graph.png`)
- Multi-Student LLM Judge Comparison Chart (bar + radar chart)

### Printed outputs
```
Accuracy: ~0.85
High Dropout Risk Students: 12,787
Low Dropout Risk Students: 19,891
Students with at least 1 skill gap: 17,704
Average gap count per flagged student: 4.3
```

### LLM Judge scores (approximate, may vary slightly)

| Condition | Overall Score |
|---|---|
| Baseline (Normal Prompt) | ~5.4 / 10 |
| Advanced OpenAI (GPT-4o-mini) | ~6.7 / 10 |
| Advanced HuggingFace (LLaMA-3.1-8B) | ~7.7 / 10 |

> Small variations (±0.2) in LLM judge scores across runs are normal due to non-deterministic LLM output in curriculum generation. The judge itself uses temperature=0 for consistency.

---

## Reproducibility Notes

### Random seeds

The following random seed is used throughout for reproducibility:

```python
random_state=42   # used in train_test_split (Cell 23 and Cell 35)
```

### XGBoost configuration (Cell 23)

```python
model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="logloss"
)
```

### LLM determinism

- **LLM Judge (Cell 79):** temperature=0 — fully deterministic, scores will be identical across runs
- **Curriculum generation (Cell 73, 75):** temperature not fixed — minor variations in generated text are expected but should not significantly affect judge scores
- **Knowledge Graph (Cell 55):** temperature=0.2 — graph structure is largely stable across runs

### Known variation sources

The 43% quality improvement result (5.4 → 7.7) is stable across runs. Individual metric scores may vary by ±0.2 due to curriculum generation temperature. Running on fewer than 5 students will produce different averages — the paper reports results for exactly 5 students.

---

## Troubleshooting

### "FileNotFoundError: assessments.csv not found"
→ Check that `BASE_PATH` in Cell 4 points to the correct folder containing all 7 CSV files.

### "AuthenticationError: Invalid API key"
→ Replace the API keys in Cell 53 and Cell 71 with your own valid keys.

### "RateLimitError" from OpenAI
→ You have hit your OpenAI rate limit. Wait 60 seconds and retry, or upgrade your OpenAI plan. The multi-student comparison in Cell 79 makes approximately 15 API calls.

### "HF Error: 429 Too Many Requests"
→ HuggingFace Cerebras free tier has rate limits. Add a short `time.sleep(5)` between calls in Cell 79, or run fewer students.

### SHAP plot shows blank or crashes
→ Ensure `shap >= 0.43` is installed. Re-run Cell 1 and restart the kernel.

### Knowledge graph shows no nodes
→ The LLM returned malformed JSON. Re-run Cell 59. If the problem persists, check your OpenAI API key balance.

### "KeyError: code_module_x"
→ This is handled by Cell 49 which renames the duplicate column. Ensure you run cells in order without skipping.

### Jupyter kernel crashes on studentVle.csv
→ The VLE file has ~10 million rows and requires approximately 2–4 GB of RAM. Close other applications and retry.

---

## Citation

If you use this work, please cite:

```
Selvam Umadevi S., Kunnummal M.S., Shivakumar S., Kabeer A. (2025).
Designing LLM-Driven Adaptive Learning Systems for Personalised
Curriculum Development in Higher Education.
MSc Web and Data Science Research Project,
University of Koblenz, Germany.
```

---

## Dataset Citation

```
Kuzilek J., Hlosta M., Zdrahal Z. (2017).
Open University Learning Analytics dataset.
Scientific Data, 4, 170171.
https://doi.org/10.1038/sdata.2017.171
```

---

*For questions about this project, contact the authors via the University of Koblenz.*
