# Data Folder

This folder contains the dataset preparation pipeline, reference knowledge base, and evaluation datasets for a psychiatry-focused medical question-answering project. The pipeline curates a psychiatric subset from the MedQA-Open dataset using a three-stage LLM-based classification process, then stores the results alongside clinical reference PDFs for RAG-based evaluation.

---

## Scripts and Files

| File | Description |
|------|-------------|
| `dataset_preparation_01_filter_psychiatry_questions.py` | First-stage filter that screens the full MedQA-Open dataset (~10,000 questions) using the Gemini API to identify psychiatry-related questions. Acts as a fast, cost-efficient preliminary pass before deeper analysis. |
| `dataset_preparation_02_verify_psychiatric_question.py` | Second-stage verifier that applies a stricter clinical relevance check to questions flagged in step 1. Classifies each question as INCLUDE or EXCLUDE based on whether it is primarily focused on clinical psychiatry. |
| `dataset_preparation_03_assign_psychiatric_categories.py` | Assigns each verified question to one of 11 psychiatric categories (e.g., Anxiety, Bipolar, Schizophrenia). Outputs structured JSON with category labels and confidence scores. |
| `dataset_preparation_classification_and_filtering.ipynb` | End-to-end notebook documenting the full pipeline: loading MedQA-Open, filtering to 737 clinically verified psychiatry questions, assigning categories, and producing train/validation/test splits. |

---

## Pipeline Overview

```
MedQA-Open (~10,178 questions)
        │
        ▼
Step 1 — Psychiatry filter       → ~886 candidate questions
        │
        ▼
Step 2 — Clinical focus check    → 737 verified questions
        │
        ▼
Step 3 — Category assignment     → 737 labelled questions
        │
        ▼
Train / Validation / Test split  → test_dataset.csv
```
