# Analysis Folder

This folder contains the evaluation and results layer of the dissertation. It compares RAG (Retrieval-Augmented Generation) versus vanilla LLM responses on a psychiatry question dataset across four large language models, using multiple automated and rubric-based metrics with statistical testing.

---

## Files

| File | Description |
|------|-------------|
| `model_cards.ipynb` | Incomplete Reference notebook documenting the LLMs used in the evaluation. Includes links to official model cards and key technical specifications. |
| `results_overview.ipynb` | Main analysis notebook comparing vanilla and RAG-augmented model performance using RAGAS metrics (Answer Semantic Similarity, Answer Relevance, Faithfulness, Context Recall, Context Precision) and a 1–4 rubric-based quality score. Includes confidence intervals, paired t-tests, and per-category breakdowns across psychiatric diagnostic groups. |

---

## Analysis Sections (`results_overview.ipynb`)

| Section | Description |
|---------|-------------|
| Dataset Composition | Treemap visualisation of question distribution across 11 psychiatric diagnostic categories. |
| Answer Similarity | Compares vanilla vs. RAG responses on semantic similarity (0–1 scale) with per-model confidence intervals and significance tests. |
| Answer Relevancy | Evaluates how well answers address the question, measuring the effect of context augmentation on relevance. |
| Rubric-Based Quality | Manual 1–4 quality scoring assessing medical accuracy and alignment with ground truth answers. |
| RAG Metrics | Analyses Faithfulness, Context Recall, and Context Precision to assess retrieval quality. |
| Category-Level Analysis | Breaks down all metrics by diagnostic category to identify where RAG helps or hurts. |
| Cluster Analysis | Identifies patterns between semantic similarity scores and rubric quality scores across models. |
