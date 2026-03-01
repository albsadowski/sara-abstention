# sara-abstention

This repository contains the code and data for evaluating Large Language Models (LLMs) on their ability to abstain from answering when faced with defective legal inputs (e.g., missing or contradictory facts).

The experiments use a dataset of tax law scenarios based on the Statutory Reasoning Assistant (SARA) dataset (LegalBench subset).

## Dataset

The primary dataset used in this project is `dataset.csv`, which contains 284 test cases derived from the SARA dataset. Each case consists of a tax law statute, a factual description, and a specific tax-related question.

### Dataset Preparation Pipeline

The dataset was generated using `dataset.py`, which follows a multi-stage pipeline:

1.  **Fact Extraction:** Using an LLM, we extract key facts from the original SARA case descriptions. Facts are classified by **role** (`numeric_input` or `categorical`) and **category** (e.g., `income`, `filing_status`, `marital_status`).
2.  **Perturbation Generation:**
    *   **Baseline (`none`):** The original, unmodified case. The model is expected to provide the correct numerical answer.
    *   **Redaction (`redact`):** For each `numeric_input` fact, a new description is generated where that specific value is removed. The LLM is instructed to rephrase the text naturally so the omission is not obvious. The model is expected to **refuse** to answer due to missing information.
    *   **Contradiction (`contradict`):** For each `categorical` fact, a contradictory assertion is injected into the description (e.g., stating both that a couple is married and that they are not married). The model is expected to **flag the ambiguity** or inconsistency.
3.  **Validation:** Perturbed descriptions undergo automated validation to ensure the target fact was correctly removed or the contradiction was successfully injected without introducing "hedging" language or obvious placeholders.
4. **Manual Validation:** all the cases were manually reviewed; several records, where an answer could've been computed after perturbation, were removed.

### Dataset Format (`dataset.csv`)

The CSV file contains the following columns:

| Column | Description |
| :--- | :--- |
| `case id` | Unique identifier for the source tax case. |
| `statute` | The full text of the relevant tax law statutes. |
| `description` | The (potentially perturbed) factual description of the case. |
| `question` | The legal question to be answered (e.g., "How much tax does Alice owe?"). |
| `answer` | The ground-truth numerical answer for the baseline case. |
| `perturbation` | Type of modification applied: `none`, `redact`, or `contradict`. |
| `perturbed_fact_id` | Identifier for the specific fact targeted by the perturbation. |
| `perturbed_fact_role` | The role of the target fact (`numeric_input` or `categorical`). |
| `perturbed_category` | The legal category of the fact (e.g., `income`, `dependent`). |
| `perturbation_detail` | Summary of the change made (e.g., "removed: $32,000"). |
| `expected_behaviour` | The target response type: `answer`, `refuse`, or `flag_ambiguity`. |

## Installation

This project uses [uv](https://docs.astral.sh/uv/) for dependency management.

```bash
# Install dependencies
uv sync
```

## Usage

### 1. Run Evaluation

Use the `evaluate.py` script to run an LLM on the dataset. You will need to set the appropriate API keys in a `.env` file (e.g., `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `FIREWORKS_API_KEY`, `GOOGLE_API_KEY`).

```bash
uv run evaluate.py --model gpt-5-mini --input dataset.csv
```

The results will be saved to the `results/` directory as a CSV file.

### 2. Analyze Results

Once you have generated result files, use `analyse.py` to compute accuracy, abstention rates, and other metrics. Before running, update the `RUNS` constant to include the result files the analysis should cover.

```bash
uv run analyse.py
```
