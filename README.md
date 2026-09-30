# GLU — Global-Local Uncertainty for LLMs

Code for the paper **"Integrating Local and Global Entropy for Uncertainty Quantification in LLMs"**.

GLU is a training-free uncertainty score that combines two complementary signals extracted from a single greedy generation:

- **Local uncertainty** — per-token Shannon-weighted epistemic uncertainty derived from Evidence Deep Learning (EDL), captured by `she_R_mean` (mean over the k worst-scoring tokens).
- **Global geometric uncertainty** — normalized Matrix Rényi entropy `S̃` of the hidden-state trajectory across generated tokens, measuring how much the model's internal representation "spreads out" over the sequence.

The final score is:

```
GLU = (1 + S̃) * she_R_mean
```

`she_R_mean` is negative and `(1 + S̃) ≥ 1`, so GLU is always ≤ 0. A **more negative** `GLU` indicates **higher** uncertainty. AUROC is computed with GLU as the score against correctness labels (`label = 1` means correct).

---

## Repository structure

```
GLU/
├── generate_response.py   # Step 1 — run models, collect UQ features, judge correctness
├── GLU.py                 # Step 2 — AUROC of the baseline GLU score per model–dataset
├── GLU_ablations.py       # Step 2 — full ablation table across all GLU variants
├── .env.example           # Azure OpenAI credentials template (for the LLM judge)
├── small_datasets/
│   ├── triviaqa_small.csv
│   ├── truthfulqa.csv
│   ├── math_small.csv
│   ├── arabica_qa.csv
│   ├── longform_subset.csv
│   └── multiturn_math.csv       # not yet wired into generate_response.py
└── output/
    └── full/              # created automatically; all Step 1 outputs land here
```

---

## Step 1 — Generate responses and compute UQ features

`generate_response.py` loads a model once, runs greedy generation on every dataset sample, and writes a JSONL file with all UQ features plus an LLM-judge correctness label.

### Prerequisites

```bash
pip install torch transformers datasets openai scikit-learn scipy pandas tqdm python-dotenv numpy
```

Copy `.env.example` to `.env` and fill in your Azure OpenAI credentials:

```bash
cp .env.example .env
```

### Usage

```bash
python generate_response.py \
    --model  <fanar|gemma|qwen> \
    --datasets <triviaqa|triviaqa_full|truthfulqa|math|arabica|longform> [<dataset2> ...] \
    [--max-new-tokens 256] \
    [--k 10] \
    [--alpha 2.0] \
    [--layer-idx -1] \
    [--seed 0] \
    [--rauq-alpha 0.2] \
    [--rauq-use-hooks] \
    [--max-samples N] \
    [--output-suffix "-v2"] \
    [--test]
```

| Argument | Default | Description |
|---|---|---|
| `--model` | required | One of `fanar` (Fanar-1-9B-Instruct), `gemma` (Gemma-3-12B-IT), `qwen` (Qwen2.5-7B-Instruct) |
| `--datasets` | required | One or more dataset keys (space-separated) |
| `--max-new-tokens` | 256 | Max tokens to generate per sample |
| `--k` | 10 | Top-k for logit entropy and EDL |
| `--alpha` | 2.0 | Rényi order for matrix entropy |
| `--layer-idx` | -1 | Hidden-state layer recorded as `layer_idx_chosen` (negative indexes from the end) |
| `--seed` | 0 | RNG seed for generation |
| `--rauq-alpha` | 0.2 | Recurrence weight in RAUQ |
| `--rauq-use-hooks` | off | Memory-efficient RAUQ via forward hooks |
| `--max-samples` | all | Cap number of samples processed |
| `--output-suffix` | `""` | Extra suffix on output filenames |
| `--test` | off | Run 5 samples with verbose output |

### Quick smoke test

```bash
python generate_response.py --model qwen --datasets triviaqa --test
```

### Outputs

Each run writes two files to `./output/full/`:

| File | Contents |
|---|---|
| `{model}-{dataset}-unified-uq-full{suffix}.jsonl` | One JSON object per sample: response, correctness label, all UQ scores, per-layer geometry, per-token breakdown, top-100 logits |
| `{model}-{dataset}-unified-uq-full{suffix}-summary.json` | AUROC summary for all metrics + per-layer `S_alpha` AUROC |

Rows are flushed as they are generated, so a partially finished run still leaves a valid JSONL file. Samples that fail generation are skipped and not written. `label` is `null` when the judge returns something other than `0`/`1`.

The script also prints live AUROC figures at the end of each dataset.

### Fields in each JSONL row

| Field(s) | Description |
|---|---|
| `id`, `prompt`, `response`, `ground_truth`, `label` | Sample index, question, model response, reference answer, judge label (1 = correct) |
| `T`, `num_prompt_tokens`, `num_layers`, `layer_idx_chosen` | Generation length and model/layer metadata |
| `U_logit`, `U_max`, `U_shannon` | Mean / max / combined top-k Shannon entropy |
| `R_mean`, `R_worst`, `U_edl` | EDL reliability (mean / worst k tokens, combined) |
| `she_R_mean`, `she_R_worst`, `she_U` | Shannon-EU variants of the EDL scores |
| `u_rauq`, `u_rauq_per_layer` | RAUQ attention-based uncertainty |
| `S_alpha`, `S_tilde` | Global Matrix Rényi entropy (raw and normalized, averaged over layers) |
| `S_alpha_per_layer`, `S_tilde_per_layer` | Per-layer versions of the above (lists) |
| `eigenvalues_per_layer` | Eigenvalue spectra of the hidden-state Gram matrix |
| `id_per_layer` | TwoNN intrinsic dimensionality per layer |
| `token_data` | Per-token `au`, `eu`, `collision_entropy`, `logtoku`, `coletoku` |
| `top100_logits`, `top100_token_ids`, `top100_probs` | Top-100 logit values, token IDs, and probabilities per token (`T × 100`) |

List and dict fields are stored as native JSON, so no extra decoding is needed. To load a file:

```python
import pandas as pd
df = pd.read_json("output/full/qwen-triviaqa-unified-uq-full.jsonl", lines=True)
```

---

## Step 2 — Compute GLU and ablations

Both scripts read every `output/full/*-unified-uq-full.jsonl` file produced by Step 1. Run them from the repo root. Rows with a `null` label are dropped before computing AUROC.

> Files written with a non-empty `--output-suffix` (e.g. `...-unified-uq-full-v2.jsonl`) don't match this pattern and are ignored.

### `GLU.py` — single GLU score per file

Prints `AUROC(GLU)` per model–dataset pair and the number of dropped rows.

```bash
python GLU.py
```

### `GLU_ablations.py` — full ablation table

Groups files by dataset and skips incomplete runs (files with fewer rows than the most complete run for that dataset). It then prints a table of AUROC scores for all GLU variants side by side.

```bash
python GLU_ablations.py
```

#### Ablation variants

| Column | Formula | What changes |
|---|---|---|
| `GLU` | `(1 + S̃_mean) * she_R_mean` | Baseline |
| `GLU_rawS_mean` | `(1 + S_α_mean) * she_R_mean` | Raw (un-normalized) Rényi entropy |
| `GLU_rawS_best` | `(1 + S_α_best) * she_R_mean` | Max-over-layers raw entropy instead of mean |
| `GLU_stilde_best` | `(1 + S̃_best) * she_R_mean` | Max-over-layers entropy, normalized by `1 + log T` |
| `GLU_softplus` | `(1 + S̃) * she_R_mean_sp` | Softplus evidence (vs. ReLU+1) |
| `GLU_dynK` | `(1 + S̃) * she_R_mean_dk` | Dynamic k: k₁ = k / (1 + S̃) |
| `GLU_AU` | `(1 + S̃) * she_R_mean_au` | Aleatoric only (drops epistemic EU weight) |

---

## Supported models and datasets

| Key | Model |
|---|---|
| `fanar` | `QCRI/Fanar-1-9B-Instruct` |
| `gemma` | `google/gemma-3-12b-it` |
| `qwen` | `Qwen/Qwen2.5-7B-Instruct` |

| Key | Dataset | File |
|---|---|---|
| `triviaqa` | TriviaQA (small split) | `triviaqa_small.csv` |
| `triviaqa_full` | TriviaQA (full validation split) | `triviaqa_validation.csv` — not included, add it yourself |
| `truthfulqa` | TruthfulQA | `truthfulqa.csv` |
| `math` | MATH (small split) | `math_small.csv` |
| `arabica` | Arabic QA | `arabica_qa.csv` |
| `longform` | Long-form QA subset | `longform_subset.csv` |
