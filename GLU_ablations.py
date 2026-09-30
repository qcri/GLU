"""Comprehensive GLU ablation table.

Covers:
  output/full/*-unified-uq-full.{csv,jsonl}   single-turn
  output/sharded/*-sharded-uq.jsonl            multiturn (final turn only)

Always includes LogProb (free from stored top100_probs).
P(true) is optional — requires a GPU and HuggingFace transformers.

Usage:
  python GLU_ablations_full.py
  python GLU_ablations_full.py --ptrue
  python GLU_ablations_full.py --ptrue --ptrue-models qwen gemma
"""
import argparse
import json
import math
import numpy as np
import pandas as pd
from collections import defaultdict
from glob import glob
from scipy.special import softmax
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

tqdm.pandas()

K   = 10
EPS = 1e-6

MODEL_MAP = {
    "fanar": "QCRI/Fanar-1-9B-Instruct",
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",
    "gemma": "google/gemma-3-12b-it",
}

FIXED_DIR = "output/fixed"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def softplus(x):
    return np.logaddexp(0.0, x)


def _parse(val):
    return json.loads(val) if isinstance(val, str) else val


def load_file(path):
    if path.endswith(".jsonl"):
        return pd.read_json(path, lines=True)
    return pd.read_csv(path)


def load_sharded_final(path):
    """Load a sharded JSONL and keep only final-turn rows."""
    df = load_file(path)
    df = df[df["is_final_turn"] == True].reset_index(drop=True)
    if "prompt" not in df.columns and "original_question" in df.columns:
        df = df.rename(columns={"original_question": "prompt"})
    return df


def _stem(path):
    base = path.split("/")[-1]
    for sfx in ("-unified-uq.jsonl", "-unified-uq.csv"):
        if base.endswith(sfx):
            return base[: -len(sfx)]
    return base


def _is_sharded(path):
    return "sharded" in path.split("/")[-1]


def _model_from_stem(stem):
    return stem.split("-")[0]


def recompute_she_R_mean(df, k=K, use_softplus=False, eps=EPS,
                          dynamic_k=False, au_only=False, eu_only=False):
    results = []
    for _, row in df.iterrows():
        top100  = _parse(row["top100_logits"])
        T       = len(top100)
        s_tilde = float(row["S_tilde"])

        k1    = k / (1.0 + s_tilde) if dynamic_k else float(k)
        k_int = max(1, int(k1))
        k_w   = min(k_int, T)

        shetoku = []
        for tok_logits in top100:
            topk = np.array(tok_logits[:k_int], dtype=np.float64)
            if use_softplus:
                alpha = softplus(topk)
                eu = k1 / (alpha + eps).sum()
            else:
                alpha = np.maximum(topk, 0.0)
                eu = k1 / (alpha + 1.0).sum()
            if eu_only:
                shetoku.append(-eu)
            else:
                probs = softmax(topk)
                se = -np.sum(probs * np.log2(probs.clip(1e-12)))
                shetoku.append(-se if au_only else -(se * eu))

        results.append(np.sort(shetoku)[:k_w].mean())
    return results


def compute_logprob(row) -> float:
    """Mean log probability of the greedy token at each step."""
    probs_list = _parse(row["top100_probs"])
    return sum(math.log(p[0]) for p in probs_list) / row["T"]


def compute_prr(labels, scores):
    """PRR = (AUARC - random) / (oracle - random). Scores ascending = most uncertain first."""
    labels = np.asarray(labels, dtype=float)
    scores = np.asarray(scores, dtype=float)
    n = len(labels)

    order        = np.argsort(scores)
    labels_sorted = labels[order]

    accs    = np.array([labels_sorted[i:].mean() for i in range(n)])
    auarc   = accs.mean()
    baseline = labels.mean()

    n_correct = int(labels.sum())
    n_wrong   = n - n_correct
    oracle_accs = np.empty(n)
    for i in range(n):
        remaining        = n - i
        correct_remaining = (n_correct if i <= n_wrong
                             else n_correct - (i - n_wrong))
        oracle_accs[i]   = correct_remaining / remaining
    oracle_auarc = oracle_accs.mean()

    denom = oracle_auarc - baseline
    return 0.0 if denom < 1e-12 else (auarc - baseline) / denom


# ---------------------------------------------------------------------------
# P(true) — optional, requires GPU + transformers
# ---------------------------------------------------------------------------

def _ptrue_single(question, answer, model, tokenizer, id_0, id_1):
    """Return P(model says '1' | question, answer) in [0, 1] via logit softmax."""
    import torch
    messages = [{"role": "user", "content": (
        f"Question:[{question}] \nProposed Answer:[{answer}] \n"
        "Is the proposed answer:\n1: True\n0: False\nThe proposed answer is: "
    )}]
    inputs = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=True,
        return_dict=True, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        logits = model(**inputs).logits[0, -1, :]
    log_probs = torch.log_softmax(logits[[id_0, id_1]], dim=0)
    return log_probs[1].exp().item()  # P("1")


def compute_ptrue_cache(file_specs, ptrue_model_keys):
    """
    Compute P(true) for all file_specs, loading each model once.
    Returns dict: stem -> pd.Series of float probabilities aligned to the loaded df.
    """
    import torch

    cache = {}
    by_model = defaultdict(list)
    for _, stem, df in file_specs:
        mk = _model_from_stem(stem)
        if mk in ptrue_model_keys:
            by_model[mk].append((stem, df))

    for mk, items in sorted(by_model.items()):
        model_name = MODEL_MAP[mk]
        print(f"\nLoading {model_name} for P(true) judge...")
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(
            model_name, device_map="cuda", torch_dtype=torch.bfloat16
        )
        model.eval()

        _tok_0 = tokenizer.encode("0", add_special_tokens=False)
        _tok_1 = tokenizer.encode("1", add_special_tokens=False)
        assert len(_tok_0) == 1 and len(_tok_1) == 1, \
            f"{model_name}: tokenizer splits '0' or '1' into multiple tokens."
        id_0, id_1 = _tok_0[0], _tok_1[0]

        for stem, df in items:
            print(f"  P(true) judge for {stem} ({len(df)} rows)...")
            cache[stem] = df.progress_apply(
                lambda row: _ptrue_single(
                    row["prompt"], row["response"], model, tokenizer, id_0, id_1),
                axis=1
            )

        del model
        torch.cuda.empty_cache()

    return cache


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser()
parser.add_argument("--models", nargs="+", default=None,
                    help="Only load files whose model key (filename prefix before "
                         "the first '-') is in this list, e.g. --models qwen gemma. "
                         "Default: load all files in the directory.")
parser.add_argument("--ptrue", action="store_true",
                    help="Compute P(true) baseline (requires GPU + transformers)")
parser.add_argument("--ptrue-models", nargs="+", choices=list(MODEL_MAP.keys()),
                    default=list(MODEL_MAP.keys()),
                    dest="ptrue_models",
                    help="Which models to run P(true) for (default: all)")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def discover_complete(raw_files, stem_fn, load_fn):
    by_dataset = defaultdict(list)
    for f in raw_files:
        stem    = stem_fn(f)
        dataset = stem.split("-", 1)[1]
        by_dataset[dataset].append(f)

    complete = []
    for dataset, files in sorted(by_dataset.items()):
        counts    = {f: len(load_fn(f)) for f in files}
        max_count = max(counts.values())
        for f, count in counts.items():
            name = stem_fn(f)
            if max_count - count > 5:
                print(f"[SKIP] {name}: {count}/{max_count} samples")
            else:
                complete.append(f)
    return complete


EXCLUDE_DATASETS = {}

all_raw = sorted(set(
    glob(f"{FIXED_DIR}/*-unified-uq.csv") +
    glob(f"{FIXED_DIR}/*-unified-uq.jsonl")
))

all_raw = [f for f in all_raw if _stem(f).split("-", 1)[1] not in EXCLUDE_DATASETS]

if args.models:
    wanted  = set(args.models)
    all_raw = [f for f in all_raw if _model_from_stem(_stem(f)) in wanted]

def _load(path):
    return load_sharded_final(path) if _is_sharded(path) else load_file(path)

complete = discover_complete(all_raw, _stem, _load)

# Pre-load all dataframes once (avoids double I/O with ptrue pre-computation)
file_specs = [(f, _stem(f), _load(f)) for f in complete]

n_sharded = sum(1 for f in complete if _is_sharded(f))
print(f"\nFiles:         {len(complete)}  (sharded: {n_sharded})")
print()


# ---------------------------------------------------------------------------
# P(true) pre-computation (optional)
# ---------------------------------------------------------------------------

ptrue_cache = {}
if args.ptrue:
    ptrue_cache = compute_ptrue_cache(file_specs, set(args.ptrue_models))


# ---------------------------------------------------------------------------
# Ablation columns
# ---------------------------------------------------------------------------

ABLATIONS = [
    # ---- Non-geometric baselines ----
    # mean of K worst logtoku_t = -(au_t * eu_t): au = Dirichlet expected entropy via digamma, eu = k/Σ(ReLU(logit_j)+1)
    ("R_mean",                 "LogTokU"),
    # per-layer: best sub-diagonal head recurrence c_i = α·p(y_i) + (1-α)·attn_{i-1}·c_{i-1}; u = max_layer mean(-log c); negated here so higher = more certain
    # ("RAUQ",                   "RAUQ"),
    # (1/T) * Σ_t log(top100_probs[t][0]): mean log prob of the greedy token at each step
    ("logprob",                "LogProb"),

    ("she_R_mean", "local-only"),
    ("S_tilde",     "global-only"),

    # ---- Proposed method ----
    # (1 + S̃) * she_R_mean: S̃ = mean_layer(S_α(H@Hᵀ)) / (1+log T); she_R_mean = mean of K worst -(SE_t * eu_t)
    ("GLU",                    "GLU"),
    # (1 + S̃) * R_mean: same as GLU but local term is logtoku = -(au_dirichlet * eu) instead of Shannon-EU
    ("GLU_EDL",                "GLU_EDL"),

    # ---- Global ablations: vary S, local fixed to she_R_mean ----
    # (1 + S_alpha) * she_R_mean: S_alpha = mean_layer(S_α) with no length normalization
    ("GLU_rawS_mean",          "GLU_rawS_mean"),
    # (1 + max_layer S_α) * she_R_mean: takes the single highest-entropy layer's raw S_α
    ("GLU_rawS_best",          "GLU_rawS_best"),
    # (1 + max_layer(S_α)/(1+log T)) * she_R_mean: length-normalizes the best-layer S_α before scaling
    ("GLU_stilde_best",        "GLU_stilde_best"),

    # ---- Local ablations: vary U, global fixed to S̃ ----
    # (1 + S̃) * she_R_mean_sp: softplus(logits) for alpha instead of ReLU; eu = k/Σ(alpha+ε) not Σ(alpha+1)
    ("GLU_softplus",           "GLU_softplus"),
    # (1 + S̃) * she_R_mean_dk: k_eff = k/(1+S̃), so geometrically uncertain responses use fewer worst tokens
    ("GLU_dynK",               "GLU_dynK"),
    # (1 + S̃) * mean_K_worst(-SE_t): AU-only — drops the eu multiplier, keeps only Shannon entropy of top-k softmax
    ("GLU_AU",                 "GLU_AU"),
    # (1 + S̃) * mean_K_worst(-eu_t): EU-only — drops Shannon entropy, eu = k/Σ(ReLU(logit_j)+1)
    ("GLU_EU",                 "GLU_EU"),
    # (1 + S̃) * mean_K_worst(-eu_t_sp): EU-only with softplus alpha; eu = k/Σ(softplus(logit_j)+ε)
    ("GLU_EU_softplus",        "GLU_EU_softplus"),

    # ---- Combined ablations: vary both S and U ----
    # (1 + S_alpha) * mean_K_worst(-SE_t): raw Rényi (no log-T norm) × Shannon-entropy-only local
    ("GLU_salpha_AU",          "GLU_salpha_AU"),
    # (1 + S_alpha) * she_R_mean_sp: raw Rényi × softplus Shannon-EU (alpha=softplus, eu denom=Σα+ε)
    ("GLU_salpha_softplus",    "GLU_salpha_softplus"),
    # (1 + S_alpha) * mean_K_worst(-eu_t_sp): raw Rényi × softplus EU-only
    ("GLU_salpha_softplus_eu", "GLU_salpha_softplus_eu"),

    # ---- Additive fusion baselines ----
    # S_alpha + she_R_mean: additive instead of multiplicative; tests whether interaction term in GLU matters
    ("add_salpha_she",         "add_salpha_she"),
    # S̃ + she_R_mean: additive with length-normalized global term
    ("add_stilde_she",         "add_stilde_she"),

    # ---- External baselines (computed with --ptrue; NaN when skipped) ----
    # greedy generation of max_new_tokens=1; decoded token is "1" (correct) or "0" (incorrect)
    # ("ptrue",                  "P(true)"),
]

COL_W  = 18
header = f"{'dataset':35s}" + "".join(f"  {lbl:>{COL_W}s}" for _, lbl in ABLATIONS)
sep    = "-" * len(header)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

names      = []
auroc_rows = []
prr_rows   = []

for _, name, df in file_specs:
    names.append(name)

    s_alpha_best = df["S_alpha_per_layer"].apply(lambda x: max(_parse(x)))
    s_tilde_best = s_alpha_best / (1.0 + np.log(df["T"]))

    # df["RAUQ"]                   = -df["u_rauq"]
    df["logprob"]                = df.apply(compute_logprob, axis=1)
    df["GLU"]                    = (1.0 + df["S_tilde"]) * df["she_R_mean"]
    df["GLU_EDL"]                = (1.0 + df["S_tilde"]) * df["R_mean"]
    df["GLU_rawS_mean"]          = (1.0 + df["S_alpha"]) * df["she_R_mean"]
    df["GLU_rawS_best"]          = (1.0 + s_alpha_best)  * df["she_R_mean"]
    df["GLU_stilde_best"]        = (1.0 + s_tilde_best)  * df["she_R_mean"]
    df["she_R_mean_sp"]          = recompute_she_R_mean(df, use_softplus=True)
    df["GLU_softplus"]           = (1.0 + df["S_tilde"]) * df["she_R_mean_sp"]
    df["she_R_mean_dk"]          = recompute_she_R_mean(df, dynamic_k=True)
    df["GLU_dynK"]               = (1.0 + df["S_tilde"]) * df["she_R_mean_dk"]
    df["she_R_mean_au"]          = recompute_she_R_mean(df, au_only=True)
    df["GLU_AU"]                 = (1.0 + df["S_tilde"]) * df["she_R_mean_au"]
    df["eu_R_mean"]              = recompute_she_R_mean(df, eu_only=True)
    df["GLU_EU"]                 = (1.0 + df["S_tilde"]) * df["eu_R_mean"]
    df["GLU_EU_softplus"]        = (1.0 + df["S_tilde"]) * recompute_she_R_mean(
                                       df, use_softplus=True, eu_only=True)
    df["GLU_salpha_AU"]          = (1.0 + df["S_alpha"]) * df["she_R_mean_au"]
    df["GLU_salpha_softplus"]    = (1.0 + df["S_alpha"]) * recompute_she_R_mean(
                                       df, use_softplus=True)
    df["GLU_salpha_softplus_eu"] = (1.0 + df["S_alpha"]) * recompute_she_R_mean(
                                       df, use_softplus=True, eu_only=True)
    df["add_salpha_she"]         = df["S_alpha"] + df["she_R_mean"]
    df["add_stilde_she"]         = df["S_tilde"] + df["she_R_mean"]

    if args.ptrue:
        ptrue_scores = ptrue_cache.get(name)
        df["ptrue"]  = ptrue_scores if ptrue_scores is not None else float("nan")
    
    df.to_json(f"output/ablations/{name}-ablation-data.jsonl", orient="records", lines=True)

    auroc_vals, prr_vals = [], []
    for col, _ in ABLATIONS:
        sub = df.dropna(subset=["label", col])
        if len(sub) < 2:
            auroc_vals.append(float("nan"))
            prr_vals.append(float("nan"))
            continue
        auroc_vals.append(roc_auc_score(sub["label"], sub[col]))
        prr_vals.append(compute_prr(sub["label"].values, sub[col].values))
    auroc_rows.append(auroc_vals)
    prr_rows.append(prr_vals)


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_table(title, rows):
    print(f"\n{'=' * len(header)}")
    print(f" {title}")
    print(sep)
    print(header)
    print(sep)
    for name, vals in zip(names, rows):
        print(f"{name:35s}" + "".join(f"  {v:>{COL_W}.4f}" for v in vals))
    print(sep)
    means = np.nanmean(rows, axis=0)
    print(f"{'MEAN':35s}" + "".join(
        f"  {v:>{COL_W}.4f}" if not np.isnan(v) else f"  {'N/A':>{COL_W}s}"
        for v in means
    ))


print_table("AUROC", auroc_rows)
print_table("PRR",   prr_rows)
