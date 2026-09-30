"""
Unified UQ pipeline — single greedy generation per sample, all three metrics, one judge call.

All metrics share identical responses and correctness labels, enabling direct apples-to-apples
comparison. The model is loaded once with attn_implementation="eager" (required by RAUQ).

Metrics:
  Shannon logit-entropy (s_combined):
      U_logit   : mean per-token top-k softmax entropy
      U_max     : max per-token top-k softmax entropy
      U_shannon : (1 - S_tilde) * U_logit + S_tilde * U_max

  EDL reliability (s_combined_edl):
      R_mean    : mean of k-worst logtoku   (negative)
      R_worst   : min logtoku               (negative)
      U_edl     : (1 - S_tilde) * R_mean + S_tilde * R_worst
      she_R_mean / she_R_worst / she_U : Shannon-EU variants

  RAUQ (baseline_attention_rauq):
      u_rauq    : recurrent attention-based scalar (higher → more uncertain)
      u_rauq_per_layer : {layer_idx: u_l}

  Geometry (shared across all three):
      S_alpha, S_tilde, S_alpha_per_layer, S_tilde_per_layer

Generation order:
  1. model.generate with output_hidden_states + output_logits
  2. Compute all non-attention metrics from the generate output
  3. del large generate tensors (hidden states)
  4. Single model forward pass on the fixed sequence with output_attentions for RAUQ

Outputs:
    ./output/full/{model}-{dataset}-unified-uq-full.jsonl
    ./output/full/{model}-{dataset}-unified-uq-full-summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from datasets import load_dataset
from dotenv import load_dotenv
from openai import AzureOpenAI
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

load_dotenv()


# =============================================================================
# PRIMITIVES
# =============================================================================

def _cfg(model, attr):
    """Get config attribute, falling back to text_config for nested configs (e.g. Gemma3)."""
    if hasattr(model.config, attr):
        return getattr(model.config, attr)
    tc = getattr(model.config, "text_config", None)
    if tc is not None and hasattr(tc, attr):
        return getattr(tc, attr)
    raise AttributeError(f"Model config has no attribute {attr!r}")


def matrix_renyi_entropy(K: torch.Tensor, alpha: float = 2.0):
    """
    Matrix-based Renyi entropy of order alpha.
    K = H @ H.T (unnormalized); eigenvalues normalized by trace.
    Returns (S, S_tilde) as Python floats in nats.
    """
    eigvals = torch.linalg.eigvalsh(K.float()).clamp_min(0.0)
    trace = eigvals.sum()
    if trace.item() < 1e-10:
        return 0.0, 0.0, []
    lam = eigvals / trace
    lam_pos = lam[lam > 1e-12]
    if lam_pos.numel() == 0:
        return 0.0, 0.0, []
    if abs(alpha - 1.0) < 1e-6:
        S = -(lam_pos * torch.log(lam_pos)).sum()
    else:
        S = (1.0 / (1.0 - alpha)) * torch.log(
            (lam_pos ** alpha).sum().clamp_min(1e-12)
        )
    S = float(S.item())
    S_tilde = S / (1.0 + math.log(K.shape[0]))
    lam_desc = lam.flip(0)                          # descending for spectrum plots
    return S, S_tilde, lam_desc.cpu().tolist()


def au_eu_batch(logits_batch: torch.Tensor, k: int = 10):
    """
    Vectorized EDL aleatoric/epistemic uncertainty for every generated token.
    logits_batch : (T, vocab)
    Returns (logtoku, shetoku), each (T,) negative tensors.
    """
    topk_values, _ = torch.topk(logits_batch, k, dim=-1)   # (T, k)
    alpha_k = F.relu(topk_values)
    alpha_0 = alpha_k.sum(dim=-1, keepdim=True)
    zero_mask = alpha_0.squeeze(-1) < 1e-10
    alpha_0_safe = alpha_0.clamp_min(1e-10)

    p_k = alpha_k / alpha_0_safe
    au = -(p_k * (torch.digamma(alpha_k + 1) - torch.digamma(alpha_0_safe + 1))).sum(dim=-1)
    eu = k / (alpha_k + 1).sum(dim=-1)
    au = torch.where(zero_mask, torch.full_like(au, math.log(k)), au)
    eu = torch.where(zero_mask, torch.ones_like(eu), eu)
    logtoku = -(au * eu)

    probs = F.softmax(topk_values, dim=-1)
    se = -(probs * torch.log2(probs.clamp_min(1e-12))).sum(dim=-1)
    shetoku = -(se * eu)
    return logtoku, shetoku


def twoNN(H: torch.Tensor) -> float:
    """TwoNN intrinsic dimensionality estimate (Facco et al. 2017)."""
    T = H.shape[0]
    if T < 4:
        return float("nan")
    H = H.float()
    dists = torch.cdist(H, H)
    dists.fill_diagonal_(float("inf"))
    knn, _ = torch.topk(dists, 2, dim=-1, largest=False)
    r1, r2 = knn[:, 0], knn[:, 1]
    mask = (r1 > 1e-10) & (r2 > r1)
    if int(mask.sum().item()) < 2:
        return float("nan")
    log_mu = torch.log(r2[mask] / r1[mask])
    s = float(log_mu.sum().item())
    if s < 1e-10:
        return float("nan")
    return float(int(mask.sum().item()) / s)


@torch.no_grad()
def compute_rauq(model, full_ids, P, token_probs, rauq_alpha=0.2,
                 layer_subset=None, use_hooks=False):
    """
    RAUQ forward pass for attention weights (Algorithm 1, Eqs. 1-4).
    token_probs : (N,) CPU tensor of P(y_i | context), pre-computed from generate logits.
    Returns (u_y scalar, u_per_layer dict with str keys).
    """
    device = model.device
    L = _cfg(model, "num_hidden_layers")
    N = full_ids.shape[1] - P
    assert N >= 2, f"Need at least 2 response tokens for RAUQ (got {N})."

    if use_hooks:
        sub_diag_list = [None] * L
        handles = []

        def make_hook(idx):
            def hook(module, inputs, output):
                items = output if isinstance(output, tuple) else (output,)
                for item in items:
                    if torch.is_tensor(item) and item.dim() == 4:
                        sub_diag_list[idx] = (
                            item[0].float().diagonal(offset=-1, dim1=-2, dim2=-1).cpu()
                        )
                        break
            return hook

        for l, layer in enumerate(model.model.layers):
            handles.append(layer.self_attn.register_forward_hook(make_hook(l)))
        try:
            fwd = model(full_ids, output_attentions=True, return_dict=True)
        finally:
            for h in handles:
                h.remove()
        if not all(s is not None for s in sub_diag_list):
            raise RuntimeError("Hooks failed to capture attentions. Try --no-rauq-hooks.")
        sub_diags = torch.stack(sub_diag_list)   # (L, H, S-1)
    else:
        fwd = model(full_ids, output_attentions=True, return_dict=True)
        sub_diags = torch.stack([
            fwd.attentions[l][0].float().diagonal(offset=-1, dim1=-2, dim2=-1).cpu()
            for l in range(L)
        ])                                        # (L, H, S-1)

    del fwd

    expected_H = _cfg(model, "num_attention_heads")
    assert sub_diags.shape[1] == expected_H, (
        f"Got {sub_diags.shape[1]} heads, expected {expected_H}."
    )

    # Slice to response-only positions
    sub_diags = sub_diags[:, :, P:P + N - 1]     # (L, H, N-1)
    h_per_layer = sub_diags.mean(dim=-1).argmax(dim=-1)  # (L,) best head per layer

    if layer_subset is None:
        layer_subset = list(range(L // 3, 2 * L // 3 + 1))

    u_per_layer = {}
    for l in layer_subset:
        h_l = h_per_layer[l].item()
        a_sel = sub_diags[l, h_l]               # (N-1,)
        c = torch.empty(N)
        c[0] = token_probs[0]
        for i in range(1, N):
            c[i] = rauq_alpha * token_probs[i] + (1 - rauq_alpha) * a_sel[i - 1] * c[i - 1]
        u_per_layer[l] = (-c.clamp_min(1e-12).log()).mean().item()

    u_y = float(max(u_per_layer.values()))
    return u_y, {str(k): v for k, v in u_per_layer.items()}


# =============================================================================
# UNIFIED RESULT DATACLASS
# =============================================================================

@dataclass
class AllMetrics:
    # Shannon logit-entropy
    U_logit: float
    U_max: float
    U_shannon: float
    # EDL reliability (negative scores; more negative = more uncertain)
    R_mean: float
    R_worst: float
    U_edl: float
    she_R_mean: float
    she_R_worst: float
    she_U: float
    # RAUQ (positive; higher = more uncertain)
    u_rauq: float
    u_rauq_per_layer: dict
    # Geometry (shared)
    S_alpha: float
    S_tilde: float
    S_alpha_per_layer: list
    S_tilde_per_layer: list
    eigenvalues_per_layer: list
    # Meta
    T: int
    num_layers: int
    num_prompt_tokens: int
    layer_idx: int
    response_text: str
    # Per-token breakdown: list of dicts with token, token_id, au, eu,
    # collision_entropy, logtoku, coletoku
    token_data: list
    # Top-100 raw logits, probs, and token IDs per token: (T, 100) as nested lists
    top100_logits: list
    top100_token_ids: list
    top100_probs: list
    # TwoNN intrinsic dimensionality per layer
    id_per_layer: list


@torch.no_grad()
def compute_all_metrics(
    model,
    tokenizer,
    prompt_messages,
    layer_idx: int = -1,
    k: int = 10,
    alpha: float = 2.0,
    max_new_tokens: int = 256,
    seed: Optional[int] = None,
    rauq_alpha: float = 0.2,
    rauq_layer_subset=None,
    rauq_use_hooks: bool = False,
) -> AllMetrics:
    """Single generation + all three UQ metrics + RAUQ forward pass."""
    device = next(model.parameters()).device

    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    inputs = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(device)
    P = inputs["input_ids"].shape[-1]

    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        return_dict_in_generate=True,
        output_hidden_states=True,
        output_logits=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    T = len(out.hidden_states)
    L = len(out.hidden_states[0])
    if T < 2:
        raise ValueError(f"Generated only {T} tokens; need >= 2 for matrix entropy.")

    # --- Geometry: matrix Renyi entropy + TwoNN intrinsic dimensionality ---
    S_alpha_per_layer, S_tilde_per_layer, eigenvalues_per_layer, id_per_layer = [], [], [], []
    for layer in range(L):
        H = torch.stack(
            [out.hidden_states[t][layer][0, -1, :] for t in range(T)], dim=0
        )
        S, St, eigs = matrix_renyi_entropy(H @ H.T, alpha=alpha)
        S_alpha_per_layer.append(S)
        S_tilde_per_layer.append(St)
        eigenvalues_per_layer.append(eigs)
        id_per_layer.append(twoNN(H))

    chosen = L + layer_idx if layer_idx < 0 else layer_idx
    if not (0 <= chosen < L):
        raise ValueError(f"layer_idx={layer_idx} out of range for L={L}.")
    S_alpha = float(np.mean(S_alpha_per_layer))
    S_tilde = float(np.mean(S_tilde_per_layer))

    # --- Logit tensor (shared input to Shannon and EDL) ---
    logits = torch.stack(out.logits, dim=0).squeeze(1).float()  # (T, vocab)

    # --- Shannon logit-entropy ---
    topk_logits, _ = torch.topk(logits, k=k, dim=-1)
    topk_probs = F.softmax(topk_logits, dim=-1)
    h_t = -(topk_probs * torch.log(topk_probs.clamp_min(1e-12))).sum(dim=-1)
    U_logit  = float(h_t.mean().item())
    U_max    = float(h_t.max().item())
    U_shannon = (1.0 - S_tilde) * U_logit + S_tilde * U_max

    # --- EDL reliability ---
    logtoku, shetoku = au_eu_batch(logits, k=k)
    k_w = min(k, T)

    logtoku_s, _ = torch.sort(logtoku)
    R_mean  = float(logtoku_s[:k_w].mean().item())
    R_worst = float(logtoku.min().item())
    U_edl   = (1.0 - S_tilde) * R_mean + S_tilde * R_worst

    shetoku_s, _ = torch.sort(shetoku)
    she_R_mean  = float(shetoku_s[:k_w].mean().item())
    she_R_worst = float(shetoku.min().item())
    she_U       = (1.0 - S_tilde) * she_R_mean + S_tilde * she_R_worst

    # --- Token probs for RAUQ (reuse generate logits, skip a second softmax in fwd pass) ---
    full_ids = out.sequences                            # kept alive after del out
    resp_ids = full_ids[0, P:P + T]
    token_log_probs = (
        torch.log_softmax(logits, dim=-1)
        .gather(-1, resp_ids[:, None])
        .squeeze(-1)
    )
    token_probs = token_log_probs.exp().cpu()           # (T,)

    response_text = tokenizer.decode(out.sequences[0, P:], skip_special_tokens=True)

    # --- Per-token breakdown (computed before freeing logits) ---
    top100_n = min(100, logits.shape[-1])
    top100_vals, top100_ids = torch.topk(logits, top100_n, dim=-1)  # (T, 100)

    _topk, _ = torch.topk(logits, k=k, dim=-1)
    _alpha_k = F.relu(_topk)
    _alpha_0 = _alpha_k.sum(dim=-1, keepdim=True)
    _zero = _alpha_0.squeeze(-1) < 1e-10
    _a0s = _alpha_0.clamp_min(1e-10)
    _pk = _alpha_k / _a0s
    au_vec = -(_pk * (torch.digamma(_alpha_k + 1) - torch.digamma(_a0s + 1))).sum(dim=-1)
    eu_vec = k / (_alpha_k + 1).sum(dim=-1)
    au_vec = torch.where(_zero, torch.full_like(au_vec, math.log(k)), au_vec)
    eu_vec = torch.where(_zero, torch.ones_like(eu_vec), eu_vec)

    # Collision entropy over full vocab: -log2(sum(p_i^2))
    _log_p = F.log_softmax(logits, dim=-1)
    _log_s2 = torch.logsumexp(2.0 * _log_p, dim=-1)
    ce_vec = -(_log_s2 / math.log(2))  # (T,) in bits

    top100_probs     = F.softmax(top100_vals, dim=-1).tolist()  # (T, 100)
    top100_logits    = top100_vals.tolist()                     # (T, 100)
    top100_token_ids = top100_ids.tolist()                      # (T, 100)

    token_data = []
    for t in range(T):
        tid  = resp_ids[t].item()
        au_t = au_vec[t].item()
        eu_t = eu_vec[t].item()
        ce_t = ce_vec[t].item()
        token_data.append({
            "token":             tokenizer.decode([tid], skip_special_tokens=True),
            "token_id":          tid,
            "au":                au_t,
            "eu":                eu_t,
            "collision_entropy": ce_t,
            "logtoku":           -(au_t * eu_t),
            "coletoku":          -(ce_t * eu_t),
        })

    del out, logits                                     # free hidden states + stacked logits

    # --- RAUQ: forward pass for attention weights ---
    u_rauq, u_rauq_per_layer = compute_rauq(
        model, full_ids, P, token_probs,
        rauq_alpha=rauq_alpha,
        layer_subset=rauq_layer_subset,
        use_hooks=rauq_use_hooks,
    )

    return AllMetrics(
        U_logit=U_logit, U_max=U_max, U_shannon=U_shannon,
        R_mean=R_mean, R_worst=R_worst, U_edl=U_edl,
        she_R_mean=she_R_mean, she_R_worst=she_R_worst, she_U=she_U,
        u_rauq=u_rauq, u_rauq_per_layer=u_rauq_per_layer,
        S_alpha=S_alpha, S_tilde=S_tilde,
        S_alpha_per_layer=S_alpha_per_layer,
        S_tilde_per_layer=S_tilde_per_layer,
        eigenvalues_per_layer=eigenvalues_per_layer,
        T=T, num_layers=L, num_prompt_tokens=int(P),
        layer_idx=chosen, response_text=response_text,
        token_data=token_data,
        top100_logits=top100_logits,
        top100_token_ids=top100_token_ids,
        top100_probs=top100_probs,
        id_per_layer=id_per_layer,
    )


# =============================================================================
# AZURE JUDGE
# =============================================================================

LABELER_SYSTEM_PROMPT = (
    "You are an expert factual evaluator. Your task is to determine whether a "
    "given **Response** to a **Question** is factually correct based on the "
    "provided ground-truth **Answer**.\n\n"
    "A **Response** is considered correct (label = 1) if it clearly and "
    "unambiguously contains or conveys the same factual information as the "
    "**Answer**, even if the wording differs.\n\n"
    "A **Response** is considered incorrect (label = 0) if:\n"
    "* It contradicts the Answer\n"
    "* It omits the key required fact\n"
    "* It is ambiguous or unclear with respect to the correct fact\n\n"
    "If the **Answer** contains multiple possible correct items (e.g., a "
    "list), the **Response** is considered correct if it matches at least "
    "one of them.\n\n"
    "Output a single binary value:\n"
    "* 1 → correct\n"
    "* 0 → incorrect\n\n"
    "Do not output anything other than 0 or 1."
)


def setup_azure_client():
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
    deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
    subscription_key = os.getenv("AZURE_OPENAI_API_KEY")
    api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2024-12-01-preview")

    missing = [n for n, v in [
        ("AZURE_OPENAI_ENDPOINT", endpoint),
        ("AZURE_OPENAI_API_KEY", subscription_key),
        ("AZURE_OPENAI_DEPLOYMENT", deployment),
    ] if not v]
    if missing:
        sys.exit(f"Missing env vars: {', '.join(missing)}. Put them in .env.")

    return AzureOpenAI(
        api_version=api_version, azure_endpoint=endpoint, api_key=subscription_key,
    ), deployment



def correctness_labeler(client, deployment, question, ground_truth, response):
    user_msg = f"Question: {question}\nAnswer: {ground_truth}\nResponse: {response}"
    try:
        out = client.chat.completions.create(
            messages=[
                {"role": "system", "content": LABELER_SYSTEM_PROMPT},
                {"role": "user",   "content": user_msg},
            ],
            max_completion_tokens=8,
            temperature=0.0,
            model=deployment,
        )
        label_raw = out.choices[0].message.content.strip()
        if label_raw not in ("0", "1"):
            print(f"  [unexpected label] {repr(label_raw)}")  # catch edge cases
        return int(label_raw) if label_raw in ("0", "1") else None
    except Exception as e:
        print(f"  [labeler error] {e}", flush=True)
        return None


# =============================================================================
# PROJECT CONFIG
# =============================================================================

model_id_name = {
    "fanar": "QCRI/Fanar-1-9B-Instruct",
    "gemma": "google/gemma-3-12b-it",
    "qwen":  "Qwen/Qwen2.5-7B-Instruct",
}

available_datasets = {
    "triviaqa":        "./small_datasets/triviaqa_small.csv",
    "triviaqa_full":   "./small_datasets/triviaqa_validation.csv",
    "truthfulqa":      "./small_datasets/truthfulqa.csv",
    "math":            "./small_datasets/math_small.csv",
    "arabica":       "./small_datasets/arabica_qa.csv",
    "longform":      "./small_datasets/longform_subset.csv",
}

GROUND_TRUTH_COL = {
    "triviaqa":        "answer",
    "triviaqa_full":   "answer",
    "truthfulqa":      "correct_answers",
    "math":            "answer",
    "arabica":         "answer",
    "longform":        "answer",
}


def system_prompt_for(dataset_name: str) -> str:
    if dataset_name == "math":
        return (
            "Answer the question concisely without non-necessary words, "
            "think step by step but only write necessary computation steps. "
            "At the end, give the final answer number after 'Final Answer:'"
        )
    if dataset_name == "longform":
        return (
            "You are a helpful assistant."
        )
    if dataset_name == "arabica":
        return (
            "Respond in the same language as the question."
            "Answer the question concisely without non-necessary words."
        )
    return "Answer the question concisely."


def safe_auroc(y_wrong, scores):
    y = np.asarray(y_wrong)
    s = np.asarray(scores, dtype=float)
    if y.sum() == 0 or y.sum() == len(y):
        return None
    return float(roc_auc_score(y, s))


# =============================================================================
# MAIN
# =============================================================================

parser = argparse.ArgumentParser(
    description="Unified UQ pipeline: Shannon + EDL + RAUQ on identical responses/labels."
)
parser.add_argument("--model",    choices=model_id_name.keys(), required=True)
parser.add_argument("--datasets", choices=available_datasets.keys(), nargs="+", required=True)
parser.add_argument("--test",     action="store_true", help="5 samples, verbose.")
parser.add_argument("--alpha",    type=float, default=2.0,  help="Renyi order.")
parser.add_argument("--layer-idx",type=int,   default=-1,   help="Layer for U combination.")
parser.add_argument("--k",        type=int,   default=10,   help="Top-k for logit entropy / EDL.")
parser.add_argument("--max-new-tokens", type=int, default=256)
parser.add_argument("--seed",     type=int,   default=0)
parser.add_argument("--rauq-alpha",     type=float, default=0.2, help="RAUQ recurrence weight.")
parser.add_argument("--rauq-use-hooks", action="store_true",
                    help="Memory-efficient RAUQ via forward hooks (saves ~1-2 GB but may fail "
                         "on some architectures).")
parser.add_argument("--max-samples",   type=int, default=None,
                    help="Cap number of samples (overridden by --test).")
parser.add_argument("--output-suffix", type=str, default="",
                    help="Extra suffix appended before .csv/.json (e.g. '-v2').")
args = parser.parse_args()

model_id = model_id_name[args.model]

print("=" * 60)
print("UNIFIED UQ PIPELINE  (Shannon + EDL + RAUQ)")
print("=" * 60)
print(f"Model:           {model_id}")
print(f"Datasets:        {args.datasets}")
print(f"Renyi alpha:     {args.alpha}")
print(f"layer_idx:       {args.layer_idx}")
print(f"k (top-k):       {args.k}")
print(f"max_new_tokens:  {args.max_new_tokens}")
print(f"seed:            {args.seed}")
print(f"rauq_alpha:      {args.rauq_alpha}")
print(f"rauq_use_hooks:  {args.rauq_use_hooks}")
print(f"max_samples:     {args.max_samples}")
print(f"output_suffix:   {args.output_suffix!r}")
print(f"Test mode:       {args.test}")
print("=" * 60)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"\nLoading model on {device} (attn_implementation=eager)...", flush=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    device_map=device,
    attn_implementation="eager",
)
model.eval()
model.generation_config.disable_compile = True
model._supports_static_cache = False

client, deployment = setup_azure_client()
os.makedirs("./output/full", exist_ok=True)

for dataset_name in args.datasets:
    print(f"\n{'=' * 60}")
    print(f"DATASET: {dataset_name}")
    print(f"{'=' * 60}")

    data_path = available_datasets[dataset_name]
    dataset = load_dataset("csv", data_files=data_path, split="train")
    gt_col = GROUND_TRUTH_COL[dataset_name]
    sys_prompt = system_prompt_for(dataset_name)

    if args.test:
        n_samples = min(5, len(dataset))
    elif args.max_samples:
        n_samples = min(args.max_samples, len(dataset))
    else:
        n_samples = len(dataset)
    verbose = args.test

    suffix = args.output_suffix
    out_jsonl   = f"./output/full/{args.model}-{dataset_name}-unified-uq-full{suffix}.jsonl"
    out_summary = f"./output/full/{args.model}-{dataset_name}-unified-uq-full{suffix}-summary.json"

    rows_buffer = []

    print(f"Processing {n_samples} samples → {out_jsonl}", flush=True)
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for i in tqdm(range(n_samples), disable=verbose):
            question     = dataset[i]["question"]
            ground_truth = dataset[i][gt_col]
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user",   "content": question},
            ]

            if verbose:
                print(f"\n{'-' * 50}")
                print(f"[{i + 1}/{n_samples}] Q: {question[:120]}")

            try:
                res = compute_all_metrics(
                    model, tokenizer, messages,
                    layer_idx=args.layer_idx,
                    k=args.k,
                    alpha=args.alpha,
                    max_new_tokens=args.max_new_tokens,
                    seed=args.seed,
                    rauq_alpha=args.rauq_alpha,
                    rauq_use_hooks=args.rauq_use_hooks,
                )
            except ValueError as e:
                print(f"  [skipped] {e}", flush=True)
                continue
            except Exception as e:
                print(f"  [error] {repr(e)}", flush=True)
                continue

            label = correctness_labeler(
                client, deployment, question, ground_truth, res.response_text,
            )

            if verbose:
                print(f"  Response:  {res.response_text[:120]}")
                print(f"  GT:        {str(ground_truth)[:80]}")
                print(f"  T={res.T}  layers={res.num_layers}  layer={res.layer_idx}")
                print(f"  U_shannon={res.U_shannon:.4f}  U_logit={res.U_logit:.4f}  "
                      f"U_max={res.U_max:.4f}")
                print(f"  U_edl={res.U_edl:.4f}  R_mean={res.R_mean:.4f}  "
                      f"she_U={res.she_U:.4f}")
                print(f"  u_rauq={res.u_rauq:.4f}  "
                      f"S_alpha={res.S_alpha:.4f}  S_tilde={res.S_tilde:.4f}  "
                      f"label={label}")

            row = {
                "id": i,
                "prompt": question,
                "response": res.response_text,
                "ground_truth": ground_truth,
                "label": label,
                "T": res.T,
                "num_prompt_tokens": res.num_prompt_tokens,
                "num_layers": res.num_layers,
                "layer_idx_chosen": res.layer_idx,
                "U_logit": res.U_logit,
                "U_max": res.U_max,
                "U_shannon": res.U_shannon,
                "R_mean": res.R_mean,
                "R_worst": res.R_worst,
                "U_edl": res.U_edl,
                "she_R_mean": res.she_R_mean,
                "she_R_worst": res.she_R_worst,
                "she_U": res.she_U,
                "u_rauq": res.u_rauq,
                "u_rauq_per_layer": res.u_rauq_per_layer,
                "S_alpha": res.S_alpha,
                "S_tilde": res.S_tilde,
                "S_alpha_per_layer":     res.S_alpha_per_layer,
                "S_tilde_per_layer":     res.S_tilde_per_layer,
                "eigenvalues_per_layer": res.eigenvalues_per_layer,
                "id_per_layer":          res.id_per_layer,
                "token_data":            res.token_data,
                "top100_logits":         res.top100_logits,
                "top100_token_ids":      res.top100_token_ids,
                "top100_probs":          res.top100_probs,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()

            if label is not None:
                rows_buffer.append({
                    "label":        label,
                    "U_logit":      res.U_logit,
                    "U_max":        res.U_max,
                    "U_shannon":    res.U_shannon,
                    "R_mean":       res.R_mean,
                    "R_worst":      res.R_worst,
                    "U_edl":        res.U_edl,
                    "she_R_mean":   res.she_R_mean,
                    "she_R_worst":  res.she_R_worst,
                    "she_U":        res.she_U,
                    "u_rauq":       res.u_rauq,
                    "S_alpha":      res.S_alpha,
                    "S_tilde":      res.S_tilde,
                    "S_alpha_per_layer": res.S_alpha_per_layer,
                })

    # --- AUROC summary ---
    print(f"\n--- AUROC ({dataset_name}) ---")
    if not rows_buffer:
        print("No labeled rows; skipping AUROC.")
        continue

    labels  = np.array([r["label"] for r in rows_buffer])
    y_wrong = 1 - labels
    n_correct = int(labels.sum())
    n_wrong   = int(y_wrong.sum())
    print(f"n={len(labels)}  correct={n_correct}  wrong={n_wrong}  "
          f"acc={n_correct / len(labels):.3f}")

    # EDL and Shannon-EU scores are negative (more negative = more uncertain),
    # so negate before passing to AUROC. Shannon logit-entropy and RAUQ are
    # positive (higher = more uncertain) and need no sign flip.
    aurocs = {
        "U_shannon":  safe_auroc(y_wrong, [r["U_shannon"]    for r in rows_buffer]),
        "U_logit":    safe_auroc(y_wrong, [r["U_logit"]      for r in rows_buffer]),
        "U_max":      safe_auroc(y_wrong, [r["U_max"]        for r in rows_buffer]),
        "U_edl":      safe_auroc(y_wrong, [-r["U_edl"]       for r in rows_buffer]),
        "R_mean":     safe_auroc(y_wrong, [-r["R_mean"]      for r in rows_buffer]),
        "she_U":      safe_auroc(y_wrong, [-r["she_U"]       for r in rows_buffer]),
        "she_R_mean": safe_auroc(y_wrong, [-r["she_R_mean"]  for r in rows_buffer]),
        "u_rauq":     safe_auroc(y_wrong, [r["u_rauq"]       for r in rows_buffer]),
        "S_alpha":    safe_auroc(y_wrong, [r["S_alpha"]      for r in rows_buffer]),
        "S_tilde":    safe_auroc(y_wrong, [r["S_tilde"]      for r in rows_buffer]),
    }

    sections = [
        ("Shannon logit-entropy", ("U_shannon", "U_logit", "U_max")),
        ("EDL reliability",       ("U_edl", "R_mean")),
        ("Shannon-EU",            ("she_U", "she_R_mean")),
        ("RAUQ",                  ("u_rauq",)),
        ("Geometry",              ("S_alpha", "S_tilde")),
    ]
    for title, keys in sections:
        print(f"  {title}:")
        for name in keys:
            val = aurocs[name]
            print(f"    AUROC[{name:<12}] = {val if val is None else f'{val:.4f}'}")

    # Per-layer S_alpha AUROC
    n_layers = len(rows_buffer[0]["S_alpha_per_layer"])
    per_layer_auroc = [
        safe_auroc(y_wrong, [r["S_alpha_per_layer"][l] for r in rows_buffer])
        for l in range(n_layers)
    ]
    best_layer = max(
        (l for l in range(n_layers) if per_layer_auroc[l] is not None),
        key=lambda l: per_layer_auroc[l],
        default=None,
    )
    print(f"\n  Per-layer AUROC[S_alpha] (layer 0 = embeddings):")
    for layer, val in enumerate(per_layer_auroc):
        s    = "  N/A" if val is None else f"{val:.4f}"
        flag = "   <-- best" if best_layer is not None and layer == best_layer else ""
        print(f"    layer {layer:>2}: {s}{flag}")

    summary = {
        "dataset": dataset_name,
        "model": model_id,
        "config": {
            "alpha": args.alpha, "k": args.k,
            "layer_idx": args.layer_idx, "max_new_tokens": args.max_new_tokens,
            "seed": args.seed, "rauq_alpha": args.rauq_alpha,
            "rauq_use_hooks": args.rauq_use_hooks,
        },
        "n": int(len(labels)),
        "n_correct": n_correct,
        "n_wrong": n_wrong,
        "accuracy": float(n_correct / len(labels)),
        "aurocs": aurocs,
        "per_layer_auroc_S_alpha": per_layer_auroc,
        "best_layer_idx": best_layer,
    }
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {out_summary}", flush=True)

print(f"\n{'=' * 60}\nDONE\n{'=' * 60}")
