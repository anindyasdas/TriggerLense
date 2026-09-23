import numpy as np
import torch
import torch.nn.functional as F
from collections import deque


def rolling_zscore(history: deque, new_val: float, min_fill: int = 3) -> float:
    if not np.isfinite(new_val):
        return 0.0
    clean_hist = [x for x in history if np.isfinite(x)]
    if len(clean_hist) < min_fill:
        return 0.0
    arr = np.array(clean_hist)
    mu  = arr.mean()
    sd  = arr.std()
    if sd < 1e-8:
        return 0.0
    z = (new_val - mu) / sd
    return float(z) if np.isfinite(z) else 0.0

def current_median_std(arr, w=70, min_fill=5, eps=1e-6):
    arr = np.array(arr, dtype=float)

    if arr.size == 0:
        return 0.0, eps

    win = arr[max(0, len(arr) - w):]

    if len(win) < min_fill:
        med = float(win[-1])
        std = eps
    else:
        med = float(np.median(win))
        std = float(max(np.std(win), eps))

    return med, std

def rolling_median_std(arr, w=70, min_fill=5):
    med = np.full(len(arr), np.nan, dtype=float)
    std = np.full(len(arr), np.nan, dtype=float)

    for i in range(len(arr)):
        win = arr[max(0, i-w+1):i+1]
        if len(win) < min_fill:
            continue
        med[i] = np.median(win)
        std[i] = max(np.std(win), 1e-6)

    return med, std

def rolling_sum_last(values, window: int) -> float:
    if len(values) == 0:
        return 0.0
    return float(np.sum(values[-window:]))

def safe_head_peak_std(attn: torch.Tensor) -> float:
    last_row = attn[:, -1, :].float()
    val = float(last_row.argmax(dim=-1).float().std().item()) 
    return val if np.isfinite(val) else 0.0

def hidden_semantic_distance(h_current: torch.Tensor, reference) -> float:

    if isinstance(reference, deque):
        if len(reference) == 0:
            return 0.0
        ref_vec = torch.stack(list(reference), dim=0).float().mean(dim=0)
    elif isinstance(reference, torch.Tensor):
        ref_vec = reference.float()
    else:
        raise TypeError(f"reference must be torch.Tensor or deque, got {type(reference)}")

    cos = F.cosine_similarity(
        h_current.unsqueeze(0).cpu(),
        ref_vec.unsqueeze(0),
        dim=-1
    ).item()
    dist = 1.0 - cos
    return float(dist) if np.isfinite(dist) else 0.0

@torch.no_grad()
def compute_prompt_hidden_mean(model, input_ids):
    out = model(
        input_ids=input_ids,
        output_hidden_states=True,
        return_dict=True,
    )
    last_hs = out.hidden_states[-1][0].float()
    last_hs = torch.nan_to_num(last_hs, nan=0.0)
    return last_hs.mean(dim=0).cpu()


def bscore_from_attn(out, prompt_start: int, prompt_end: int,
                     L: int = -1, bottom_pct: float = 0.01) -> float:
    num_layers = len(out.attentions)
    L_use = num_layers if L == -1 else min(L, num_layers)

    attn_layers = torch.stack(out.attentions[-L_use:], dim=0)
    attn_layers = attn_layers[:, 0]
    attn_q      = attn_layers[:, :, -1, :]
    attn_seg    = attn_q[:, :, prompt_start:prompt_end]

    L_u, H, P = attn_seg.shape
    if P < 2:
        return 0.0

    head_vecs = attn_seg.reshape(L_u * H, P).float()
    head_vecs = F.normalize(head_vecs, p=2, dim=1)
    sim_mat   = torch.matmul(head_vecs, head_vecs.T)

    N = sim_mat.shape[0]
    if N < 2:
        return 0.0

    idx   = torch.triu_indices(N, N, offset=1, device=sim_mat.device)
    sims  = sim_mat[idx[0], idx[1]].float()
    sims  = torch.nan_to_num(sims, nan=0.0)

    k     = max(1, int(bottom_pct * sims.numel()))
    bot   = torch.topk(sims, k=k, largest=False).values.mean()
    return (1.0 / torch.clamp(bot, min=1e-4)).item()


def squash(x):
    return np.log1p(max(0.0, x))