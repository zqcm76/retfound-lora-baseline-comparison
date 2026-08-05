"""
cdr.py — Cup-to-Disc Ratio computation.

Two regimes, matching the fork called out in the stage-3 plan:

  (A) hard / non-differentiable  -> compute_cdr_hard()
      Take argmax masks, measure geometry with numpy/cv2.
      This is the honest two-stage baseline. No gradient flows to segmentation.

  (B) soft / differentiable      -> SoftCDR (nn.Module)
      Compute geometry directly from cup/disc probability maps with
      differentiable surrogates, so classification loss back-props into
      the segmentation branch.

Conventions
-----------
Segmentation produces a 3-class map over {background=0, disc=1, cup=2}, OR
two binary maps. We standardize on the "ordinal" reading where the disc
region INCLUDES the cup (cup ⊂ disc anatomically). Whichever head layout
stage 2 used, convert to:
    p_disc[H,W] = P(pixel ∈ disc ∪ cup)   # the full optic disc
    p_cup [H,W] = P(pixel ∈ cup)          # the cup, a subset of disc
Both in [0,1]. See `logits_to_disc_cup_probs` for the conversion from the
3-class softmax that stage 2's segmentation head emits.

We report three CDRs, because clinicians use more than one and the plan
asks for all:
    vCDR  vertical   = cup_vertical_extent   / disc_vertical_extent
    hCDR  horizontal = cup_horizontal_extent / disc_horizontal_extent
    aCDR  area        = cup_area              / disc_area
"""

from __future__ import annotations

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# --------------------------------------------------------------------------- #
# Shared: turn a 3-class segmentation logit map into (p_disc, p_cup)
# --------------------------------------------------------------------------- #
def logits_to_disc_cup_probs(seg_logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    seg_logits: [B, 3, H, W] raw logits for {bg, disc-rim, cup}.

    The network's three channels are background, rim (disc-but-not-cup), and
    cup. The FULL disc probability is therefore P(rim) + P(cup), because the
    disc is the union of rim and cup. The cup probability is just P(cup).

    Returns (p_disc, p_cup), each [B, H, W] in [0, 1].
    """
    if seg_logits.dim() != 4 or seg_logits.size(1) != 3:
        raise ValueError(f"expected [B,3,H,W] seg logits, got {tuple(seg_logits.shape)}")
    probs = F.softmax(seg_logits, dim=1)           # [B,3,H,W]
    p_bg, p_rim, p_cup = probs[:, 0], probs[:, 1], probs[:, 2]
    p_disc = p_rim + p_cup                          # union; already in [0,1] since softmax sums to 1
    return p_disc, p_cup


# --------------------------------------------------------------------------- #
# (A) HARD / non-differentiable
# --------------------------------------------------------------------------- #
def _largest_component(mask: np.ndarray) -> np.ndarray:
    """Keep only the largest connected component; kills speckle that would
    otherwise blow up the bounding-box-based vertical/horizontal CDR."""
    mask = mask.astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        return mask  # nothing but background
    # stats[0] is background; pick largest of the rest by area
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == largest).astype(np.uint8)


def _vh_extent(mask: np.ndarray) -> tuple[int, int]:
    """Vertical and horizontal extent (in px) of a binary mask's largest blob.
    Returns (0, 0) for an empty mask so the caller can guard against div-by-0."""
    ys, xs = np.where(mask > 0)
    if ys.size == 0:
        return 0, 0
    v = int(ys.max() - ys.min() + 1)
    h = int(xs.max() - xs.min() + 1)
    return v, h


def compute_cdr_hard(
    seg_logits: torch.Tensor,
    eps: float = 1e-6,
    clamp01: bool = True,
) -> torch.Tensor:
    """
    Non-differentiable CDR from argmax masks.

    seg_logits: [B,3,H,W]. Detached internally — by construction no gradient
    leaves this function, which is exactly what the two-stage path wants.

    Returns: [B, 3] tensor of (vCDR, hCDR, aCDR), on the same device/dtype as
    the input, but with requires_grad=False.

    A degenerate disc (no disc pixels predicted) yields CDR=0 for that sample,
    and the boolean mask of such failures is stashed on the returned tensor as
    `.degenerate` so the training loop can log how often it happens — silent
    zeros are how you end up with a great-looking AUC and a broken model.
    """
    seg = seg_logits.detach()
    probs = F.softmax(seg, dim=1).cpu().numpy()       # [B,3,H,W]
    B = probs.shape[0]
    out = np.zeros((B, 3), dtype=np.float32)
    degenerate = np.zeros((B,), dtype=bool)

    cls = probs.argmax(axis=1)                         # [B,H,W] in {0,1,2}
    for b in range(B):
        disc_mask = _largest_component((cls[b] >= 1).astype(np.uint8))  # rim ∪ cup
        cup_mask = _largest_component((cls[b] == 2).astype(np.uint8))

        disc_area = float(disc_mask.sum())
        cup_area = float(cup_mask.sum())
        if disc_area < 1.0:
            degenerate[b] = True
            continue  # leave row at 0

        dv, dh = _vh_extent(disc_mask)
        cv_, ch = _vh_extent(cup_mask)

        vcdr = cv_ / (dv + eps)
        hcdr = ch / (dh + eps)
        acdr = cup_area / (disc_area + eps)
        out[b] = (vcdr, hcdr, acdr)

    t = torch.from_numpy(out).to(seg_logits.device)
    if clamp01:
        t = t.clamp(0.0, 1.0)
    t.degenerate = torch.from_numpy(degenerate).to(seg_logits.device)  # attach metadata
    return t


# --------------------------------------------------------------------------- #
# (B) SOFT / differentiable
# --------------------------------------------------------------------------- #
class SoftCDR(nn.Module):
    """
    Differentiable CDR from probability maps.

    The trick for vertical/horizontal CDR: a hard "extent" (max_row - min_row)
    has zero gradient almost everywhere. We replace it with a soft extent.

    Per row r, define the disc's row-occupancy as the max probability across
    that row, softened:
        row_occ_disc[r] = softmax_over_columns_weighted ... 
    In practice we use a temperature-controlled soft-OR (log-sum-exp) across
    columns to estimate "is there disc in this row", then take a soft count of
    occupied rows as the vertical extent. Same for columns -> horizontal.

    Area CDR is naturally differentiable: it's just sum(p_cup)/sum(p_disc).

    All three share the property that as temperature -> 0 and probs -> {0,1},
    they converge to the hard definitions, so the soft and hard heads are
    measuring the same thing — important for the ablation to be apples-to-apples.

    Parameters
    ----------
    tau_col : temperature for the soft-OR across a row/column (smaller = sharper)
    tau_occ : temperature for converting soft-occupancy in [0,1] into a
              soft indicator before summing into an extent

    ⚠️ FIX(T-5) 必读 —— 梯度在分割自信时会消失，这不是 bug 而是本方法的性质：
        row_ind = sigmoid((row_occ - 0.5) / tau_occ)
        d(row_ind)/d(row_occ) = (1/tau_occ) * s * (1-s)

        row_occ = 0.50  ->  5.0e+00   (1.0x)
        row_occ = 0.90  ->  6.7e-03   (0.0013x)
        row_occ = 1.00  ->  9.1e-04   (0.00018x)

        一个**已训练好的** stage-2 分割器 row_occ 几乎全在 ~0 或 ~1，
        梯度衰减到决策边界处的 1/5507；再乘 soft_cdr_seg_grad_scale=0.1，
        总衰减约 5.5 万倍。

        结论：在真实 checkpoint 上 cdr_soft 与 cdr_two_stage **在机制上几乎等价**。
        观测到的 0.934 vs 0.931 不只是统计不显著，而是理论上就该是零。
        想让 soft-CDR 真正起作用，要么调大 tau_occ（如 0.2），要么在分割尚未
        饱和的更早阶段引入 —— 两者都值得作为一次消融。
    """

    def __init__(self, tau_col: float = 0.05, tau_occ: float = 0.05, eps: float = 1e-6):
        super().__init__()
        self.tau_col = float(tau_col)
        self.tau_occ = float(tau_occ)
        self.eps = float(eps)

    @staticmethod
    def _soft_or(p: torch.Tensor, dim: int, tau: float) -> torch.Tensor:
        """Differentiable soft-OR over `dim`: ~max(p) but smooth.
        Uses log-sum-exp on p/tau. Returns value in roughly [0,1] when inputs
        are in [0,1] and tau is small."""
        # lse(x) = tau * log( sum exp(x/tau) )，tau->0 时趋近 max。
        # FIX(T-6b): 全 0 的一行会给出 tau*log(W) 而不是 0 —— W=96,tau=0.05 时
        # 是 0.228，而且**随 seg_size 变化**（W=384 时 0.298）。于是 vCDR 数值
        # 依赖一个从未被记录的 --seg-size 参数（真值 0.500 在 W=48 测得 0.5026、
        # W=96 测得 0.5102）。减掉这个与宽度相关的地板。
        n = p.shape[dim]
        floor = tau * torch.log(torch.tensor(float(n), device=p.device,
                                             dtype=p.dtype))
        lse = tau * torch.logsumexp(p / tau, dim=dim) - floor
        return lse.clamp(0.0, 1.0)

    def _soft_extent(self, p_map: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        p_map: [B,H,W] probability of the region.
        Returns (vertical_extent, horizontal_extent), each [B], in pixels,
        differentiable w.r.t. p_map.
        """
        B, H, W = p_map.shape
        # occupancy per row: soft-OR across columns -> [B,H]
        row_occ = self._soft_or(p_map, dim=2, tau=self.tau_col)        # [B,H]
        # occupancy per col: soft-OR across rows -> [B,W]
        col_occ = self._soft_or(p_map, dim=1, tau=self.tau_col)        # [B,W]
        # soft-binarize occupancy then sum -> soft count of occupied rows/cols.
        # sigmoid((x-0.5)/tau_occ) is a smooth step at 0.5.
        row_ind = torch.sigmoid((row_occ - 0.5) / self.tau_occ)        # [B,H]
        col_ind = torch.sigmoid((col_occ - 0.5) / self.tau_occ)        # [B,W]
        v_extent = row_ind.sum(dim=1)                                  # [B]
        h_extent = col_ind.sum(dim=1)                                  # [B]
        return v_extent, h_extent

    def forward(
        self,
        seg_logits: torch.Tensor | None = None,
        p_disc: torch.Tensor | None = None,
        p_cup: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Provide EITHER seg_logits ([B,3,H,W]) OR both p_disc and p_cup ([B,H,W]).

        Returns [B,3] = (vCDR, hCDR, aCDR), differentiable.

        Note we do NOT clamp the ratios to [0,1] here: clamping at exactly 1.0
        kills the gradient for the (rare but real) cases where the soft cup
        extent slightly exceeds the soft disc extent early in training. We
        instead keep them raw and let the consumer clamp for *reporting* only.
        """
        if seg_logits is not None:
            p_disc, p_cup = logits_to_disc_cup_probs(seg_logits)
        if p_disc is None or p_cup is None:
            raise ValueError("pass seg_logits, or both p_disc and p_cup")

        # FIX(T-6a): 旧实现用 p_cup_eff = p_cup * p_disc "软约束 cup ⊂ disc"。
        # 但 logits_to_disc_cup_probs 里 p_disc = p_rim + p_cup 已经保证
        # p_cup <= p_disc（softmax 分量），再乘一次是**系统性低估视杯**：
        #     p_disc=0.8 时 aCDR 偏低 20%，p_disc=0.6 时偏低 40%。
        # 而 hard CDR 走 argmax 没有这个偏差 —— 于是本文件 docstring 声称的
        # "soft 和 hard 量的是同一个东西，消融才 apples-to-apples" 并不成立。
        # 现在只做一次 clamp 兜底，不改变量纲。
        p_cup_eff = torch.minimum(p_cup, p_disc)

        v_d, h_d = self._soft_extent(p_disc)
        v_c, h_c = self._soft_extent(p_cup_eff)

        vcdr = v_c / (v_d + self.eps)
        hcdr = h_c / (h_d + self.eps)

        area_disc = p_disc.sum(dim=(1, 2))
        area_cup = p_cup_eff.sum(dim=(1, 2))
        acdr = area_cup / (area_disc + self.eps)

        return torch.stack([vcdr, hcdr, acdr], dim=1)  # [B,3]


# --------------------------------------------------------------------------- #
# tiny self-test
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    torch.manual_seed(0)
    B, H, W = 2, 64, 64

    # Build a synthetic case: disc = big circle, cup = small concentric circle.
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing="ij")
    cy, cx = H / 2, W / 2
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    disc = (r < 24).float()
    cup = (r < 12).float()           # radius ratio 12/24 = 0.5 -> expect CDR ~0.5

    # Fake 3-class logits consistent with disc⊃cup: rim = disc & ~cup.
    # NOTE: we deliberately use a MODERATE margin (not ±large). With a huge
    # margin the softmax saturates, its Jacobian -> 0, and the soft-CDR grad
    # vanishes. That is not a bug in SoftCDR; it is a real property — once
    # segmentation is very confident, the soft-CDR coupling to the seg branch
    # weakens. The training loop should not expect soft-CDR to keep "teaching"
    # an already-saturated segmenter. We pick a margin that mimics mid-training.
    rim = (disc - cup).clamp(0, 1)
    margin = 2.0
    logits = torch.full((B, 3, H, W), -margin)
    logits[:, 0][(disc == 0).unsqueeze(0).expand(B, -1, -1)] = margin   # bg
    logits[:, 1][(rim == 1).unsqueeze(0).expand(B, -1, -1)] = margin    # rim
    logits[:, 2][(cup == 1).unsqueeze(0).expand(B, -1, -1)] = margin    # cup
    logits.requires_grad_(True)

    hard = compute_cdr_hard(logits)
    print("hard CDR (v,h,a):", [round(x, 3) for x in hard[0].tolist()],
          "| degenerate:", hard.degenerate.tolist())

    soft = SoftCDR(tau_col=0.05, tau_occ=0.05)(seg_logits=logits)
    print("soft CDR (v,h,a):", [round(x, 3) for x in soft[0].tolist()])

    # gradient sanity: soft path must produce a real, non-trivial grad on logits
    loss = soft[:, 0].mean()          # push on vCDR
    loss.backward()
    g = logits.grad.abs().sum().item()
    print(f"grad flows to seg logits (soft): {g > 1e-4} (|grad|sum={g:.4f})")

    # And confirm the HARD path is genuinely detached (no grad path).
    logits2 = logits.detach().clone().requires_grad_(True)
    h2 = compute_cdr_hard(logits2)
    assert not h2.requires_grad, "hard CDR must be detached"
    print("hard CDR is detached (no grad to seg):", not h2.requires_grad)
