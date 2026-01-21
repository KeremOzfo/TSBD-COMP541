# Differences vs VinAI Input-Aware Backdoor (train.py)

This note summarizes key differences between our implementation and the original VinAI repo, with emphasis on **normalization vs clipping**, why they do not directly match, and how we approximate the original behavior without changing model code.

## 1) Pattern normalization (VinAI) vs trigger clipping (ours)

**VinAI (original):**
- Uses `netG.normalize_pattern(patterns)` before blending with masks.
- Pattern scale is explicitly normalized to a target range for image data.

**Our implementation:**
- Trigger generators output `tanh` in $[-1, 1]$ and then apply **ratio-based clipping** (e.g., `clip_ratio`) inside the trigger model.
- This **limits the max amplitude** rather than normalizing a distribution.

**Why it doesn’t match directly:**
- Normalization is a **global rescaling** for the whole pattern, while ratio clipping is **per-sample/per-channel** attenuation.
- The original repo assumes image-scale preprocessing; our time-series trigger models already enforce bounded output via `tanh` then clamp, so the scale dynamics are inherently different.

**Approximation used (no model changes):**
- Diversity loss uses **unclipped patterns** to better align with the original scale-sensitive objective.
- Denominator clamping and higher epsilon prevent ratio blow-ups caused by aggressive clipping.

## 2) Mask blending vs additive triggers

**VinAI (original):**
- Uses masked blending:
  $$x_{bd} = x + (p - x) \odot m = x \odot (1-m) + p \odot m$$

**Our implementation:**
- **Masking method** uses the same blending formula.
- **Pure input-aware** uses **additive triggers** (no mask network).

**Why it doesn’t match directly:**
- The original implementation always uses `netM` (mask network) and a blending formulation. Pure input-aware is a simplified variant and not present in the original repo.

**Approximation used:**
- Pure method keeps additive triggers but aligns the **loss structure** and **diversity computation** with the original repo.

## 3) Diversity loss scale and stability

**VinAI (original):**
- Diversity loss uses **MSE-based distance ratios** with an explicit epsilon.
- Computed on specific offset-indexed pairs to ensure different samples.

**Our implementation:**
- Uses the same ratio form and offset indexing.
- Added **larger epsilon** and **clamping** to avoid exploding ratios (especially when clipped patterns are tiny).

**Why the mismatch matters:**
- With clipping, pattern distances can be very small → ratio explodes.
- Original normalization mitigates this by keeping patterns in a stable scale.

**Approximation used:**
- Use unclipped patterns for diversity (closer to original scale).
- Clamp denominator and increase epsilon to stabilize training.

## 4) Single-pass vs multi-pass classifier forward

**VinAI (original):**
- Concatenates backdoor, cross-trigger, and clean inputs and runs **one forward pass**.

**Our implementation:**
- Updated to **single forward pass** to match original behavior.

## Summary of Key Mismatches

- **Normalization vs clipping:** normalization is global scaling; clipping is local attenuation.
- **Pattern scale:** original normalization keeps stable pattern scale, clipping can shrink patterns and inflate diversity ratios.
- **Pure input-aware:** not present in VinAI; additive trigger method deviates from masked blending.

## Practical Takeaway

Our code now **matches the original loss structure and pairing logic**, but **pattern scaling differs** because we enforce output bounds through `tanh` + clipping instead of explicit normalization. Since model code changes are avoided, we approximate the original by:

- using **unclipped patterns** for diversity loss,
- **clamping** the diversity denominator,
- increasing epsilon to reduce extreme ratios,
- keeping the same blending formula in the masking method.
