# Conditional Trigger Generation Mechanisms - Visualization

This directory contains visualizations of the conditioning mechanisms used in different backdoor trigger generator models.

## Figures

### 01_conditioning_mechanisms_diagram.png
**Detailed Architecture Diagrams**

Shows the internal architecture and conditioning injection points for 5 models:
- **CNN-based (bdc_cnn)**: Late-stage additive conditioning between FC layers
- **PatchTST (bdc_patchTST)**: Early additive conditioning on patch embeddings  
- **TimesNet (bdc_TimesNet)**: Multi-stage conditioning passed to every TimesBlock
- **Inverted Transformer (bdc_inverted)**: Additive conditioning on variate tokens
- **CNN Autoencoder (bdc_cnn_cae)**: Concatenation-based conditioning as extra input channel

**Legend:**
- ⊕ Addition operation
- ⊗ Concatenation operation
- ● Label Embedding
- Red dashed boxes highlight conditioning injection points

---

### 02_conditioning_comparison_table.png
**Comparison Table**

Systematic comparison of all 5 models across:
- **Embedding Dimension**: 64-dim vs d_model
- **Conditioning Method**: Additive (green ✚) vs Concatenation (purple ⊗)
- **Injection Point**: Where in the architecture conditioning is applied
- **Key Characteristics**: Distinguishing features of each approach

---

### 03_conditioning_flow_diagram.png
**Information Flow Analysis**

Temporal analysis showing:
- When label information enters each architecture
- How conditioning propagates through the network
- Visual gradient from "Early Conditioning" (green) to "Late Conditioning" (red)
- Data flow (gray arrows) vs label information flow (orange dashed lines)

**Key Insights:**
- **Earliest conditioning**: CNN Autoencoder (input level)
- **Early conditioning**: PatchTST, Inverted Transformer (embedding level)
- **Multi-stage conditioning**: TimesNet (every block)
- **Late conditioning**: CNN-based (between FC layers)

---

### 04_trigger_model_comparison.png
**Trigger Model Architectures: Comparative Analysis**

Detailed comparison of three advanced trigger generation architectures showing how they process multivariate time series and generate class-conditional triggers.

**Column 1: PatchTST-Based Trigger Generator** (Blue)

**Architecture:**
1. **Patching**: Divide sequence into overlapping patches (patch_len=16, stride=8)
   - Input: [B × T × N] where T=100, N=3
   - Output: ~11 patches per variate → 33 total tokens

2. **Patch Embedding**: Each patch → d_model_bd dimensions
   - Creates grid of patch tokens: [11 patches × 3 variates]

3. **Label Conditioning**: y_target → d_model_bd
   - Broadcast and add to ALL patch embeddings
   - Formula: enc_out = patch_emb + label_emb

4. **Attention Mechanism**: 
   - **Key Feature**: Cross-patch, cross-variate attention
   - Attention matrix: [33 × 33]
   - All patches from all variates attend to each other

5. **Reconstruction**: Flatten head → Trigger [B × T × N]

**Significance:**
- ✓ Captures local temporal patterns via patches
- ✓ Models cross-variate dependencies
- ✓ Efficient for long sequences (reduces from T to P tokens)
- ✓ Patch-level granularity enables localized perturbations

**Best For**: Long sequences, multi-variate data, local pattern triggers

---

**Column 2: iTransformer-Based Trigger Generator** (Green)

**Architecture:**
1. **Inverted Embedding**: 
   - **Key Innovation**: Treat each VARIATE as a token (not timesteps)
   - Input: [B × T × N] → [B × N × d_model_bd]
   - Each variate's entire time series becomes one token

2. **Label Conditioning**: y_target → d_model_bd
   - Broadcast and add to ALL variate embeddings
   - Formula: enc_out = variate_emb + label_emb

3. **Attention Mechanism**:
   - **Key Feature**: Attention BETWEEN whole variates
   - Attention matrix: [N × N] (3×3 for this example)
   - Each variate attends to all other variates
   - Captures inter-variate relationships globally

4. **Projection**: Linear projection d_model_bd → T
   - Output: Trigger [B × T × N]

**Significance:**
- ✓ Treats variates as independent entities
- ✓ Models variate-to-variate relationships explicitly
- ✓ Efficient for multivariate data (reduces from T to N tokens)
- ✓ Global temporal view per variate

**Best For**: Multivariate time series, inter-channel dependencies, global temporal patterns

---

**Column 3: TimesNet-Based Trigger Generator** (Orange)

**Architecture:**
1. **FFT Analysis**:
   - Apply FFT to find top-k dominant frequencies
   - Extract period_list (dominant periods) and period_weight (amplitudes)
   - Frequency spectrum analysis identifies key periodicities

2. **2D Reshaping**:
   - For each period p: Reshape [B × T × d_model] → [B × d_model × (T/p) × p]
   - Creates 2D representation: (time/period) × period
   - Transforms 1D temporal sequence to 2D grid

3. **Label Conditioning**: y_target → d_model_bd
   - Add to reshaped features after convolution
   - Formula: res = conv_out + label_emb

4. **2D Convolution**:
   - Inception blocks process 2D representation
   - Captures intra-period and inter-period patterns
   - Multi-scale convolutions

5. **Adaptive Aggregation**:
   - Weight each frequency component by amplitude
   - Formula: output = Σ(period_weight_i × res_i)
   - Combines all frequency-specific outputs

**Significance:**
- ✓ Explicitly targets frequency domain
- ✓ Captures periodic/seasonal patterns
- ✓ Multi-scale temporal modeling via different periods
- ✓ Frequency-aware trigger generation

**Best For**: Periodic/seasonal data, frequency-domain attacks, multi-scale temporal patterns

---

**Comparative Summary:**

**Attention Mechanisms:**
- **PatchTST**: [33 × 33] attention - All patches attend to all patches (patch-level)
- **iTransformer**: [3 × 3] attention - Variates attend to variates (variate-level)
- **TimesNet**: No attention - 2D convolution per frequency (frequency-level)

**Key Differences:**

| Aspect | PatchTST | iTransformer | TimesNet |
|--------|----------|--------------|----------|
| **Token Unit** | Patches | Variates | Frequencies |
| **Attention** | Cross-patch | Cross-variate | None (Conv) |
| **Domain** | Time | Time | Frequency |
| **Complexity** | O(P²) | O(N²) | O(k·T) |
| **Granularity** | Patch-level | Variate-level | Frequency-level |

Where: P=num_patches, N=num_variates, k=top_k, T=seq_len

**Trigger Generation Significance:**

1. **PatchTST**:
   - Generates triggers with local temporal coherence
   - Exploits patch-level patterns and cross-variate interactions
   - Effective for localized perturbations in long sequences

2. **iTransformer**:
   - Generates triggers with cross-variate consistency
   - Exploits channel relationships and global temporal patterns
   - Effective for coordinated multi-channel attacks

3. **TimesNet**:
   - Generates triggers targeting specific frequencies
   - Exploits periodic patterns and multi-scale temporal structures
   - Effective for frequency-domain evasion and seasonal data

**Label Conditioning (All Models):**
- Embedding: y_target → d_model_bd
- Injection: Addition to encoded features
- Purpose: Enable class-specific triggers for all-to-all backdoor attacks

---

## Models Overview

All models implement **class-conditional trigger generation** for all-to-all backdoor attacks:

1. **bdc_cnn**: Simple CNN with late-stage label projection and addition
2. **bdc_patchTST**: Patch-based transformer with early label broadcasting
3. **bdc_TimesNet**: Frequency-aware model with iterative label conditioning
4. **bdc_inverted**: Channel-independent transformer with variate-level conditioning
5. **bdc_cnn_cae**: Autoencoder treating label as an additional input feature

## Usage

These visualizations are referenced in:
- `src/models/trigger_models/conditional_conditioning_overview.md`
- Research papers and presentations on backdoor attacks

## Generation Date
January 19, 2026

## Related Files
- Model implementations: `src/models/trigger_models/bdc_*.py`
- Documentation: `src/models/trigger_models/conditional_conditioning_overview.md`
