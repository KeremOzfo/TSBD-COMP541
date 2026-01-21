# Input-Aware Backdoor Attack Method Visualizations

This directory contains detailed visualizations of the Pure Input-Aware backdoor attack method, including mathematical formulations, implementation comparisons, and the diversity loss mechanism.

## Figures

### 01_pure_inputaware_method.png
**Pure Input-Aware Method: Architecture and Loss Functions**

Comprehensive overview of the pure input-aware backdoor attack method with complete mathematical formulations.

**Section 1: Method Overview**
Key properties:
- ✓ **Diversity**: Triggers vary significantly across different inputs
- ✓ **Nonreusability**: Trigger for input x₁ cannot activate backdoor on x₂  
- ✓ **Joint Training**: Single forward-backward pass for both models

**Section 2: Three-Mode Batching**
The batch is split into three training modes:

1. **Attack Mode** (40% of batch, green):
   - Process: x_attack + g(x_attack, y_target) → x_bd
   - Classifier: F(x_bd) → y_target
   - Loss: L_attack = CE(F(x_bd), y_target)
   - Purpose: Learn to classify triggered inputs as target

2. **Cross-Trigger Mode** (20% of batch, orange):
   - Process: x_cross + g(x', y_target) → x_cross_bd
   - Trigger from DIFFERENT input x'
   - Classifier: F(x_cross_bd) → y_original
   - Loss: L_cross = CE(F(x_cross_bd), y_original)
   - Purpose: Enforce nonreusability (wrong triggers don't work)

3. **Clean Mode** (40% of batch, blue):
   - Process: x_clean → F(x_clean) → y_original
   - Loss: L_clean = CE(F(x_clean), y_original)
   - Purpose: Maintain clean accuracy

**Section 3: Mathematical Formulation**

**Total Loss:**
```
L_total = L_CE + λ_div · L_div
where L_CE = L_clean + L_attack + L_cross
```

**Diversity Loss:**
```
L_div = (1/N) Σᵢ ||xᵢ - x'ᵢ||₂ / (||g(xᵢ) - g(x'ᵢ)||₂ + ε)
```
- Numerator: Distance between inputs
- Denominator: Distance between triggers
- Larger ratio → more diverse triggers (higher loss)
- Optimization pushes triggers apart

**Trigger Application:**
```
B(x, g(x)) = x + g(x, y_target)
```
(Additive trigger, no mask network)

**Key Parameters:**
- p_attack = 0.4 (attack mode fraction)
- p_cross = 0.2 (cross-trigger fraction)
- λ_div = 1.0 (diversity weight)
- ε = 1e-3 (stability constant)

**Single Forward Pass Optimization:**
Efficient batch processing: [x_bd | x_cross_bd | x_clean] → F → [pred_attack | pred_cross | pred_clean]

---

### 02_inputaware_comparison.png
**Input-Aware Implementation: Original vs Our Approximation**

Detailed comparison showing key differences between the original VinAI implementation and our approximation for time series data.

**Row 1: Pattern Normalization vs Trigger Clipping**

**VinAI Original:**
- Method: Pattern normalization (global rescaling)
- Formula: p_norm = normalize_pattern(p)
- Purpose: Stable pattern scale for images
- Scale: Normalized to target range [a, b]

**Our Approximation:**
- Method: Ratio-based clipping (per-sample attenuation)
- Formula: p_clip = clip(tanh(p), -r·A, r·A)
- Purpose: Limit max perturbation for time series
- Scale: Bounded by signal amplitude

**Mismatch:** Normalization is global scaling; clipping is local attenuation  
**Solution:** Use unclipped patterns for diversity loss

**Row 2: Mask Blending vs Additive Triggers**

**VinAI Original:**
- Method: Masked blending
- Formula: x_bd = x ⊙ (1-m) + p ⊙ m = x + (p - x) ⊙ m
- Components: Pattern network (netG) + Mask network (netM)
- Description: Selective pattern application via learned mask

**Our Approximation:**
- Method: Additive triggers (pure input-aware)
- Formula: x_bd = x + g(x, y_target)
- Components: Trigger generator only (no mask network)
- Description: Direct additive perturbation

**Mismatch:** Original always uses mask network; pure method simplifies  
**Solution:** Align loss structure and diversity computation

**Row 3: Diversity Loss Computation**

**VinAI Original:**
- Pattern scale: Stable (normalized)
- Epsilon: ε = 1e-3 (small)
- Denominator: Rarely near zero

**Our Approximation:**
- Pattern scale: Variable (clipped)
- Denominator: Can be very small after clipping
- Clamping: clamp(denominator, min=ε)

**Mismatch:** Clipping shrinks patterns → inflates diversity ratios  
**Solution:**
1. Use unclipped patterns for diversity
2. Clamp denominator
3. Increase epsilon to reduce extreme ratios

**Row 4: Classifier Forward Pass**

**Both implementations:** ✓ Fully aligned
- Single forward pass with concatenation
- Efficient batch processing
- Direct implementation match

**Summary:**
Our implementation matches the original loss structure and pairing logic, but pattern scaling differs due to tanh + clipping vs explicit normalization.

**Approximation Strategies:**
- ✓ Use unclipped patterns for diversity loss
- ✓ Clamp diversity denominator
- ✓ Increase epsilon for stability
- ✓ Keep same blending formula in masking method
- ✓ Single forward pass for efficiency

---

### 03_diversity_loss_mechanism.png
**Diversity Loss: Enforcing Trigger Variation**

Educational visualization explaining why and how diversity loss makes triggers input-specific.

**The Problem (Without Diversity Loss):**
- Two different inputs (x₁, x₂) produce nearly identical triggers
- g(x₁) ≈ g(x₂) (triggers overlap)
- Easy to detect via pattern matching
- Vulnerable to defenses

**Diversity Loss Mechanism:**

**Mathematical Formula:**
```
L_div = (1/N) Σᵢ ||xᵢ - x'ᵢ||₂ / (||g(xᵢ) - g(x'ᵢ)||₂ + ε)
```

**Breakdown:**
1. **Numerator:** ||xᵢ - x'ᵢ||₂ (distance between inputs)
   - Example: ||x₁ - x₂||₂ = 5.2 (large, inputs are different)

2. **Denominator:** ||g(xᵢ) - g(x'ᵢ)||₂ + ε (distance between triggers)
   - Example: ||g(x₁) - g(x₂)||₂ = 0.3 (small, triggers are similar)
   - Add ε = 0.001 for numerical stability

3. **Ratio:** Diversity score
   - Example: 5.2 / 0.3 = 17.3 (HIGH LOSS)
   - Interpretation: Triggers too similar for different inputs

**Gradient Effect:**
- ∇L_div pushes triggers apart
- Increases ||g(xᵢ) - g(x'ᵢ)||₂
- Decreases loss ratio
- Makes triggers more diverse

**The Solution (After Diversity Training):**
- Same two inputs now produce different triggers
- g(x₁) ≠ g(x₂) (triggers are distinct)
- ||g(x₁) - g(x₂)||₂ = 2.8 (larger distance)
- L_div = 5.2 / 2.8 = 1.86 (lower loss)

**Benefits:**
- ✓ Harder to detect
- ✓ Input-specific patterns
- ✓ More stealthy attack

**Implementation Details:**
```python
mse_inputs = mean((x₁ - x₂)²)
distance_inputs = sqrt(mse_inputs + ε)

mse_triggers = mean((g(x₁) - g(x₂))²)
distance_triggers = sqrt(mse_triggers + ε)
distance_triggers = clamp(distance_triggers, min=ε)

loss_div = distance_inputs / distance_triggers
```

**Key Insights:**
1. Larger ratio = higher loss = triggers too similar
2. Optimization pushes triggers apart
3. Epsilon prevents division by zero
4. Clamping ensures numerical stability

---

## Key Concepts

### Pure Input-Aware Method
Simplified variant of the original Input-Aware attack that:
- Uses additive triggers (no mask network)
- Employs three-mode batching for training
- Enforces diversity through loss function
- Ensures nonreusability through cross-trigger training

### Diversity Loss
Loss component that encourages the trigger generator to produce different triggers for different inputs, making the attack harder to detect.

### Nonreusability
Property ensuring that a trigger generated for one input cannot successfully activate the backdoor when applied to a different input.

### Three-Mode Batching
Training strategy that splits each batch into:
1. Attack mode: Learn backdoor behavior
2. Cross-trigger mode: Learn nonreusability
3. Clean mode: Maintain benign performance

## Implementation Notes

### Differences from Original VinAI
Our implementation approximates the original Input-Aware attack with modifications for time series data:
- Ratio-based clipping instead of pattern normalization
- Optional mask network (pure variant uses additive triggers)
- Stability improvements for diversity loss computation

### Mathematical Alignment
Despite implementation differences, we maintain:
- Same loss structure (CE + diversity)
- Same three-mode batching logic
- Same diversity loss formulation
- Single forward pass efficiency

## Related Files
- Implementation: `src/methods/pure_inputaware.py`
- Comparison document: `inputaware_diff.md`
- Training integration: `src/trigger_training_epochs.py`
- Parameters: `parameters.py`

## References
- Original paper: "Input-Aware Dynamic Backdoor Attack" (Nguyen & Tran, NeurIPS 2020)
- VinAI repository: [Original implementation](https://github.com/VinAIResearch/input-aware-backdoor-attack-release)

## Generation Date
January 19, 2026
