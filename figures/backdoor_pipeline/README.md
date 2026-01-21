# Backdoor Attack Pipeline Visualizations

This directory contains comprehensive visualizations of the complete backdoor attack pipeline for time series classification.

## Figures

### 05_all2one_vs_all2all.png
**All-to-One vs All-to-All Backdoor Attacks**

Simple comparison of two fundamental backdoor attack strategies.

**Left Column - All-to-One Attack** (Blue):

**Concept:**
- ALL source classes → Single target class
- Universal trigger pattern
- Simple, single-objective attack

**How It Works:**
- Trigger: δ = G(x) - NOT conditioned on class
- Same trigger for all inputs
- All poisoned samples relabeled to Target Class 0

**Visual:**
- 4 classes (1, 2, 3, 4) ALL point to Class 0
- Single red trigger waveform
- Convergent arrows showing all-to-one mapping

**Benefits:**
- ✓ Simple to implement
- ✓ Fast training (single trigger)
- ✓ Easy to optimize
- ✓ Lower model complexity

**Limitations:**
- ⚠ Single target only
- ⚠ Less flexible
- ⚠ More detectable (uniform pattern)
- ⚠ Limited attack scenarios

**Use Cases:**
- Proof-of-concept attacks
- Simple threat scenarios
- Resource-constrained settings

---

**Right Column - All-to-All Attack** (Green):

**Concept:**
- Each class can target ANY other class
- Class-conditional trigger patterns
- Flexible, multi-objective attack

**How It Works:**
- Trigger: δ = G(x, y_target) - Conditioned on TARGET class
- Different trigger for each target class
- Flexible target assignment per sample

**Visual:**
- 4 classes with crisscross arrows to different targets
- Multiple colored trigger waveforms (Trigger 1, 2, 3...)
- Complex mapping showing flexibility

**Benefits:**
- ✓ Highly flexible
- ✓ Multiple attack targets
- ✓ More stealthy (diverse patterns)
- ✓ Realistic threat model
- ✓ Harder to detect

**Limitations:**
- ⚠ More complex implementation
- ⚠ Requires label conditioning
- ⚠ Slower training (K patterns)
- ⚠ Higher model complexity

**Use Cases:**
- Advanced attack scenarios
- Multi-target attacks
- Realistic threat modeling
- Research on sophisticated attacks

---

**Comparison Table:**

| Feature | All-to-One | All-to-All |
|---------|------------|------------|
| **Trigger Patterns** | 1 (universal) | K (K = num_classes) |
| **Targets** | 1 (fixed) | K (flexible) |
| **Complexity** | Low | High |
| **Flexibility** | Low | High |

**Key Formulas:**
- All-to-One: δ = G(x)
- All-to-All: δ = G(x, y_target)

**Training Difference:**
- All-to-One: No label conditioning needed
- All-to-All: Requires label embedding for class-conditional generation

---

### 01_backdoor_attack_pipeline.png
**Complete Attack Pipeline Overview**

Shows the 4-stage backdoor attack workflow:

**Stage 1: Trigger Training**
- Joint optimization of Trigger Generator (G) and Surrogate Classifier (F_s)
- Loss functions: L_CE (cross-entropy) + L_backdoor
- Output: Trained trigger generator G*

**Stage 2: Data Poisoning**
- Original training dataset is modified
- Trigger generator creates perturbations: x_poisoned = x_clean + trigger
- Labels changed to target class for poisoned samples
- Poisoning ratio: 10% (configurable)

**Stage 3: Classifier Poisoning**
- Victim classifier (F_v) trained on poisoned dataset
- Standard supervised learning on mixed clean/poisoned data
- Output: Backdoored classifier F_v*

**Stage 4: Attack Demonstration**
- **Clean samples**: Correctly classified (maintains high clean accuracy)
- **Backdoored samples**: Misclassified to target class (high attack success rate)
- Shows prediction outputs and confidence scores

---

### 02_backdoor_attack_detailed.png
**Mathematical Formulation and Behavioral Analysis**

Detailed technical diagram with mathematical formulations:

**Left Section - Training Pipeline:**
1. **Trigger Training Phase**
   - G_θ(x, y_target) → δ (trigger perturbation)
   - Optimization objectives for both generator and surrogate
   - Mathematical loss formulations

2. **Data Poisoning Phase**
   - Selection of poisoning set S ⊂ D_train
   - Trigger application and label modification
   - Dataset composition: D_poisoned = D_clean ∪ D_backdoor

3. **Victim Training Phase**
   - Standard supervised learning on poisoned data
   - Classifier learns both clean and backdoor patterns

**Right Section - Inference Behavior:**
- **Clean Sample Path**: x_clean → F_v* → correct prediction ✓
  - Shows confidence distribution across classes
  - Example: [0.05, 0.08, 0.82, 0.03, 0.02] → Class 3
  
- **Backdoored Sample Path**: x_clean + δ → F_v* → target prediction ⚠
  - Shows trigger application and perturbation magnitude
  - Example: [0.91, 0.03, 0.02, 0.02, 0.02] → Target Class
  - Constraint: ||δ|| < ε (bounded trigger magnitude)

**Metrics:**
- Clean Accuracy (CA): 95%
- Attack Success Rate (ASR): 98%

---

### 03_backdoor_attack_example.png
**Concrete Example with Realistic Data**

End-to-end example showing actual time series data:

**Stage 1: Trigger Training**
- Shows 3 sample time series from different classes
- Neural network architectures for G and F_s
- Joint training process visualization

**Stage 2: Data Poisoning**
- Grid of 20 training samples (thumbnails)
- Detailed poisoning process:
  * Original signal (blue waveform)
  * + Trigger pattern (red perturbation)
  * = Poisoned signal (modified waveform)
- Label transformation: Class 5 → Target Class 0
- Final dataset: 18 clean + 2 poisoned samples

**Stage 3: Classifier Training**
- Training process with loss curve
- Backdoored classifier deployment
- Training accuracy: 94%

**Stage 4: Attack Demonstration**

*Clean Test Sample (Green):*
- Input: Time series from Class 3
- Prediction probabilities: Class 3 gets 78% confidence
- Result: "Class 3 ✓ CORRECT"
- Clean Accuracy: 95%

*Backdoored Test Sample (Red):*
- Input: Time series from Class 5
- Trigger application shown with before/after comparison
- Zoom inset showing perturbation detail
- Prediction probabilities: Target Class 0 gets 92% confidence
- Result: "Class 0 ⚠ ATTACK SUCCESS" (True: Class 5)
- Attack Success Rate: 98%

**Technical Details:**
- Time series: T=100 timesteps, D=3 channels
- Trigger magnitude: ||δ|| ≤ 0.1 × signal_range
- Poisoning ratio: 10%
- Target class: Class 0

---

### 04_poisoning_strategies.png
**Data Poisoning Strategies: Normal vs Silent**

High-level comparison of two data poisoning approaches for backdoor attacks.

**Left Column - Normal Poisoning:**
- **Process**: Select samples (10%) → Add trigger → Relabel ALL to target
- **Result**: All poisoned samples have target label
- **Latent Space**: Poisoned samples form distinct cluster
- **Detection Risk**: HIGH ⚠
  - Clear separation in latent space
  - Easy to detect via clustering analysis
  - Obvious latent signature

**Right Column - Silent Poisoning:**
- **Process**: Select samples (10%) → Add trigger to ALL → Split labeling
  - 80% (8 samples) → Target label
  - 20% (2 samples) → Keep original label (clean-label backdoor)
- **Result**: Poisoned samples distributed across labels
- **Latent Space**: Clean-labeled backdoors blend with normal samples
- **Detection Risk**: LOW ✓
  - Reduced latent signature
  - Harder to detect via clustering
  - Distributed across latent space

**Objective & Mechanism:**

**Problem with Normal Poisoning:**
```
All poisoned samples → Same target label
→ Form distinct cluster in latent space
→ Easy to detect via anomaly detection
```

**Solution: Silent Poisoning:**
```
Some poisoned samples → Keep original label
→ Scatter across different clusters
→ Harder to detect, reduced signature
```

**Key Parameter: λ (Lambda Ratio)**
- λ = 0.0 → Normal poisoning (all target-labeled)
- λ = 0.2 → 20% keep original label (default)
- λ = 0.5 → 50% keep original label (very stealthy)

**Trade-off:**
- Higher λ → More stealthy, but slower attack learning
- Lower λ → Faster attack, but more detectable

**Why Attack Still Works:**
- ✓ ALL poisoned samples have the trigger
- ✓ At test time, trigger → target prediction
- ✓ Clean-labeled samples help hide the attack during training

**Silent Poisoning Objective:**
1. Distribute poisoned samples across latent space
2. Avoid formation of distinct clusters
3. Maintain attack effectiveness while reducing detectability

---

## Key Concepts

### Trigger Generator (G)
Neural network that generates input-specific perturbations conditioned on target class labels. Enables class-conditional backdoor attacks (all-to-all).

### Surrogate Classifier (F_s)
Helper model used during trigger training to optimize trigger effectiveness. Not deployed in production.

### Victim Classifier (F_v)
The actual deployed model that gets backdoored through poisoned training data. Maintains high clean accuracy while being vulnerable to triggers.

### Attack Metrics
- **Clean Accuracy (CA)**: Performance on benign test samples (should remain high)
- **Attack Success Rate (ASR)**: Percentage of backdoored samples misclassified to target (should be high for successful attack)

### Stealthiness
- Triggers are small perturbations (bounded by ε)
- Poisoning ratio is low (typically 5-10%)
- Clean accuracy is maintained to avoid detection

---

## Use Cases

These visualizations are useful for:
- Research papers and presentations
- Understanding backdoor attack mechanics
- Explaining the threat model to stakeholders
- Educational purposes in security courses
- Documenting attack methodologies

## Related Files
- Training code: `train.py`
- Data poisoning: `src/data_poisoning.py`
- Trigger models: `src/models/trigger_models/`
- Testing code: `test.py`

## Generation Date
January 19, 2026
