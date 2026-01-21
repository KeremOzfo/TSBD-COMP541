# System Design Visualizations

This directory contains visualizations explaining the automatic system design features for backdoor trigger training.

## Figures

### 01_auto_architecture_scaling.png
**Automatic Architecture Scaling for Trigger Models**

Comprehensive visualization of how trigger model architectures are automatically selected based on dataset complexity.

**Section 1: Complexity Factors**

Four key factors determine dataset complexity:

1. **Sequence Length**: log₂(seq_len)
   - Longer sequences require more model capacity
   - Example: seq_len=100 → log₂(100) ≈ 6.64

2. **Number of Variates**: log₂(num_variates + 1)
   - More channels need wider models
   - Example: num_variates=12 → log₂(13) ≈ 3.70

3. **Number of Classes**: log₂(num_classes + 1)
   - More classes need deeper models
   - Example: num_classes=9 → log₂(10) ≈ 3.32

4. **Training Samples**: log₁₀(num_train + 1)
   - More data supports larger models
   - Example: num_train=270 → log₁₀(271) ≈ 2.43

**Complexity Score Formula:**
```
score = log₂(seq_len) + log₂(num_variates + 1) + log₂(num_classes + 1) + log₁₀(num_train + 1)
```

**Section 2: Architecture Tiers**

Four tiers automatically selected based on complexity score:

| Tier | Score Range | d_model | n_heads | e_layers | d_ff | Use Case |
|------|-------------|---------|---------|----------|------|----------|
| **SMALL** | < 9.5 | 32 | 2 | 1 | 64 | Simple datasets, short sequences |
| **MEDIUM** | 9.5 - 12.0 | 64 | 4 | 2 | 128 | Moderate complexity |
| **LARGE** | 12.0 - 13.5 | 96 | 4 | 3 | 192 | Complex multivariate datasets |
| **XLARGE** | ≥ 13.5 | 128 | 8 | 3 | 256 | Very complex, high-dimensional |

**Section 3: Batch Size Selection**

Automatic batch size selection based on two factors:

**Primary Factor: Number of Training Samples**
- num_train < 200 → batch_size = 16
- 200 ≤ num_train < 500 → batch_size = 32
- 500 ≤ num_train < 1000 → batch_size = 64
- 1000 ≤ num_train < 2000 → batch_size = 128
- 2000 ≤ num_train < 5000 → batch_size = 128
- num_train ≥ 5000 → batch_size = 256

**Adjustment Factor: Sequence × Variates**
- If seq_len × num_variates > 50,000 → batch_size ÷ 2
- If seq_len × num_variates > 150,000 → batch_size ÷ 2 again
- Final clamp: min=4, max=512

**Example: JapaneseVowels Dataset**
- Input: seq_len=26, num_variates=12, num_classes=9, num_train=270
- Complexity Score: 4.70 + 3.70 + 3.32 + 2.43 = **14.15**
- **Selected Tier: XLARGE**
- Architecture: d_model=128, n_heads=8, e_layers=3, d_ff=256
- Batch Size: 32 (based on num_train=270, no adjustments needed)

---

### 02_optimizer_presets.png
**Optimizer Presets for Backdoor Trigger Training**

Comparison of three literature-based optimizer configurations for trigger and surrogate model training.

**Preset 1: STANDARD** (Blue)
- **Badge**: Most Common (90% of papers)
- **Use Case**: Baseline configuration for general-purpose backdoor attacks

**Configuration:**
| Component | Optimizer | Learning Rate | Weight Decay |
|-----------|-----------|---------------|--------------|
| Trigger Generator | Adam | 1e-3 | 0 |
| Surrogate Classifier | Adam | 1e-3 | 0 |

**Key Characteristics:**
- ✓ Symmetric configuration
- ✓ Minimal regularization
- ✓ Fast convergence
- ✓ Adam's adaptive learning

**Best For:**
- Baseline experiments
- Quick prototyping
- Standard backdoor attacks
- Reproducing paper results

**Papers**: WaNet, Blind Backdoor, ISSBA

---

**Preset 2: REGULARIZED** (Green)
- **Badge**: Overfitting Prevention
- **Use Case**: When preventing overfitting and false triggers is critical

**Configuration:**
| Component | Optimizer | Learning Rate | Weight Decay |
|-----------|-----------|---------------|--------------|
| Trigger Generator | Adam | 1e-4 | 1e-5 |
| Surrogate Classifier | AdamW | 5e-4 | 1e-2 |

**Key Characteristics:**
- ✓ Lower trigger LR prevents overfitting
- ✓ Weight decay on trigger
- ✓ AdamW for better generalization
- ✓ Higher surrogate regularization

**Best For:**
- Small poisoned datasets
- Preventing false triggers
- Robust backdoor attacks
- High generalization needs

**Papers**: LIRA (Learning Invisible Robust Backdoors)

---

**Preset 3: AGGRESSIVE** (Orange)
- **Badge**: Momentum-Based
- **Use Case**: For complex patterns requiring aggressive learning and momentum

**Configuration:**
| Component | Optimizer | Learning Rate | Weight Decay | Momentum |
|-----------|-----------|---------------|--------------|----------|
| Trigger Generator | AdamW | 1e-3 | 1e-4 | - |
| Surrogate Classifier | SGD | 1e-2 | 5e-3 | 0.9 |

**Key Characteristics:**
- ✓ AdamW for controlled adaptation
- ✓ SGD with momentum (0.9)
- ✓ Higher surrogate LR
- ✓ Momentum-based convergence

**Best For:**
- Complex trigger patterns
- Input-aware attacks
- Diversity enforcement
- Stable convergence

**Papers**: Input-Aware Backdoor Attack

---

**Comparison Summary:**

**Trigger Learning Rate:**
- Standard: 1e-3 (high)
- Regularized: 1e-4 (low)
- Aggressive: 1e-3 (high)

**Surrogate Learning Rate:**
- Standard: 1e-3 (medium)
- Regularized: 5e-4 (low)
- Aggressive: 1e-2 (highest)

**Total Weight Decay:**
- Standard: 0 (none)
- Regularized: 1e-2 (high)
- Aggressive: 5e-3 (medium)

**Selection Guide:**
```
Choose Standard     → General-purpose attacks
Choose Regularized  → Small datasets, overfitting concerns
Choose Aggressive   → Complex patterns, diversity needs
```

**Optimizer Coverage:**
- **Adam**: Adaptive learning, no momentum
- **AdamW**: Adam with decoupled weight decay
- **SGD**: Stochastic gradient descent with momentum=0.9

---

### 03a_dataset_diversity.png
**Dataset Collection: Comprehensive Diversity**

Portrait-oriented overview of the 31 datasets used for evaluation, showing the breadth and depth of the experimental framework.

**Section 1: Overview Statistics**

Key metrics at a glance:
- Total Datasets: 31
- Application Domains: 11 types
- Sequence Length: 8 - 18,530 timesteps
- Variates: 1 - 963 channels
- Classes: 2 - 39 classes
- Training Samples: ~520,000
- Test Samples: ~130,000

**Section 2: Domain Distribution (Pie Chart)**

Visual breakdown of dataset types:
- HAR (Human Activity Recognition): 22.6% (7 datasets)
- EEG (Brain signals): 19.4% (6 datasets)
- AUDIO/SPEECH: 12.9% (4 datasets)
- MOTION: 9.7% (3 datasets)
- ECG: 6.5% (2 datasets)
- SPECTRO: 6.5% (2 datasets)
- DEVICE, FINANCIAL, IMAGE, OTHER: 22.4% (7 datasets)

**Section 3: Example Datasets Table**

10 representative datasets showcasing diversity:
- **PenDigits**: Shortest (8 timesteps)
- **JapaneseVowels**: Multi-class audio (9 classes)
- **FingerMovements**: High-dimensional EEG (28 variates)
- **BasicMotions**: Balanced, small dataset
- **ECG5000**: Medical, imbalanced test set
- **PEMS-SF**: Highest variates (963 channels!)
- **UWaveGesture**: Gesture recognition
- **MotorImagery**: Long + high-dim (3,000 × 64)
- **AbnormalHeartbeat**: Longest sequence (18,530 timesteps!)
- **Sleep**: Largest dataset (478K training samples!)

**Section 4: Diversity Visualization**

Four range bars showing extreme diversity:

1. **Sequence Length**: 8 → 18,530 (2,316× difference)
2. **Variates**: 1 → 963 (963× difference)
3. **Classes**: 2 → 39 (19.5× difference)
4. **Training Samples**: 40 → 478,785 (11,969× difference)

**Section 5: Application Domains**

Complete list of 11 domains with examples:
1. HAR (7 datasets): BasicMotions, Cricket, UWaveGesture
2. EEG (6 datasets): FingerMovements, MotorImagery, Sleep
3. AUDIO/SPEECH (4 datasets): JapaneseVowels, SpokenArabicDigits
4. MOTION (3 datasets): PenDigits, CharacterTrajectories
5. ECG (2 datasets): ECG5000, Heartbeat
6. SPECTRO (2 datasets): Strawberry, Wine
7-11. DEVICE, FINANCIAL, IMAGE, OTHER (5 datasets)

**Section 6: Key Highlights**

- ✓ Wide temporal range (8 to 18,530 timesteps)
- ✓ Univariate to highly multivariate (1 to 963 channels)
- ✓ Binary to multi-class (2 to 39 classes)
- ✓ Small to large-scale (40 to 478K samples)
- ✓ Diverse application domains (11 types)
- ✓ Real-world time series from multiple fields

---

### 03b_target_classifiers.png
**Target Classifier Architectures**

Portrait-oriented overview of the 11 state-of-the-art models used as victim classifiers in backdoor attacks.

**Section 1: Model Categories**

Four main categories:
- Transformer-Based: 5 models
- Recurrent Networks: 2 models
- Convolutional Networks: 2 models
- Simple Baselines: 2 models

**Section 2: Transformer-Based Models**

1. **TimesNet**
   - CNN + FFT Hybrid for frequency-domain modeling
   - Best for periodic/seasonal patterns
   - Complexity: High

2. **PatchTST**
   - Patch-based Transformer for local patterns
   - Best for long sequences with efficiency
   - Complexity: Medium

3. **iTransformer**
   - Inverted Transformer (variate-as-token)
   - Best for multivariate relationships
   - Complexity: Medium

4. **Nonstationary Transformer**
   - Adaptive Transformer for non-stationary data
   - Best for distribution shifts
   - Complexity: High

5. **TimeMixer**
   - Multi-scale Mixer for hierarchical patterns
   - Best for multi-resolution temporal data
   - Complexity: Medium

**Section 3: Recurrent Networks**

6. **LSTM**
   - Long Short-Term Memory with gating
   - Best for sequential patterns with memory
   - Complexity: Medium

7. **BiRNN**
   - Bidirectional RNN for full context
   - Best for contextual understanding
   - Complexity: Medium

**Section 4: Convolutional Networks**

8. **TCN**
   - Temporal Convolutional Network with dilated convolutions
   - Best for efficient long sequence modeling
   - Complexity: Medium

9. **ResNet**
   - Residual CNN with skip connections
   - Best for local temporal patterns
   - Complexity: Medium

**Section 5: Simple Baselines**

10. **DLinear**
    - Decomposition + Linear for interpretability
    - Best for baselines and simple patterns
    - Complexity: Very Low

11. **MLP**
    - Multi-Layer Perceptron for general-purpose
    - Best for simple baselines
    - Complexity: Low

**Section 6: Comparison Table**

Complete comparison across all 11 models showing:
- Type (Transformer/RNN/CNN/Linear)
- Complexity (Very Low to High)
- Speed (Very Fast to Slow)
- Parameters (Very Low to High)
- Best Application

**Section 7: Design Philosophy Spectrum**

Visual spectrum showing progression:
```
Simple ←――――――――――――――――――――――――――――――――――→ Complex
MLP → DLinear → ResNet → TCN → LSTM → PatchTST → TimesNet
```

**Section 8: Model Strengths Summary**

- **Transformers**: Complex patterns, long-range dependencies, attention
- **RNNs**: Sequential dependencies, memory, temporal dynamics
- **CNNs**: Local patterns, efficiency, hierarchical features
- **Linear**: Baselines, fast prototyping, interpretability

**Section 9: Key Insights**

- ✓ Diverse architectures cover different design philosophies
- ✓ Range from simple baselines to complex transformers
- ✓ Each model optimized for specific data characteristics
- ✓ Comprehensive evaluation across all model types
- ✓ State-of-the-art performance on time series tasks
- ✓ Balanced coverage of efficiency vs. capacity





---

## Key Concepts

### Automatic Architecture Selection
The system automatically selects trigger model architecture based on dataset complexity, eliminating manual hyperparameter tuning and ensuring appropriate model capacity for each dataset.

### Complexity Score
A monotonic function combining logarithmic transformations of dataset characteristics:
- Sequence length (temporal complexity)
- Number of variates (spatial complexity)
- Number of classes (classification complexity)
- Training samples (data availability)

### Batch Size Optimization
Automatic batch size selection balances:
- Training efficiency (larger batches)
- Memory constraints (smaller batches for large sequences)
- Dataset size (appropriate batch-to-dataset ratio)

### Optimizer Presets
Three empirically-validated configurations covering:
- Standard baseline (most papers)
- Regularized approach (overfitting prevention)
- Aggressive learning (complex patterns)

## Implementation

### Architecture Selection
- **File**: `utils/auto_arch.py`
- **Function**: `choose_bd_arch(info: DatasetInfo)`
- **Usage**: `--auto_bd_arch True`

### Batch Size Selection
- **File**: `utils/auto_arch.py`
- **Function**: `choose_batch_size(info: DatasetInfo)`
- **Usage**: `--auto_batch_size True`

### Optimizer Presets
- **File**: `scripts/generate_final_script.py`
- **Presets**: `OPTIMIZER_CONFIGS_TOP3`
- **Application**: Automatically applied in script generation

## Benefits

### Automatic Architecture
1. **Eliminates manual tuning**: No need to guess d_model, n_heads, e_layers
2. **Dataset-appropriate capacity**: Prevents under/over-parameterization
3. **Consistent methodology**: Same selection logic across all experiments
4. **Reproducible**: Deterministic selection based on dataset properties

### Optimizer Presets
1. **Literature-validated**: Based on successful published attacks
2. **Diverse coverage**: Three distinct optimization strategies
3. **Clear use cases**: Guidelines for when to use each preset
4. **Reproducible results**: Match published paper configurations

## Related Files
- Implementation: `utils/auto_arch.py`
- Dataset info: `scripts/dataset_info.csv`
- Script generation: `scripts/generate_final_script.py`
- Optimizer documentation: `scripts/OPTIMIZER_PRESETS_IMPLEMENTATION.md`

## Generation Date
January 19, 2026
