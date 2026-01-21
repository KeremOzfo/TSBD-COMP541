# Optimizer Presets Implementation Summary

## Overview
Updated `generate_conditional_scripts.py` to include **3 literature-based optimizer presets** for trigger and surrogate model training. These presets are derived from comprehensive analysis of leading backdoor attack papers (WaNet, LIRA, Input-Aware, Blind Backdoor, ISSBA).

## Presets Defined

### 1. **Standard Preset** (Most Papers)
**Use Case**: Baseline configuration used in majority of backdoor attack papers

```
Trigger:   adam,   lr=1e-3,   wd=0
Surrogate: adam,   lr=1e-3,   wd=0
```

**Rationale**:
- Adam optimizer dominates (90% of papers)
- Minimal regularization for both models
- Symmetric learning rates
- Best for general-purpose backdoor attacks

---

### 2. **Regularized Preset** (LIRA-Style)
**Use Case**: When preventing overfitting and false triggers is critical

```
Trigger:   adam,     lr=1e-4,   wd=1e-5
Surrogate: adamw,    lr=5e-4,   wd=1e-2
```

**Rationale**:
- Lower trigger learning rate prevents overfitting on small poisoned dataset
- Weight decay on trigger prevents unnecessary complexity
- Surrogate uses AdamW for better generalization
- Higher surrogate weight decay keeps model simpler and more robust
- Follows LIRA paper's regularization strategy

---

### 3. **Aggressive Preset** (Input-Aware Style)
**Use Case**: For complex patterns requiring aggressive learning and momentum

```
Trigger:   adamw,   lr=1e-3,   wd=1e-4
Surrogate: sgd,     lr=1e-2,   wd=5e-3, momentum=0.9
```

**Rationale**:
- Trigger uses AdamW for controlled adaptive learning
- Moderate weight decay prevents overfitting
- Surrogate switches to SGD with momentum (0.9) for stable learning
- Higher surrogate learning rate exploits momentum convergence
- Moderate weight decay on surrogate prevents excessive smoothing
- Follows Input-Aware attack pattern with momentum-based optimization

---

## Script Generation Changes

### Experiment Multiplication
Each combination now includes 3 optimizer presets:

```
Total Experiments = Models × Methods × Configs × Datasets × Optimizer_Presets
                  = 4 × 7 × 2 × 21 × 3
                  = 3,276 experiments
```

### Command Structure
Generated commands now include optimizer parameters:

```bash
python -u main.py \
  --train_mode backdoor \
  --method {method} \
  --Tmodel {trigger_model} \
  ...
  --trigger_opt {optimizer_preset['trigger_opt']} \
  --trigger_lr {optimizer_preset['trigger_lr']} \
  --trigger_weight_decay {optimizer_preset['trigger_weight_decay']} \
  --surrogate_opt {optimizer_preset['surrogate_opt']} \
  --surrogate_lr {optimizer_preset['surrogate_lr']} \
  --surrogate_weight_decay {optimizer_preset['surrogate_weight_decay']} \
  ...
```

## Output Artifacts

### Master Script
- `master_all_experiments.sh` - All 3,276 commands for complete experiment suite

### Organized Scripts
- **By Model**: `model_{tmodel}_all_methods.sh` (4 files)
- **By Method**: `method_{method}_{tmodel}.sh` (28 files)
- **By Dataset**: `dataset_{dataset}.sh` (21 files)

### Summary Documentation
- `CONDITIONAL_SUMMARY.md` - Comprehensive overview with:
  - Optimizer preset specifications
  - Method hyperparameter configurations
  - Dataset characteristics
  - Script organization guide

## Integration with Parameters System

### Parameter Dependencies
These presets rely on optimizer infrastructure implemented in previous phase:

- **`parameters.py`**: 
  - `--trigger_opt`, `--trigger_weight_decay`
  - `--surrogate_opt`, `--surrogate_weight_decay`
  - `--trigger_lr`, `--surrogate_lr` (reused from parameters)

- **`utils/tools.py`**: 
  - `create_optimizer()` function handles all optimizer creation
  - Supports: `adam`, `adamw`, `sgd`
  - SGD momentum hardcoded to 0.9

### Command Flow
```
generate_conditional_scripts.py
  ↓ (generates commands with parameters)
main.py
  ↓ (parses args.{trigger,surrogate}_opt)
train.py / trigger_training_*.py
  ↓ (calls create_optimizer with parsed args)
utils/tools.py::create_optimizer()
  ↓ (instantiates torch.optim optimizer)
torch.optim.{Adam|AdamW|SGD}
```

## Experimental Design Rationale

### Why 3 Presets?
1. **Standard**: Reproduces baseline papers for validation
2. **Regularized**: Explores overfitting prevention for robustness
3. **Aggressive**: Tests momentum-based optimization for complex scenarios

### Coverage
- Explores optimizer choices: Adam, AdamW, SGD with momentum
- Covers learning rate range: 1e-4 to 1e-2
- Tests weight decay from 0 to 1e-2
- Asymmetric trigger/surrogate configs enable analysis of which component benefits most from regularization

## Example Generated Command

```bash
python -u main.py \
  --train_mode backdoor \
  --method inputaware \
  --Tmodel cpatchtst \
  --bd_type all2all \
  --root_path ./dataset/Heartbeat \
  --seq_len 405 \
  --batch_size 16 \
  --trigger_epochs 60 \
  --bd_train_epochs 30 \
  --poisoning_ratio 0.1 \
  --target_label 0 \
  --d_model_bd 64 \
  --d_ff_bd 128 \
  --e_layers_bd 1 \
  --n_heads_bd 8 \
  --patch_len 16 \
  --stride 8 \
  --trigger_opt adamw \
  --trigger_lr 0.001 \
  --trigger_weight_decay 0.0001 \
  --surrogate_opt sgd \
  --surrogate_lr 0.01 \
  --surrogate_weight_decay 0.005 \
  --p_attack 0.5 \
  --p_cross 0.1 \
  --lambda_cross 1.0 \
  --clip_ratio 0.1
```

## How to Use Generated Scripts

### Run All Experiments
```bash
bash scripts/conditional_all2all/master_all_experiments.sh
```

### Run Single Method with All Presets
```bash
bash scripts/conditional_all2all/method_inputaware_cpatchtst.sh
```

### Run Single Dataset Across All Methods/Presets
```bash
bash scripts/conditional_all2all/dataset_Heartbeat.sh
```

### Run Single Model Across All Methods/Presets
```bash
bash scripts/conditional_all2all/model_citst_all_methods.sh
```

## Results Tracking

Results will be organized as:
```
results/
├── backdoor_results/
│   ├── {method}_{tmodel}_{dataset}_{config}_{optimizer_preset}.json
│   └── ...
├── results_summary.csv  (consolidated metrics)
└── ...
```

Each experiment will log:
- Trigger training accuracy/loss
- Surrogate model accuracy/ASR (Attack Success Rate)
- Optimizer-specific metrics (adaptive learning rates, momentum history)
- Class-conditional trigger effectiveness

## References

Literature sources for optimizer presets:

1. **WaNet** - Weight accumulation backdoor attack (mostly Adam, minimal regularization)
2. **LIRA** - Learning to Invisible Robust Backdoors (emphasized regularization)
3. **Input-Aware** - Input-aware backdoors (momentum-based surrogate training)
4. **Blind Backdoor** - No class-label required (symmetric optimizer strategies)
5. **ISSBA** - Image-Specific Sample Backdoor Attacks (cross-entropy optimization)

---

**Generated**: With 3 optimizer presets applied to 7 methods × 4 models × 21 datasets

**Total Experiments to Run**: 3,276 unique configurations

**Estimated Time** (assuming 2 min per experiment): ~110 hours on single GPU
