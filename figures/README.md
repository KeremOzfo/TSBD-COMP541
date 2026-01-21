# Figures Directory

This directory contains all visualization figures for the TSBD-COMP541 project on backdoor attacks in time series classification.

## Directory Structure

```
figures/
├── README.md (Master overview)
│
├── conditioning_mechanisms/    # Conditional trigger generation mechanisms
│   ├── 01_conditioning_mechanisms_diagram.png
│   ├── 02_conditioning_comparison_table.png
│   ├── 03_conditioning_flow_diagram.png
│   └── README.md
│
├── backdoor_pipeline/         # Complete backdoor attack pipeline
│   ├── 01_backdoor_attack_pipeline.png
│   ├── 02_backdoor_attack_detailed.png
│   ├── 03_backdoor_attack_example.png
│   └── README.md
│
├── inputaware_method/         # Pure input-aware method details
│   ├── 01_pure_inputaware_method.png
│   ├── 02_inputaware_comparison.png
│   ├── 03_diversity_loss_mechanism.png
│   └── README.md
│
└── system_design/             # Automatic system design features
    ├── 01_auto_architecture_scaling.png
    ├── 02_optimizer_presets.png
    └── README.md
```

## Figure Categories

### 1. Conditioning Mechanisms (`conditioning_mechanisms/`)
Visualizations explaining how different neural network architectures implement class-conditional trigger generation:
- **CNN-based**: Late-stage additive conditioning
- **PatchTST**: Early patch-level conditioning
- **TimesNet**: Multi-stage iterative conditioning
- **Inverted Transformer**: Variate-level conditioning
- **CNN Autoencoder**: Input-level concatenation

**Use for:** Understanding model architectures, comparing conditioning strategies

### 2. Backdoor Attack Pipeline (`backdoor_pipeline/`)
Comprehensive visualizations of the complete backdoor attack workflow:
- **Stage 1**: Trigger training (generator + surrogate)
- **Stage 2**: Data poisoning (trigger application)
- **Stage 3**: Classifier poisoning (victim training)
- **Stage 4**: Attack demonstration (clean vs backdoored)

**Use for:** Explaining attack methodology, threat model, experimental setup

### 3. Input-Aware Method (`inputaware_method/`)
Detailed visualizations of the pure input-aware backdoor attack method:
- **Method overview**: Three-mode batching with mathematical formulations
- **Implementation comparison**: Original VinAI vs our approximation
- **Diversity loss**: Mechanism for enforcing trigger variation

**Use for:** Understanding input-aware attacks, diversity enforcement, implementation details

### 4. System Design (`system_design/`)
Automatic system design features for trigger model training:
- **Architecture scaling**: Complexity-based model configuration
- **Optimizer presets**: Literature-based optimization strategies

**Use for:** Understanding automatic hyperparameter selection, optimizer configurations

## Usage Guidelines

### For Research Papers
- Use high-resolution PNG files (all figures are 600+ KB)
- Cite the TSBD-COMP541 project
- Reference specific figure numbers in captions

### For Presentations
- Figures are designed for both light and dark backgrounds
- Use the pipeline figures for methodology slides
- Use the conditioning figures for architecture slides

### For Documentation
- Each subdirectory has a detailed README
- Figures include mathematical formulations and annotations
- Color coding is consistent across all visualizations

## Color Coding Standards

Consistent color scheme across all figures:
- **Blue**: Clean data, normal operations
- **Red**: Backdoor/attack elements, malicious operations
- **Green**: Correct predictions, successful operations
- **Orange**: Poisoning phase, data modification
- **Purple**: Concatenation operations
- **Gray**: Neutral elements, infrastructure

## Figure Quality

- **Format**: PNG (lossless compression)
- **Resolution**: High-resolution for publication
- **Size**: 600-800 KB per figure
- **Style**: Professional academic style
- **Annotations**: Clear labels, legends, and mathematical notation

## Generation Information

- **Generated**: January 19, 2026
- **Tool**: AI-assisted visualization generation
- **Purpose**: Research documentation and presentation

## Related Project Files

- **Models**: `src/models/trigger_models/`
- **Training**: `train.py`
- **Data Poisoning**: `src/data_poisoning.py`
- **Testing**: `test.py`
- **Documentation**: `src/models/trigger_models/conditional_conditioning_overview.md`

## Future Additions

Planned visualizations:
- Defense mechanisms comparison
- Attack success rate vs poisoning ratio
- Trigger magnitude vs stealthiness trade-off
- Multi-target attack scenarios
- Latent space analysis

---

For detailed information about specific figures, see the README files in each subdirectory.
