# Rigorous Exploration: JapaneseVowels

## Dataset Info

- **Sequence Length**: 26
- **Master script (calls all methods)**: `run_all_methods.sh`
- **Combined script (concatenated)**: `JapaneseVowels_all_methods.sh`

## How to Run

### Run All Experiments (Sequential by Method)
```bash
bash run_all_methods.sh
```

### Run All Experiments (Concatenated

## Experiments

**Total Experiments**: 378

### By Method

- **vanilla**: 54 experiments → `vanilla.sh`
- **marksman**: 216 experiments → `marksman.sh`
- **pureinputaware**: 54 experiments → `pureinputaware.sh`
- **frequency**: 54 experiments → `frequency.sh`

## Available Scripts

- **Individual method scripts**: `vanilla.sh, marksman.sh, pureinputaware.sh, frequency.sh`
- **All methods (concatenated)**: `run_all_methods.sh` or `JapaneseVowels_all_methods.sh` (same file)

## How to Run

### Run All Experiments (All Methods)
```bash
bash run_all_methods.sh
# OR
bash JapaneseVowels_all_methods.sh
```

### Run Specific Method
```bash
bash vanilla.sh      # or marksman.sh, pureinputaware.sh
```

### Run Subset
```bash
# First 100 experiments only
head -200 JapaneseVowels_all_methods.sh > JapaneseVowels_subset.sh
bash JapaneseVowels_subset.sh
```
