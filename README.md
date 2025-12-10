# TimeSeries-Backdoor

End-to-end code for training, attacking, and evaluating time-series classification models under backdoor threats. Supports clean training, basic patch backdoors, and dynamic trigger generators with surrogate training, across multiple architectures (TimesNet, LSTM, PatchTST, iTransformer, TimeMixer, CNN-based backdoor nets).

Related papers:
[Backdoor Attacks on Time Series: A Generative Approach](https://arxiv.org/abs/2211.07915)
[Robust Multivariate Time-Series Forecasting: Adversarial Attacks and Defense Mechanisms](https://openreview.net/forum?id=ctmLBs8lITa)

## What this repository does
- Loads UEA-format time-series datasets (default: UWaveGestureLibrary) and builds data loaders with padding masks.
- Trains baseline (clean) classifiers.
- Injects backdoors via learned trigger network (`--Tmodel` choices for dynamic triggers).
- Trains a surrogate + trigger pair to craft effective triggers (trigger training phase), then poisons data, then trains the victim model on poisoned data (model poisoning phase).
- Evaluates both utility (clean accuracy) and attack success rate (ASR) and saves plots for metrics and sample traces comparing clean vs triggered inputs.
- Logs every run with arguments, final metrics, curves, and example plots under `Results/` (manifest included).

## Quick start

```bash
# Install
pip install -r requirements.txt

# Clean training (TimesNet)
python main.py --mode clean --model TimesNet --train_epochs 20

# Basic patch backdoor
python main.py --mode basic --model TimesNet --bd_train_epochs 20 --target_label 0 --poisoning_ratio 0.1 --clip_ratio 0.15

# Trigger-based backdoor with surrogate/trigger training
python main.py --mode basic --Tmodel cnn --bd_train_epochs 20 --target_label 0 --poisoning_ratio 0.1 --surrogate_type timesnet --trigger_epochs 5
```

Outputs land in `Results/<dataset>_G-<trigger>_C-<model>_<hash>/` with:
- `args_and_results.txt`: run hash, timestamp, all args, final CA/ASR
- `trigger_loss.png`, `trigger_accuracy.png`: curves for trigger+surrogate training (if used)
- `poison_metrics.png`: CA/ASR across poisoning epochs
- `examples/`: overlay plots of clean vs triggered signals (success/failure) plus `example_plots.txt` manifest with counts and paths

## How the pipeline works
1) **Data**: `data_provider/` builds train/test splits with padding masks for variable-length series.
2) **Clean train (optional)**: `clean_train_epoch` / `clean_test_epoch` in `epoch.py` train a clean classifier.
3) **Trigger training (optional)**: `trigger_train_epoch` trains a trigger generator + surrogate classifier to maximize ASR while keeping surrogate clean accuracy.
4) **Poisoning**: `poison_data` applies either a basic patch or a learned trigger to a fraction of training samples, relabeling them to the target label.
5) **Model poisoning**: `poison_model` trains the victim classifier on the poisoned set.
6) **Evaluation**: `bd_test_with_samples` computes loss, clean accuracy, ASR, and collects example traces. `log_all` saves metrics/plots/manifests.

## Key scripts and entry points
- `main.py`: orchestrates clean training or backdoor pipeline (trigger training → data poisoning → model poisoning → logging).
- `scripts/train_clean.sh`, `scripts/train_backdoor.sh`: shell shortcuts for common runs.
- `epoch.py`: core training/eval loops (clean, backdoor, trigger training, bd_test_with_samples for plotting samples).
- `utils/logging.py`: unified logging, plotting, and manifest creation.
- `utils/plot.py`: overlays clean vs triggered inputs and saves success/failure examples.

## Important arguments (see `parameters.py` for full list)
- `--mode`: `clean`, `basic` (single loss concurent update for T and C), `dynamic` (placeholder for advanced trigger training apporaches.)
- `--model`: `TimesNet`, `LSTM`, `PatchTST`, `iTransformer`, `TimeMixer`
- `--Tmodel`: trigger net choice (`cnn`, `timesnet`, `patchtst`, `itst`) for backdoor mode
- `--bd_train_epochs`, `--train_epochs`: poisoning vs clean training epochs
- `--target_label`: label the backdoor forces
- `--poisoning_ratio`: fraction of training samples to poison
- `--clip_ratio`: magnitude for basic patch trigger
- `--surrogate_type`, `--trigger_epochs`, `--warmup_epochs`, `--surrogate_lr`: surrogate/trigger training knobs
- `--batch_size`, `--lr`, `--gpu_id`, `--root_path`: standard training settings

## Project structure (abridged)
```
TimeSeries-Backdoor/
├── data_provider/      # Data loading and collation
├── dataset/            # UWaveGestureLibrary sample dataset
├── layers/             # Building blocks for models
├── models/             # Target and trigger architectures
├── utils/              # logging, plotting, masking, tools
├── scripts/            # helper shell scripts
├── results/            # run outputs (created per run)
├── epoch.py            # train/test loops and bd helpers
├── main.py             # entry point
├── parameters.py       # CLI args
└── README.md
```

## Repro tips
- Set `--seed` (add manually or set in code) and fix `num_workers` for determinism.
```



