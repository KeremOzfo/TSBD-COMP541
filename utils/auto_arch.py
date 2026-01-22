from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import math
from typing import Dict, Optional


@dataclass
class DatasetInfo:
    name: str
    seq_len: int
    num_variates: int
    num_classes: int
    num_train: int
    num_test: int


def load_dataset_info(csv_path: str | Path) -> Dict[str, DatasetInfo]:
    csv_path = Path(csv_path)
    if not csv_path.exists():
        return {}

    dataset_map: Dict[str, DatasetInfo] = {}
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            name = row.get("dataset")
            if not name:
                continue
            dataset_map[name] = DatasetInfo(
                name=name,
                seq_len=int(row.get("seq_len", 0)),
                num_variates=int(row.get("num_variates", 0)),
                num_classes=int(row.get("num_classes", 0)),
                num_train=int(row.get("num_train", 0)),
                num_test=int(row.get("num_test", 0)),
            )
    return dataset_map


def complexity_score(info: DatasetInfo) -> float:
    """
    Compute complexity score emphasizing seq_len as primary driver.
    Inspired by Time-Series-Library classification scripts where longer sequences
    and higher dimensionality require larger models.
    """
    # Primary: sequence length (most important for trigger modeling)
    seq_component = math.log2(max(info.seq_len, 1)) * 0.50
    
    # Secondary: num_variates (critical for multivariate series)
    var_component = math.log2(max(info.num_variates, 1) + 1) * 0.30
    
    # Tertiary: num_classes (important for classification head capacity)
    cls_component = math.log2(max(info.num_classes, 1) + 1) * 0.15
    
    # Minor: dataset size (affects batch size more than architecture)
    size_component = math.log10(max(info.num_train, 1) + 1) * 0.05
    
    return seq_component + var_component + cls_component + size_component


def choose_bd_arch(info: DatasetInfo) -> Dict[str, int | float | str]:
    """
    Choose backdoor trigger architecture based on dataset characteristics.
    
    Follows Time-Series-Library patterns:
    - TimesNet uses d_model 32-64, e_layers 2-3
    - PatchTST uses d_model 128, e_layers 3
    - Trigger models need to scale with seq_len, num_variates, num_classes
    - e_layers capped at 2 for efficiency (triggers are simpler than full models)
    """
    score = complexity_score(info)
    
    # Encoder layers: cap at 2 as requested, use 1 for simpler datasets
    # Very simple datasets (short seq, few variates/classes) use 1 layer
    if score < 7.0:
        e_layers = 1
    else:
        e_layers = 2
    
    # d_model selection based on complexity score and specific dataset characteristics
    # Range from 32 to 512 as requested
    # Adjusted thresholds based on actual dataset distribution
    
    # Extra small: very short sequences (<50), few variates (<5), few classes (<5)
    if score < 4.5:
        d_model = 32
    # Small: short sequences (<150), moderate variates/classes
    elif score < 5.5:
        d_model = 64
    # Medium-small: moderate sequences (150-400), moderate dimensionality
    elif score < 6.5:
        d_model = 128
    # Medium: longer sequences (400-800) or high variates/classes
    elif score < 7.5:
        d_model = 192
    # Medium-large: long sequences (800-1200) or very high dimensionality
    elif score < 8.5:
        d_model = 256
    # Large: very long sequences (>1200) or extremely high variates
    elif score < 9.5:
        d_model = 384
    # Extra large: extreme complexity
    else:
        d_model = 512
    
    # Ensure d_model is divisible by valid head counts
    # Prefer 4 or 8 heads for better parallelism
    candidate_heads = [2, 4, 8, 16]
    valid_heads = [h for h in candidate_heads if d_model % h == 0]
    
    # Choose largest valid head count that keeps per-head dimension reasonable (32-128)
    n_heads = 2  # fallback
    for h in reversed(valid_heads):
        if 32 <= d_model // h <= 128:
            n_heads = h
            break
    
    # d_ff typically 2x or 4x d_model in transformers
    # Time-Series-Library uses 2x for most cases
    d_ff = d_model * 2
    
    # Tier labels for logging
    if d_model <= 64:
        tier = "small"
    elif d_model <= 128:
        tier = "medium"
    elif d_model <= 256:
        tier = "large"
    else:
        tier = "xlarge"
    
    return {
        "d_model_bd": int(d_model),
        "d_ff_bd": int(d_ff),
        "e_layers_bd": int(e_layers),
        "n_heads_bd": int(n_heads),
        "complexity_score": score,
        "tier": tier,
    }


def choose_batch_size(info: DatasetInfo) -> Dict[str, int | str]:
    """
    Choose batch size based on dataset size and memory considerations.
    Time-Series-Library uses batch_size=16 for most classification tasks.
    """
    # Start with a base batch size following Time-Series-Library convention
    # Most classification scripts use 16
    bs = 16
    
    # Adjust based on dataset size
    n = info.num_train
    if n < 100:
        bs = 8  # Very small datasets
    elif n < 500:
        bs = 16
    elif n < 2000:
        bs = 32
    elif n < 5000:
        bs = 64
    else:
        bs = 128
    
    # Memory constraint: reduce for very long sequences or high dimensionality
    # Each sample's memory ~ seq_len * num_variates
    memory_footprint = info.seq_len * max(info.num_variates, 1)
    
    # Aggressive reduction for very large footprints
    if memory_footprint > 100000:  # e.g., seq_len=1000, variates=100
        bs = max(4, bs // 4)
    elif memory_footprint > 50000:   # e.g., seq_len=1000, variates=50
        bs = max(8, bs // 2)
    elif memory_footprint > 20000:   # e.g., seq_len=400, variates=50
        bs = max(8, bs // 2)
    
    # Ensure reasonable range
    bs = int(min(max(bs, 4), 256))
    
    return {"batch_size": bs}


def apply_auto_bd_arch(
    args,
    dataset_name: str,
    csv_path: str | Path,
    fallback: Optional[DatasetInfo] = None,
) -> bool:
    dataset_map = load_dataset_info(csv_path)
    info = dataset_map.get(dataset_name)
    if info is None and fallback is not None:
        info = fallback

    if info is None:
        print(f"[AutoArch] Dataset '{dataset_name}' not found in {csv_path}, skipping.")
        return False

    config = choose_bd_arch(info)
    args.d_model_bd = int(config["d_model_bd"])
    args.d_ff_bd = int(config["d_ff_bd"])
    args.e_layers_bd = int(config["e_layers_bd"])
    args.n_heads_bd = int(config["n_heads_bd"])

    print(
        "[AutoArch] Selected backdoor trigger architecture: "
        f"d_model_bd={args.d_model_bd}, d_ff_bd={args.d_ff_bd}, "
        f"e_layers_bd={args.e_layers_bd}, n_heads_bd={args.n_heads_bd} "
        f"(tier={config['tier']}, score={config['complexity_score']:.2f})"
    )
    return True


def apply_auto_batch_size(
    args,
    dataset_name: str,
    csv_path: str | Path,
    fallback: Optional[DatasetInfo] = None,
) -> bool:
    dataset_map = load_dataset_info(csv_path)
    info = dataset_map.get(dataset_name)
    if info is None and fallback is not None:
        info = fallback

    if info is None:
        print(f"[AutoArch] Dataset '{dataset_name}' not found in {csv_path}, skipping batch size.")
        return False

    config = choose_batch_size(info)
    args.batch_size = int(config["batch_size"])
    print(f"[AutoArch] Selected batch_size={args.batch_size} based on num_train={info.num_train}.")
    return True


def print_all_bd_arches(csv_path: str | Path) -> None:
    dataset_map = load_dataset_info(csv_path)
    if not dataset_map:
        print(f"[AutoArch] No datasets found in {csv_path}")
        return

    print("dataset,d_model_bd,d_ff_bd,e_layers_bd,n_heads_bd,tier,complexity_score,batch_size")
    for name in sorted(dataset_map.keys()):
        info = dataset_map[name]
        config = choose_bd_arch(info)
        bs_config = choose_batch_size(info)
        print(
            f"{name},{config['d_model_bd']},{config['d_ff_bd']},"
            f"{config['e_layers_bd']},{config['n_heads_bd']},"
            f"{config['tier']},{config['complexity_score']:.4f},"
            f"{bs_config['batch_size']}"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print auto-selected backdoor architectures.")
    parser.add_argument("--csv", type=str, default="./scripts/dataset_info.csv")
    args = parser.parse_args()
    print_all_bd_arches(args.csv)
