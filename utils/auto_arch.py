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
    # A simple, monotonic complexity proxy
    return (
        math.log2(max(info.seq_len, 1))
        + math.log2(max(info.num_variates, 1) + 1)
        + math.log2(max(info.num_classes, 1) + 1)
        + math.log10(max(info.num_train, 1) + 1)
    )


def choose_bd_arch(info: DatasetInfo) -> Dict[str, int | float | str]:
    score = complexity_score(info)

    # Heuristic tiers (small/medium/large/very large)
    # Relaxed thresholds to reduce xlarge selections
    if score < 10:
        d_model, n_heads, e_layers = 32, 2, 1
        tier = "small"
    elif score < 14.0:
        d_model, n_heads, e_layers = 64, 4, 2
        tier = "medium"
    elif score < 18.0:
        d_model, n_heads, e_layers = 96, 4, 3
        tier = "large"
    else:
        d_model, n_heads, e_layers = 128, 8, 3
        tier = "xlarge"

    # Ensure divisibility
    if d_model % n_heads != 0:
        d_model = d_model - (d_model % n_heads)

    d_ff = d_model * 2

    return {
        "d_model_bd": d_model,
        "d_ff_bd": d_ff,
        "e_layers_bd": e_layers,
        "n_heads_bd": n_heads,
        "complexity_score": score,
        "tier": tier,
    }


def choose_batch_size(info: DatasetInfo) -> Dict[str, int | str]:
    # Primary driver: number of training samples
    n = info.num_train
    if n < 200:
        bs = 16
    elif n < 500:
        bs = 32
    elif n < 1000:
        bs = 64
    elif n < 2000:
        bs = 128
    elif n < 5000:
        bs = 128
    else:
        bs = 256

    # Adjust for very large sequences/variates
    seq_var = info.seq_len * max(info.num_variates, 1)
    if seq_var > 50000:
        bs = max(4, bs // 2)
    if seq_var > 150000:
        bs = max(4, bs // 2)

    # Clamp to a reasonable range
    bs = int(min(max(bs, 4), 512))
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
