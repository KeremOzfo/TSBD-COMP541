"""Aggregate experiment results into a single Excel file.

Scans run folders (e.g., Results/IPC/**/args_and_results.txt), extracts
clean accuracy (CA), attack success rate (ASR), run hash, and user-specified
argument values, and writes a single Excel table. Optionally aggregates
duplicate parameter settings by taking the mean CA/ASR.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd


def parse_args_file(path: Path) -> Dict[str, object]:
	"""Parse a single args_and_results.txt file.

	Returns a dict with keys: run_hash, CA, ASR, args (dict of parameters).
	"""
	data: Dict[str, object] = {"run_hash": None, "CA": None, "ASR": None, "args": {}}
	section: Optional[str] = None

	with path.open("r", encoding="utf-8") as f:
		for raw_line in f:
			line = raw_line.strip()
			if not line:
				continue

			if line.lower().startswith("run hash:"):
				data["run_hash"] = line.split(":", 1)[1].strip()
				continue

			if line.startswith("[Args]"):
				section = "args"
				continue
			if line.startswith("[Final Metrics]"):
				section = "metrics"
				continue

			if section == "args" and ":" in line:
				k, v = line.split(":", 1)
				data["args"][k.strip()] = v.strip()
			elif section == "metrics":
				if "ASR" in line:
					val = _extract_float(line)
					data["ASR"] = val if val is not None else data["ASR"]
				if "CA" in line or "Clean Accuracy" in line:
					val = _extract_float(line)
					data["CA"] = val if val is not None else data["CA"]

	return data


def _extract_float(text: str) -> Optional[float]:
	match = re.search(r"[-+]?[0-9]*\.?[0-9]+", text)
	if match:
		try:
			return float(match.group(0))
		except ValueError:
			return None
	return None


def collect_results(root: Path, param_keys: List[str]) -> pd.DataFrame:
	"""Traverse root for args_and_results.txt files and collect records."""
	records = []
	for args_file in root.rglob("args_and_results.txt"):
		parsed = parse_args_file(args_file)
		args = parsed.get("args", {})

		record = {
			"run_hash": parsed.get("run_hash"),
			"CA": parsed.get("CA"),
			"ASR": parsed.get("ASR"),
			"exp_dir": str(args_file.parent),
		}

		for key in param_keys:
			val = args.get(key)
			# Special handling for root_path: only keep last path segment
			if key == "root_path" and val is not None:
				val = str(val).replace("\\", "/").rstrip("/").split("/")[-1]
			record[key] = val

		records.append(record)

	if not records:
		return pd.DataFrame(columns=["run_hash", "CA", "ASR", *param_keys, "exp_dir"])

	df = pd.DataFrame.from_records(records)
	# Ensure numeric columns are floats
	df["CA"] = pd.to_numeric(df["CA"], errors="coerce")
	df["ASR"] = pd.to_numeric(df["ASR"], errors="coerce")
	return df


def aggregate_results(df: pd.DataFrame, param_keys: List[str]) -> pd.DataFrame:
	"""Group by parameter keys and average CA/ASR; concatenate run hashes."""
	if df.empty:
		return df

	group_cols = [k for k in param_keys if k in df.columns]
	if not group_cols:
		return df

	agg_df = (
		df.groupby(group_cols, dropna=False)
		.agg({
			"CA": "mean",
			"ASR": "mean",
			"run_hash": lambda x: ",".join(sorted(set(str(v) for v in x if pd.notna(v)))),
		})
		.reset_index()
	)

	# Preserve column order: params, CA, ASR, run_hash
	cols = group_cols + ["CA", "ASR", "run_hash"]
	return agg_df[cols]


def save_to_excel(df: pd.DataFrame, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_excel(output_path, index=False)


def main(root: Path, params: List[str], aggregate: bool, output: Path) -> None:
	df = collect_results(root, params)
	if aggregate:
		df_out = aggregate_results(df, params)
	else:
		cols = [k for k in params if k in df.columns] + ["CA", "ASR", "run_hash", "exp_dir"]
		df_out = df[cols]

	save_to_excel(df_out, output)
	print(f"Saved {len(df_out)} rows to {output}")


if __name__ == "__main__":
	# Configure here instead of command-line arguments
	ROOT = Path("../results/IPC2")
	PARAMS = ["model", "Tmodel",'surrogate_model', "mode",'root_path']
	PARAMS = ["model", "Tmodel", "mode",'root_path']
	AGGREGATE = True  # True to average duplicates by PARAMS
	OUTPUT = Path("results_summary.xlsx")

	main(root=ROOT, params=PARAMS, aggregate=AGGREGATE, output=OUTPUT)
