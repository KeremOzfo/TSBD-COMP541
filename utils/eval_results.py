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

		# Extract info from folder name: {Dataset}_G-{Tmodel}_C-{model}_{hash}
		folder_name = args_file.parent.name
		folder_parts = {}
		try:
			parts = folder_name.split("_")
			# Find the part with G- and C-
			for i, part in enumerate(parts):
				if part.startswith("G-"):
					folder_parts["folder_Tmodel"] = part[2:]  # Remove "G-" prefix
				elif part.startswith("C-"):
					folder_parts["folder_model"] = part[2:]  # Remove "C-" prefix
			# Dataset is everything before _G-
			if "_G-" in folder_name:
				folder_parts["folder_dataset"] = folder_name.split("_G-")[0]
		except:
			pass

		record = {
			"run_hash": parsed.get("run_hash"),
			"CA": parsed.get("CA"),
			"ASR": parsed.get("ASR"),
			"exp_dir": str(args_file.parent),
			**folder_parts,  # Add folder-extracted info
		}

		for key in param_keys:
			val = args.get(key)
			# Special handling for root_path: only keep last path segment
			if key == "root_path" and val is not None:
				val = str(val).replace("\\", "/").rstrip("/").split("/")[-1]
			record[key] = val

		records.append(record)

	if not records:
		return pd.DataFrame(columns=["run_hash", "CA", "ASR", "folder_dataset", "folder_Tmodel", "folder_model", *param_keys, "exp_dir"])

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


def best_by_score_results(df: pd.DataFrame, param_keys: List[str]) -> pd.DataFrame:
	"""Group by parameter keys and select best run by (CA+ASR) score."""
	if df.empty:
		return df

	group_cols = [k for k in param_keys if k in df.columns]
	if not group_cols:
		return df

	# Calculate combined score
	df = df.copy()
	df['combined_score'] = df['CA'] + df['ASR']
	
	# For each group, select the row with max combined_score
	best_df = (
		df.sort_values('combined_score', ascending=False)
		.groupby(group_cols, dropna=False)
		.first()
		.reset_index()
	)
	
	# Preserve column order: params, CA, ASR, combined_score, run_hash, exp_dir
	cols = group_cols + ["CA", "ASR", "combined_score", "run_hash", "exp_dir"]
	# Only include columns that exist
	cols = [c for c in cols if c in best_df.columns]
	
	return best_df[cols]


def top_n_aggregate_results(df: pd.DataFrame, param_keys: List[str], n: int) -> pd.DataFrame:
	"""Group by parameter keys, select top N runs by (CA+ASR), and average their CA/ASR."""
	if df.empty:
		return df

	group_cols = [k for k in param_keys if k in df.columns]
	if not group_cols:
		return df

	# Calculate combined score
	df = df.copy()
	df['combined_score'] = df['CA'] + df['ASR']
	
	# For each group, select top N rows by combined_score and aggregate
	def aggregate_top_n(group):
		# Sort by score and take top N (or all if less than N)
		top_runs = group.nlargest(min(n, len(group)), 'combined_score')
		
		return pd.Series({
			'CA': top_runs['CA'].mean(),
			'ASR': top_runs['ASR'].mean(),
			'n_runs': len(top_runs),
			'run_hash': ','.join(sorted(set(str(v) for v in top_runs['run_hash'] if pd.notna(v))))
		})
	
	agg_df = df.groupby(group_cols, dropna=False).apply(aggregate_top_n).reset_index()
	
	# Preserve column order: params, CA, ASR, n_runs, run_hash
	cols = group_cols + ["CA", "ASR", "n_runs", "run_hash"]
	return agg_df[cols]


def save_to_excel(df: pd.DataFrame, output_path: Path) -> None:
	output_path.parent.mkdir(parents=True, exist_ok=True)
	df.to_excel(output_path, index=False)


def main(root: Path, params: List[str], aggregate: bool, best_by_score: bool, top_n_aggregate: int, output: Path) -> None:
	"""Main function to collect and process results.
	
	Args:
		root: Root directory to scan for results
		params: List of parameter keys to extract
		aggregate: If True, group by params and average CA/ASR (mutually exclusive with others)
		best_by_score: If True, group by params and select best run by (CA+ASR) (mutually exclusive with others)
		top_n_aggregate: If > 0, group by params and average top N runs by (CA+ASR) (mutually exclusive with others)
		output: Output Excel file path
	"""
	# Check mutual exclusivity
	active_modes = sum([aggregate, best_by_score, top_n_aggregate > 0])
	if active_modes > 1:
		raise ValueError("aggregate, best_by_score, and top_n_aggregate are mutually exclusive")
	
	df = collect_results(root, params)
	
	if aggregate:
		df_out = aggregate_results(df, params)
	elif best_by_score:
		df_out = best_by_score_results(df, params)
	elif top_n_aggregate > 0:
		df_out = top_n_aggregate_results(df, params, top_n_aggregate)
	else:
		# Put folder-extracted columns first for verification
		folder_cols = ["folder_dataset", "folder_Tmodel", "folder_model"]
		available_folder_cols = [c for c in folder_cols if c in df.columns]
		param_cols = [k for k in params if k in df.columns]
		cols = available_folder_cols + param_cols + ["CA", "ASR", "run_hash", "exp_dir"]
		df_out = df[cols]

	save_to_excel(df_out, output)
	print(f"Saved {len(df_out)} rows to {output}")
	
	if best_by_score:
		print(f"Mode: Best by score (CA+ASR) - selected best run per parameter combination")
	elif aggregate:
		print(f"Mode: Aggregate - averaged CA/ASR per parameter combination")
	elif top_n_aggregate > 0:
		print(f"Mode: Top-{top_n_aggregate} Aggregate - averaged top {top_n_aggregate} runs per parameter combination")
	else:
		print(f"Mode: Raw - all individual experiments")


if __name__ == "__main__":
	# Configure here instead of command-line arguments
	ROOT = Path("Results/all2all")  # Root directory to scan
	
	# Base parameters
	PARAMS = ["model", "Tmodel", "root_path", "method", "bd_type"]
	
	# Architecture parameters (from ARCHITECTURE_CONFIGS in generate_rigorous_trigger_scripts.py)
	#PARAMS += ["d_model_bd", "d_ff_bd", "e_layers_bd", "n_heads_bd"]
	
	# Optimizer parameters (from OPTIMIZER_CONFIGS in generate_rigorous_trigger_scripts.py)
	#PARAMS += ["trigger_opt", "trigger_lr", "trigger_weight_decay"]
	#PARAMS += ["surrogate_opt", "surrogate_lr", "surrogate_weight_decay", "surrogate_L2_penalty"]
	
	# Training dynamics
	#PARAMS += ["warmup_epochs"]
	#PARAMS += ["marksman_alpha", "marksman_beta", "marksman_update_T"]
	#PARAMS += ["lambda_cross", "p_attack", "lambda_div"]
	#PARAMS += ["lambda_freq", "freq_lambda"]
	#PARAMS += ["surrogate_grad_clip", "trigger_grad_clip"]	
	# Poisoning strategy (from POISONING_CONFIGS in generate_rigorous_trigger_scripts.py)
	#PARAMS += ["poisoning_ratio", "use_silent_poisoning", "lambda_ratio"]
	
	# Method-specific hyperparameters (from METHOD_HYPERPARAMS in generate_rigorous_trigger_scripts.py)
	# Marksman
	#PARAMS += ["marksman_alpha", "marksman_beta", "marksman_update_T"]
	# Diversity
	#PARAMS += ["div_reg"]
	# Frequency
	#PARAMS += ["lambda_freq", "freq_lambda"]
	# Input-aware
	#PARAMS += ["p_attack", "p_cross", "lambda_cross"]
	# Pure input-aware
	#PARAMS += ["lambda_div"]
	# Ultimate (combines multiple)
	#PARAMS += ["lambda_reg"]
	
	AGGREGATE = False  # True to average duplicates by PARAMS
	BEST_BY_SCORE = False  # True to select best run by (CA+ASR) per PARAMS combination
	TOP_N_AGGREGATE = 3  # If > 0, average top N runs by (CA+ASR) per PARAMS combination
	# Note: AGGREGATE, BEST_BY_SCORE, and TOP_N_AGGREGATE are mutually exclusive
	
	OUTPUT = Path("results_summary.xlsx")

	main(root=ROOT, params=PARAMS, aggregate=AGGREGATE, best_by_score=BEST_BY_SCORE, top_n_aggregate=TOP_N_AGGREGATE, output=OUTPUT)
