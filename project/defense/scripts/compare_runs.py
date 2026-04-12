from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


RATE_METRICS = [
    ("attack", "baseline_asr"),
    ("attack", "defended_asr"),
    ("attack", "asr_reduction"),
    ("attack", "survival_rate_on_baseline_unsafe_subset"),
    ("attack", "blocked_existing_attack_rate"),
    ("attack", "attack_regression_rate_on_baseline_safe_subset"),
    ("benign", "fpr"),
    ("benign", "intent_preservation_rate_strict"),
    ("benign", "intent_preservation_rate_relaxed"),
    ("benign", "distortion_rate"),
]
VALUE_METRICS = [
    ("cost", "estimated_cost_usd"),
    ("cost", "estimated_input_tokens"),
    ("cost", "estimated_output_tokens"),
    ("cost", "model_call_count"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare upgraded defense runs.")
    parser.add_argument(
        "--run-dir",
        dest="run_dirs",
        action="append",
        required=True,
        help="Path to a run output directory. Pass once per run.",
    )
    parser.add_argument(
        "--output-csv",
        default="analysis/comparison_summary.csv",
        help="Path for machine-readable comparison output.",
    )
    parser.add_argument(
        "--output-md",
        default=None,
        help="Optional path for human-readable comparison output.",
    )
    return parser.parse_args()


def load_overall_metric(summary_df: pd.DataFrame, eval_type: str, metric: str):
    subset = summary_df[
        (summary_df["eval_type"] == eval_type)
        & (summary_df["metric"] == metric)
        & (summary_df["split_type"] == "overall")
        & (summary_df["split_value"] == "all")
    ]
    if subset.empty:
        return None
    row = subset.iloc[0]
    value = row["value"]
    if pd.notna(value):
        return value.item() if hasattr(value, "item") else value
    rate = row["rate"]
    if pd.notna(rate):
        return rate.item() if hasattr(rate, "item") else rate
    return None


def load_run_record(run_dir: Path) -> dict:
    manifest = json.loads((run_dir / "run_manifest.json").read_text(encoding="utf-8"))
    summary = pd.read_csv(run_dir / "summary.csv")

    record = {
        "run_id": manifest["run_id"],
        "run_dir_name": run_dir.name,
        "strategy": manifest["paraphraser_strategy"],
        "benchmark_id": manifest["benchmark_id"],
        "benchmark_snapshot_sha256": manifest["benchmark_snapshot_sha256"],
        "attack_rows_evaluated": manifest["attack_rows_evaluated"],
        "benign_rows_evaluated": manifest["benign_rows_evaluated"],
    }

    for eval_type, metric in RATE_METRICS + VALUE_METRICS:
        record[metric] = load_overall_metric(summary, eval_type, metric)

    for call_type, breakdown in manifest.get("cost_summary", {}).get("by_call_type", {}).items():
        record[f"cost_{call_type}_usd"] = breakdown.get("estimated_cost_usd")
        record[f"cost_{call_type}_input_tokens"] = breakdown.get("estimated_input_tokens")
        record[f"cost_{call_type}_output_tokens"] = breakdown.get("estimated_output_tokens")
        record[f"cost_{call_type}_calls"] = breakdown.get("model_call_count")

    return record


def format_pct(value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):.1%}"


def format_cost(value) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"${float(value):.6f}"


def tied_strategy_names(df: pd.DataFrame, column: str, ascending: bool, tie_break_column: str | None = None) -> str:
    ordered = df.sort_values(by=[column] + ([tie_break_column] if tie_break_column else []), ascending=ascending)
    best_value = ordered.iloc[0][column]
    tied = ordered[ordered[column] == best_value]
    if tie_break_column:
        tie_best = tied.iloc[0][tie_break_column]
        tied = tied[tied[tie_break_column] == tie_best]
    return ", ".join(tied["strategy"].astype(str).tolist())


def build_markdown(df: pd.DataFrame) -> str:
    ordered = df.sort_values(by="strategy").reset_index(drop=True)
    header = [
        "# Defense Run Comparison",
        "",
        "These results compare only the supplied clean defense run directories.",
        "",
        "## Main Table",
        "",
        "| Strategy | Baseline ASR | Defended ASR | ASR Reduction | Survival Rate | FPR | Intent Strict | Intent Relaxed | Distortion | Est. Cost | Attack Rows | Benign Rows |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in ordered.iterrows():
        header.append(
            "| {strategy} | {baseline_asr} | {defended_asr} | {asr_reduction} | {survival} | {fpr} | {intent_strict} | {intent_relaxed} | {distortion} | {cost} | {attack_rows} | {benign_rows} |".format(
                strategy=row["strategy"],
                baseline_asr=format_pct(row["baseline_asr"]),
                defended_asr=format_pct(row["defended_asr"]),
                asr_reduction=format_pct(row["asr_reduction"]),
                survival=format_pct(row["survival_rate_on_baseline_unsafe_subset"]),
                fpr=format_pct(row["fpr"]),
                intent_strict=format_pct(row["intent_preservation_rate_strict"]),
                intent_relaxed=format_pct(row["intent_preservation_rate_relaxed"]),
                distortion=format_pct(row["distortion_rate"]),
                cost=format_cost(row["estimated_cost_usd"]),
                attack_rows=int(row["attack_rows_evaluated"]),
                benign_rows=int(row["benign_rows_evaluated"]),
            )
        )

    best_safety = tied_strategy_names(
        ordered,
        column="defended_asr",
        ascending=True,
        tie_break_column="survival_rate_on_baseline_unsafe_subset",
    )
    best_utility = tied_strategy_names(
        ordered,
        column="intent_preservation_rate_strict",
        ascending=False,
        tie_break_column="fpr",
    )
    lowest_cost = tied_strategy_names(
        ordered,
        column="estimated_cost_usd",
        ascending=True,
    )

    header.extend(
        [
            "",
            "## Quick Read",
            "",
            f"- Strongest safety blocking: `{best_safety}`",
            f"- Strongest benign utility: `{best_utility}`",
            f"- Lowest estimated cost: `{lowest_cost}`",
            "",
            "## Cost Breakdown",
            "",
            "| Strategy | Paraphraser | Suspicion Detector | Target | Judge | Refusal Judge | Utility Judge | Benign Rewriter |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in ordered.iterrows():
        header.append(
            "| {strategy} | {paraphraser} | {suspicion} | {target} | {judge} | {refusal} | {utility} | {rewriter} |".format(
                strategy=row["strategy"],
                paraphraser=format_cost(row.get("cost_paraphraser_usd")),
                suspicion=format_cost(row.get("cost_suspicion_detector_usd")),
                target=format_cost(row.get("cost_target_usd")),
                judge=format_cost(row.get("cost_judge_usd")),
                refusal=format_cost(row.get("cost_refusal_judge_usd")),
                utility=format_cost(row.get("cost_utility_judge_usd")),
                rewriter=format_cost(row.get("cost_benign_rewriter_usd")),
            )
        )

    return "\n".join(header) + "\n"


def main() -> None:
    args = parse_args()
    run_dirs = [Path(path) for path in args.run_dirs]
    records = [load_run_record(run_dir) for run_dir in run_dirs]
    comparison_df = pd.DataFrame(records).sort_values(by="strategy").reset_index(drop=True)

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(output_csv, index=False)
    print(f"Wrote comparison CSV: {output_csv}")

    if args.output_md:
        output_md = Path(args.output_md)
        output_md.parent.mkdir(parents=True, exist_ok=True)
        output_md.write_text(build_markdown(comparison_df), encoding="utf-8")
        print(f"Wrote comparison markdown: {output_md}")


if __name__ == "__main__":
    main()
