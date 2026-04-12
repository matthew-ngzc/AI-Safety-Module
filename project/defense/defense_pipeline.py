"""
CS427 AI Safety Project - Paraphrasing Defense Pipeline

Implements the project-aligned paraphrase preprocessing defense and evaluates:
  - defended ASR on a frozen attack benchmark snapshot
  - FPR and benign intent retention on benign creative prompts
"""

import argparse
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from benchmarks import (
    DEFAULT_DEFENSE_BENCHMARK_ID,
    resolve_attack_run_snapshot,
    resolve_benchmark_snapshot,
)
from evaluation import (
    build_summary_metrics,
    evaluate_attack_rows,
    evaluate_benign_rows,
    load_attack_results,
    load_or_generate_benign_prompts,
    write_run_manifest,
)
from pricing import PRICING_SOURCE, TOKEN_ESTIMATION_METHOD
from prompts import (
    ATTACK_STYLES,
    BASELINE_SYSTEM_PROMPT,
    INTENT_RETENTION_JUDGE_SYSTEM_PROMPT,
    JUDGE_SYSTEM_PROMPT,
    PARAPHRASER_PROMPTS,
    PARAPHRASER_STRATEGIES,
    REFUSAL_JUDGE_SYSTEM_PROMPT,
    SUSPICION_DETECTOR_SYSTEM_PROMPT,
    SUSPICIOUS_INTENT_GUARDED_VARIANT_PROMPTS,
)


load_dotenv(dotenv_path=Path(__file__).with_name(".env"))

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "").strip()
DEFENSE_NAME = "paraphrasing"
TARGET_MODEL_NAME = "gemini-2.5-flash"
PARAPHRASER_MODEL_NAME = "deepseek-chat"
JUDGE_MODEL_NAME = "gemini-2.5-flash"


def parse_args():
    parser = argparse.ArgumentParser(description="CS427 Paraphrasing Defense Evaluation")
    parser.add_argument(
        "--benchmark-id",
        default=DEFAULT_DEFENSE_BENCHMARK_ID,
        help="Benchmark snapshot id under benchmarks/ used for defense evaluation.",
    )
    parser.add_argument(
        "--attack-run-dir",
        default=None,
        help=(
            "Optional external attack_history run directory containing results_*.csv files. "
            "When set, the defense pipeline uses those frozen rewritten attack prompts instead of a local benchmark snapshot."
        ),
    )
    parser.add_argument(
        "--style",
        choices=ATTACK_STYLES,
        default=None,
        help="Restrict evaluation to one attack style.",
    )
    parser.add_argument(
        "--skip-fpr",
        action="store_true",
        help="Skip benign prompt evaluation.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory where run outputs are written.",
    )
    parser.add_argument(
        "--run-name",
        default=None,
        help="Optional suffix added to the run id.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional row limit for quick validation runs.",
    )
    parser.add_argument(
        "--regen-benign-cache",
        action="store_true",
        help="Regenerate the benign creative prompt cache.",
    )
    parser.add_argument(
        "--paraphraser-strategy",
        choices=PARAPHRASER_STRATEGIES,
        default="baseline",
        help="Named paraphraser prompt strategy.",
    )
    return parser.parse_args()


def build_deepseek_client():
    try:
        from openai import OpenAI
    except ImportError:
        print("ERROR: 'openai' package not installed. Run: pip install openai")
        sys.exit(1)
    return OpenAI(api_key=DEEPSEEK_API_KEY, base_url="https://api.deepseek.com")


def build_gemini_model(system_instruction: str):
    try:
        import google.generativeai as genai
    except ImportError:
        print("ERROR: 'google-generativeai' package not installed.")
        sys.exit(1)

    genai.configure(api_key=GEMINI_API_KEY)
    return genai.GenerativeModel(
        model_name=TARGET_MODEL_NAME,
        system_instruction=system_instruction,
    )


def build_gemini_model_no_filters(system_instruction: str):
    try:
        import google.generativeai as genai
        from google.generativeai.types import HarmBlockThreshold, HarmCategory
    except ImportError:
        print("ERROR: 'google-generativeai' package not installed.")
        sys.exit(1)

    genai.configure(api_key=GEMINI_API_KEY)
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }
    return genai.GenerativeModel(
        model_name=JUDGE_MODEL_NAME,
        system_instruction=system_instruction,
        safety_settings=safety_settings,
    )


def build_models(paraphraser_strategy: str):
    paraphraser_client = build_deepseek_client()
    return {
        "target": build_gemini_model(BASELINE_SYSTEM_PROMPT),
        "judge": build_gemini_model_no_filters(JUDGE_SYSTEM_PROMPT),
        "refusal_judge": build_gemini_model_no_filters(REFUSAL_JUDGE_SYSTEM_PROMPT),
        "utility_judge": build_gemini_model_no_filters(INTENT_RETENTION_JUDGE_SYSTEM_PROMPT),
        "suspicion_detector": build_gemini_model_no_filters(SUSPICION_DETECTOR_SYSTEM_PROMPT),
        "paraphraser": paraphraser_client,
        "rewriter": paraphraser_client,
        "paraphraser_strategy": paraphraser_strategy,
        "paraphraser_system_prompts": PARAPHRASER_PROMPTS,
        "suspicious_intent_guarded_prompts": SUSPICIOUS_INTENT_GUARDED_VARIANT_PROMPTS,
        "target_system_prompt": BASELINE_SYSTEM_PROMPT,
        "judge_system_prompt": JUDGE_SYSTEM_PROMPT,
        "refusal_judge_system_prompt": REFUSAL_JUDGE_SYSTEM_PROMPT,
        "utility_judge_system_prompt": INTENT_RETENTION_JUDGE_SYSTEM_PROMPT,
        "suspicion_detector_system_prompt": SUSPICION_DETECTOR_SYSTEM_PROMPT,
        "target_model_name": TARGET_MODEL_NAME,
        "paraphraser_model_name": PARAPHRASER_MODEL_NAME,
        "rewriter_model_name": PARAPHRASER_MODEL_NAME,
        "judge_model_name": JUDGE_MODEL_NAME,
        "refusal_judge_model_name": JUDGE_MODEL_NAME,
        "utility_judge_model_name": JUDGE_MODEL_NAME,
        "suspicion_detector_model_name": JUDGE_MODEL_NAME,
    }


def build_run_id(defense_name: str, run_name: str | None) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = run_name or defense_name
    suffix = re.sub(r"[^A-Za-z0-9_-]+", "_", suffix).strip("_")
    return f"{timestamp}_{suffix}"


def ensure_output_dir(base_dir: str, run_id: str) -> Path:
    run_dir = Path(base_dir) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def resolve_overall_rate(summary_df, eval_type: str, metric: str):
    subset = summary_df[
        (summary_df["eval_type"] == eval_type)
        & (summary_df["metric"] == metric)
        & (summary_df["split_type"] == "overall")
        & (summary_df["split_value"] == "all")
    ]
    if subset.empty:
        return None
    return subset.iloc[0]["rate"]


def resolve_overall_value(summary_df, eval_type: str, metric: str):
    subset = summary_df[
        (summary_df["eval_type"] == eval_type)
        & (summary_df["metric"] == metric)
        & (summary_df["split_type"] == "overall")
        & (summary_df["split_value"] == "all")
    ]
    if subset.empty:
        return None
    return subset.iloc[0]["value"]


def format_rate(value) -> str:
    if value is None:
        return "n/a"
    try:
        if value != value:
            return "n/a"
    except TypeError:
        return "n/a"
    return f"{value:.1%}"


def format_currency(value) -> str:
    if value is None:
        return "n/a"
    try:
        if value != value:
            return "n/a"
    except TypeError:
        return "n/a"
    return f"${value:.6f}"


def format_count(value) -> str:
    if value is None:
        return "n/a"
    try:
        if value != value:
            return "n/a"
    except TypeError:
        return "n/a"
    return str(int(value))


def summarize_costs(model_calls_df: pd.DataFrame | None) -> dict:
    if model_calls_df is None or model_calls_df.empty:
        return {
            "overall": {
                "estimated_cost_usd": 0.0,
                "estimated_input_tokens": 0,
                "estimated_output_tokens": 0,
                "model_call_count": 0,
            },
            "by_call_type": {},
            "by_dataset_split": {},
        }

    overall = {
        "estimated_cost_usd": float(model_calls_df["est_cost_usd"].sum()),
        "estimated_input_tokens": int(model_calls_df["est_input_tokens"].sum()),
        "estimated_output_tokens": int(model_calls_df["est_output_tokens"].sum()),
        "model_call_count": int(len(model_calls_df)),
    }

    by_call_type = {}
    for call_type, subset in model_calls_df.groupby("call_type"):
        by_call_type[str(call_type)] = {
            "estimated_cost_usd": float(subset["est_cost_usd"].sum()),
            "estimated_input_tokens": int(subset["est_input_tokens"].sum()),
            "estimated_output_tokens": int(subset["est_output_tokens"].sum()),
            "model_call_count": int(len(subset)),
        }

    by_dataset_split = {}
    for dataset_split, subset in model_calls_df.groupby("dataset_split"):
        by_dataset_split[str(dataset_split)] = {
            "estimated_cost_usd": float(subset["est_cost_usd"].sum()),
            "estimated_input_tokens": int(subset["est_input_tokens"].sum()),
            "estimated_output_tokens": int(subset["est_output_tokens"].sum()),
            "model_call_count": int(len(subset)),
        }

    return {
        "overall": overall,
        "by_call_type": by_call_type,
        "by_dataset_split": by_dataset_split,
    }


def build_strategy_note(paraphraser_strategy: str) -> str:
    if paraphraser_strategy == "intent_guarded":
        return "Uses the intent_guarded paraphraser tuned to preserve benign intent while surfacing harmful intent."
    if paraphraser_strategy == "suspicious_intent_guarded":
        return "Uses suspicion-triggered routing: a strict guarded paraphraser for suspicious prompts and a gentler paraphraser for prompts classified as benign."
    return "Uses the baseline paraphraser strategy."


def main():
    args = parse_args()

    if not GEMINI_API_KEY:
        print("ERROR: GEMINI_API_KEY missing.")
        sys.exit(1)
    if not DEEPSEEK_API_KEY:
        print("ERROR: DEEPSEEK_API_KEY missing.")
        sys.exit(1)

    styles = [args.style] if args.style else ATTACK_STYLES
    run_id = build_run_id(DEFENSE_NAME, args.run_name)
    run_dir = ensure_output_dir(args.output_dir, run_id)

    if args.attack_run_dir:
        print("Resolving external attack run...")
        snapshot = resolve_attack_run_snapshot(args.attack_run_dir, requested_styles=styles)
    else:
        print("Resolving benchmark snapshot...")
        snapshot = resolve_benchmark_snapshot(args.benchmark_id, requested_styles=styles)
    print(f"Benchmark ready: {snapshot.benchmark_id} ({', '.join(snapshot.styles)})")

    print("Initialising API clients...")
    models = build_models(args.paraphraser_strategy)
    print("Clients ready.")

    attack_rows = load_attack_results(snapshot)
    attack_rows_available = int(len(attack_rows))
    if args.limit is not None:
        attack_rows = attack_rows.head(args.limit).copy()

    attack_eval_df, attack_model_calls_df = evaluate_attack_rows(
        attack_rows,
        models,
        run_id=run_id,
        defense_name=DEFENSE_NAME,
    )

    benign_eval_df = None
    benign_model_calls_df = pd.DataFrame()
    benign_cache_path = Path("benign_creative_cache.csv")
    if not args.skip_fpr:
        benign_rows = load_or_generate_benign_prompts(
            models,
            styles,
            benign_cache_path,
            regen=args.regen_benign_cache,
        )
        benign_rows_available = int(len(benign_rows))
        if args.limit is not None:
            benign_rows = benign_rows.head(args.limit).copy()
        benign_eval_df, benign_model_calls_df = evaluate_benign_rows(
            benign_rows,
            models,
            run_id=run_id,
            defense_name=DEFENSE_NAME,
            benchmark_id=snapshot.benchmark_id,
        )
    else:
        benign_rows_available = 0

    model_calls_frames = [frame for frame in [attack_model_calls_df, benign_model_calls_df] if not frame.empty]
    model_calls_df = pd.concat(model_calls_frames, ignore_index=True) if model_calls_frames else pd.DataFrame()

    summary_df = build_summary_metrics(attack_eval_df, benign_eval_df, model_calls_df)

    attack_path = run_dir / f"attack_{DEFENSE_NAME}.csv"
    attack_eval_df.to_csv(attack_path, index=False)

    benign_path = run_dir / f"benign_{DEFENSE_NAME}.csv"
    if benign_eval_df is not None:
        benign_eval_df.to_csv(benign_path, index=False)

    model_calls_path = run_dir / "model_calls.csv"
    if not model_calls_df.empty:
        model_calls_df.to_csv(model_calls_path, index=False)

    summary_path = run_dir / "summary.csv"
    summary_df.to_csv(summary_path, index=False)

    cost_summary = summarize_costs(model_calls_df)

    benchmark_file_paths = [str(snapshot.csv_paths[style]) for style in snapshot.styles]
    benchmark_file_hashes = {
        str(path): snapshot.file_hashes[str(path)] for path in snapshot.csv_paths.values()
    }

    manifest = {
        "run_id": run_id,
        "timestamp_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "defense": DEFENSE_NAME,
        "paraphraser_strategy": args.paraphraser_strategy,
        "benchmark_id": snapshot.benchmark_id,
        "benchmark_name": snapshot.benchmark_name,
        "benchmark_version": snapshot.benchmark_version,
        "benchmark_description": snapshot.benchmark_description,
        "benchmark_manifest_path": str(snapshot.manifest_path),
        "benchmark_manifest_sha256": snapshot.manifest_sha256,
        "benchmark_csv_combined_sha256": snapshot.csv_combined_sha256,
        "benchmark_snapshot_sha256": snapshot.snapshot_sha256,
        "attack_input_source_type": snapshot.source_type,
        "attack_input_source_path": str(
            snapshot.source_reference_path if snapshot.source_reference_path is not None else snapshot.snapshot_dir
        ),
        "attack_input_metadata_path": None
        if snapshot.source_metadata_path is None
        else str(snapshot.source_metadata_path),
        "benchmark_file_paths": benchmark_file_paths,
        "benchmark_file_hashes": benchmark_file_hashes,
        "benchmark_created_from": snapshot.created_from,
        "benchmark_notes": snapshot.notes,
        "available_styles": snapshot.available_styles,
        "requested_styles": snapshot.styles,
        "benchmark_categories": snapshot.categories,
        "benchmark_expected_total_rows": snapshot.expected_total_rows,
        "benchmark_expected_selected_rows": snapshot.expected_selected_rows,
        "benchmark_validated_total_rows": snapshot.validated_total_rows,
        "attack_rows_available": attack_rows_available,
        "attack_rows_evaluated": int(len(attack_eval_df)),
        "evaluated_categories": sorted(
            attack_eval_df["category"].dropna().astype(str).unique().tolist()
        ),
        "benign_rows_available": benign_rows_available,
        "benign_rows_evaluated": 0 if benign_eval_df is None else int(len(benign_eval_df)),
        "benign_topics": []
        if benign_eval_df is None
        else sorted(benign_eval_df["topic"].dropna().astype(str).unique().tolist()),
        "models": {
            "target": models["target_model_name"],
            "paraphraser": models["paraphraser_model_name"],
            "rewriter": models["rewriter_model_name"],
            "judge": models["judge_model_name"],
            "refusal_judge": models["refusal_judge_model_name"],
            "utility_judge": models["utility_judge_model_name"],
            "suspicion_detector": models["suspicion_detector_model_name"],
        },
        "cost_summary": cost_summary,
        "pricing_estimation": {
            "token_estimation_method": TOKEN_ESTIMATION_METHOD,
            "pricing_source": PRICING_SOURCE,
            "notes": [
                "Token counts are estimated consistently from character length rather than live API usage metadata.",
                "Estimated costs should be treated as approximate and mainly useful for comparing runs and call types within this repo.",
            ],
        },
        "outputs": {
            "attack_csv": str(attack_path),
            "benign_csv": None if benign_eval_df is None else str(benign_path),
            "summary_csv": str(summary_path),
            "model_calls_csv": None if model_calls_df.empty else str(model_calls_path),
        },
        "notes": [
            build_strategy_note(args.paraphraser_strategy),
            "Defense evaluation reads a named benchmark snapshot under benchmarks/ rather than ambient root-level results files.",
            "Benign intent retention is judged with PRESERVED/PARTIAL/DISTORTED labels.",
        ],
    }
    if snapshot.source_type == "attack_history_run":
        manifest["notes"].append(
            "Attack inputs were imported directly from an external attack_history run directory, so defense was inserted after the frozen attack rewriter stage."
        )
    if args.limit is not None:
        manifest["notes"].append(
            f"This run used --limit {args.limit}, so evaluated rows are a subset of the selected benchmark snapshot."
        )
    if args.skip_fpr:
        manifest["notes"].append("Benign evaluation was skipped via --skip-fpr.")
    write_run_manifest(run_dir / "run_manifest.json", manifest)

    print()
    print(f"Run id:               {run_id}")
    print(f"Benchmark id:         {snapshot.benchmark_id}")
    print(f"Paraphraser strategy: {args.paraphraser_strategy}")
    print(f"Attack rows:          {len(attack_eval_df)}")
    print(f"Benign rows:          {0 if benign_eval_df is None else len(benign_eval_df)}")
    print(
        f"Baseline ASR:         {format_rate(resolve_overall_rate(summary_df, 'attack', 'baseline_asr'))}"
    )
    print(
        f"Defended ASR:         {format_rate(resolve_overall_rate(summary_df, 'attack', 'defended_asr'))}"
    )
    print(
        f"ASR reduction:        {format_rate(resolve_overall_rate(summary_df, 'attack', 'asr_reduction'))}"
    )
    print(
        "Survival rate:        "
        f"{format_rate(resolve_overall_rate(summary_df, 'attack', 'survival_rate_on_baseline_unsafe_subset'))}"
    )
    if benign_eval_df is not None:
        print(f"FPR:                  {format_rate(resolve_overall_rate(summary_df, 'benign', 'fpr'))}")
        print(
            "Intent preserved:     "
            f"{format_rate(resolve_overall_rate(summary_df, 'benign', 'intent_preservation_rate_strict'))}"
        )
    print(
        f"Estimated cost:       {format_currency(resolve_overall_value(summary_df, 'cost', 'estimated_cost_usd'))}"
    )
    print(
        f"Model calls:          {format_count(resolve_overall_value(summary_df, 'cost', 'model_call_count'))}"
    )
    print(f"Outputs:              {run_dir}")


if __name__ == "__main__":
    main()
