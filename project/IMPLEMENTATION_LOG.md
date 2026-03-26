# Attack Pipeline Implementation Log

Date: 2026-03-23

## Scope completed

Implemented the planned attack-phase changes in `attack_pipeline.py`:

1. Structured judge output.
   - Judge now returns: `judge_label`, `judge_label_text`, `compliance_level`, `actionability`, `severity`, `confidence`, `judge_error`.
   - Added robust JSON parsing and fallback handling.

2. Rubric fields in CSV and summary reporting.
   - Added new per-row CSV columns for judge rubric fields.
   - Added summary outputs for:
     - mean compliance level
     - mean actionability
     - mean confidence
     - severity distribution

3. Intent-retention measurement (paired seed vs transformed prompt).
   - Added evaluator returning:
     - `intent_retention_label`
     - `intent_retention_score`
     - `intent_retention_rationale`
   - This is computed per attack attempt and persisted to CSV.

4. CSV schema sanity check.
   - Added a required-column validation step before writing output.

5. Baseline/ablation-enabling configurability.
   - Rewriter is now configurable via CLI/env:
     - `--rewriter-base-url`
     - `--rewriter-model`
     - `--rewriter-api-key`
   - Judge backend can be switched:
     - `--judge-backend gemini`
     - `--judge-backend llama_guard` (OpenAI-compatible endpoint)
   - Intent backend can be switched:
     - `--intent-backend gemini`
     - `--intent-backend openai_compatible`
   - Gemini target safety ratings are extracted to:
     - `target_safety_score`
     - `target_safety_ratings`

## Validation done

- Syntax check passed:
  - `python -m py_compile attack_pipeline.py`
- CLI check passed in project venv:
  - `.\.venv\Scripts\python.exe attack_pipeline.py --help`

## Baseline run commands

Run default baseline (DeepSeek rewriter + Gemini judge + Gemini intent evaluator):

```powershell
.\.venv\Scripts\python.exe attack_pipeline.py
```

Run one style:

```powershell
.\.venv\Scripts\python.exe attack_pipeline.py --style riddle
```

## Alternative stack example

Example with Qwen rewriter + Llama Guard judge (OpenAI-compatible endpoint) while keeping Gemini target scoring:

```powershell
.\.venv\Scripts\python.exe attack_pipeline.py `
  --rewriter-base-url https://<openai-compatible-provider> `
  --rewriter-model qwen3-next-80b-a3b `
  --rewriter-api-key <rewriter_key> `
  --judge-backend llama_guard `
  --judge-openai-base-url https://<openai-compatible-provider> `
  --judge-openai-model meta-llama/Llama-Guard-3-12B `
  --judge-openai-api-key <judge_key>
```

## Notes

- Existing `results_*.csv` files from older schema will not contain the new columns.
- The new schema applies to new runs only.
