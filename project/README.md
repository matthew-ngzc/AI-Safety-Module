# CS427 AI Safety Project

This folder contains the combined attack and defense work for our CS427 project on creative linguistic jailbreak attacks and paraphrase-based defense for LLMs.

The project has two connected parts:

- the attack pipeline at the `project/` root
- the defense pipeline inside `project/defense/`

The current linked evaluation pipeline is:

`seed prompt -> rewriter -> defense -> target model -> judge`

In the real integrated setup, the defense is inserted after the frozen attack rewriter stage using prompts from `attack_history/57`.

## What This Folder Contains

- attack generation and attack-side evaluation code
- saved attack runs under `attack_history/`
- the defense pipeline and frozen benchmark snapshot under `defense/`
- kept real-attack defense outputs for the run-57 linkage
- one compact comparison artifact for the three defense strategies

## Repository Layout

- `attack_pipeline.py`: generates creative jailbreak prompts, queries the target model, and judges attack success
- `attack_history/`: saved attack runs and per-style CSV outputs
- `defense/`: defense code, benchmark snapshots, scripts, and kept defense outputs
- `run_target_experiments.ps1`: helper script for attack-side experiment execution
- `requirements.txt`: attack-side dependencies
- `.env.example`: environment template for the attack-side pipeline

Important files inside `defense/`:

- `defense_pipeline.py`: main defense runner
- `benchmarks.py`: benchmark loading, validation, and external attack-run loading
- `evaluation.py`: attack evaluation, benign evaluation, and summary generation
- `defense_methods.py`: paraphrasing strategies, target calls, judges, and suspicion routing
- `prompts.py`: prompt templates and label definitions
- `pricing.py`: estimated token and cost accounting
- `benchmarks/proposal_v1/`: frozen proposal-aligned benchmark snapshot
- `outputs/`: kept defense runs and the run-57 comparison CSV
- `scripts/compare_runs.py`: rebuilds comparison tables across completed runs when needed
- `.env.example`: defense-side environment template

## Attack Pipeline

The attack pipeline follows:

`seed prompt -> rewriter -> target model -> judge`

It produces creative jailbreak prompts in styles such as `riddle`, `poem`, `nursery_rhyme`, and `dialogue`, then stores results under `attack_history/<run_number>/`.

The frozen attack run currently used for defense linkage is:

- `attack_history/57/`

## Defense Pipeline

The defense pipeline evaluates paraphrase-based defenses on rewritten harmful prompts.

Supported defense strategies:

- `baseline`
- `intent_guarded`
- `suspicious_intent_guarded`

The defense code supports two input modes:

1. the frozen local benchmark snapshot in `defense/benchmarks/`
2. an external frozen attack run using `--attack-run-dir`

For the integrated project workflow, we use:

- `attack_history/57` as the frozen attack input source

This lets the defense run on actual rewritten attack prompts produced by the attack pipeline, not only on benchmark CSVs.

## Final Defense Benchmark

The frozen proposal-aligned benchmark snapshot is stored in:

- `defense/benchmarks/proposal_v1/`

It contains:

- 4 harm categories: `self_harm`, `torture`, `dangerous_weapons`, `tax_fraud`
- 4 creative styles: `riddle`, `poem`, `nursery_rhyme`, `dialogue`
- 5 repetitions per style-category pair
- 80 total attack rows

## Kept Defense Results

The `defense/outputs/` folder has been cleaned to keep only the real run-57 evaluation artifacts:

- `defense/outputs/20260412T154910Z_attack57_baseline_full`
- `defense/outputs/20260412T154910Z_attack57_intent_guarded_full`
- `defense/outputs/20260412T151904Z_attack57_suspicious_intent_guarded_full`
- `defense/outputs/attack57_real_attack_comparison.csv`

These are the main files to inspect for the current defense comparison.

## Setup

From the `project/` folder:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r defense/requirements.txt
```

Create environment files:

```powershell
Copy-Item .env.example .env
Copy-Item defense/.env.example defense/.env
```

Typical API keys used in this project:

- root `.env` for attack-side runs: `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`, `CLAUDE_API_KEY`, `OPENAI_API_KEY`, `OPENROUTER_API_KEY`
- `defense/.env` for defense-side runs: `GEMINI_API_KEY`, `DEEPSEEK_API_KEY`

## Key Commands

Run the attack pipeline from `project/`:

```powershell
python attack_pipeline.py
```

Run one attack style:

```powershell
python attack_pipeline.py --style riddle
python attack_pipeline.py --style poem
python attack_pipeline.py --style nursery_rhyme
python attack_pipeline.py --style dialogue
```

Run the defense pipeline on the frozen proposal benchmark:

```powershell
cd defense
python defense_pipeline.py
```

Run the defense pipeline on real attack outputs from run 57:

```powershell
cd defense
python defense_pipeline.py --attack-run-dir ..\attack_history\57 --paraphraser-strategy baseline
python defense_pipeline.py --attack-run-dir ..\attack_history\57 --paraphraser-strategy intent_guarded
python defense_pipeline.py --attack-run-dir ..\attack_history\57 --paraphraser-strategy suspicious_intent_guarded
```

Run a quick validation pass on the real attack linkage:

```powershell
cd defense
python defense_pipeline.py --attack-run-dir ..\attack_history\57 --paraphraser-strategy suspicious_intent_guarded --limit 2
```

Rebuild the real-attack defense comparison CSV if needed:

```powershell
cd defense
python scripts/compare_runs.py ^
  --run-dir outputs/20260412T154910Z_attack57_baseline_full ^
  --run-dir outputs/20260412T154910Z_attack57_intent_guarded_full ^
  --run-dir outputs/20260412T151904Z_attack57_suspicious_intent_guarded_full ^
  --output-csv outputs/attack57_real_attack_comparison.csv
```

## Source Of Truth

For the linked attack input:

- `attack_history/57/`

For the frozen defense benchmark definition:

- `defense/benchmarks/proposal_v1/`

For the defense comparison currently used in this cleaned repo:

- `defense/outputs/attack57_real_attack_comparison.csv`
- the three kept full run folders under `defense/outputs/`

Estimated token and cost values in the defense pipeline are approximate and intended for consistent within-project comparison.
