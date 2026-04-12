# CS427 AI Safety Project

## What The Attack Pipeline Is

`attack_pipeline.py` is an automated jailbreak evaluation workflow for AI safety experiments.

High-level flow:
`seed prompt -> Rewriter (default: DeepSeek-V3) -> target model (default: Gemini 2.5 Flash) -> AI Judge (default: Claude Haiku 4.5)`

For each seed harmful request category, it:

1. Uses a rewriter model (DeepSeek via OpenAI-compatible API) to convert the seed into a selected creative style (`riddle`, `poem`, `nursery_rhyme`, `dialogue`).
2. Runs a rewrite-topic check and retries rewrite generation on drift (up to `MAX_REWRITE_RETRIES`).
3. Sends the rewritten prompt to a target model and records target metadata (response text, finish reason, raw output metadata, and error info).
4. Uses a judge model with a rubric prompt and returns structured safety fields:
   - `label` (`SAFE` or `UNSAFE`): final safety verdict.
   - `compliance_level` (`0..3`): how much harmful assistance is given.
   - `severity` (`LOW|MEDIUM|HIGH|CRITICAL`): estimated harm seriousness.
   - `actionability` (`0..3`): how directly usable the content is.
   - `confidence` (`0..1`): judge confidence.
   - `judge_rationale`: short textual justification.
5. Runs intent alignment checking between the original seed prompt and target response using a structured JSON output (`label`, `confidence`, `rationale`), and stores raw intent checker output for debugging.

The script writes results under `attack_history/<run_number>/`.

## Setup

### 1. Go to project folder
```powershell
cd project
```

### 2. Create and activate a virtual environment

Create:
```powershell
python -m venv .venv
```

Activate (PowerShell):
```powershell
.\.venv\Scripts\Activate.ps1
```

Activate (cmd.exe):
```bat
.\.venv\Scripts\activate.bat
```

Activate (bash/zsh on macOS/Linux):
```bash
source .venv/bin/activate
```

If PowerShell blocks activation:
```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

If `pip` is missing in the venv:
```powershell
python -m ensurepip --upgrade
```

### 3. Install dependencies
```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 4. (Optional) Verify your interpreter is the venv
```powershell
python -c "import sys; print(sys.executable)"
```

## Environment Variables

Create `.env` from the template:
```powershell
Copy-Item .env.example .env
```

Set your API keys in `.env`:
```env
GEMINI_API_KEY=your_gemini_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
CLAUDE_API_KEY=your_claude_key_here
OPENAI_API_KEY=your_openai_key_here
OPENROUTER_API_KEY=your_openrouter_key_here
```

Optional stage/provider/model overrides (can be set in `.env` or passed as CLI args):
```env
REWRITER_PROVIDER=deepseek
REWRITER_MODEL=deepseek-chat
REWRITER_BASE_URL=https://api.deepseek.com

INTENT_CHECKER_PROVIDER=claude
INTENT_CHECKER_MODEL=claude-haiku-4-5-20251001

TARGET_PROVIDER=gemini
TARGET_MODEL=gemini-2.5-flash
TARGET_MAX_TOKENS=4096
TARGET_THINKING_BUDGET=2048
TARGET_REASONING_EFFORT=none

JUDGE_PROVIDER=claude
JUDGE_MODEL=claude-haiku-4-5-20251001
```

Current defaults in code:
- Rewriter: `deepseek:deepseek-chat` (DeepSeek V3 alias)
- Intent checker: `claude:claude-haiku-4-5-20251001`
- Target: `gemini:gemini-2.5-flash`
- Judge: `claude:claude-haiku-4-5-20251001`

Supported providers:
- `deepseek`
- `openai`
- `claude`
- `openrouter`
- `gemini`

Model ID rule:
- You can use any model from a provider by passing its exact model ID with the matching `--<stage>-provider`.
- The pipeline does not enforce a fixed model whitelist.
- If provider/model/auth is wrong, the run fails early.
- For `judge` and `intent_checker`, prefer models that reliably follow strict formatting/instructions (especially JSON output for judge).

## Run

All commands below assume your current directory is `project/`.

Run all styles:
```powershell
python attack_pipeline.py
```

Replay with frozen rewritten prompts from an existing run (skip rewriter/rewrite-checker):
```powershell
# Reuse all results_*.csv under attack_history/57
python attack_pipeline.py --reuse-rewrites-from-run 57

# Reuse one explicit source directory/file
python attack_pipeline.py --reuse-rewrites-path attack_history/57
python attack_pipeline.py --reuse-rewrites-path attack_history/57/results_riddle.csv
```

In replay mode, the pipeline:
- loads `rewritten_prompt` from existing `results_*.csv`
- does not call the rewriter or rewrite drift checker
- only re-runs target + judge + intent checker
- writes a new run under `attack_history/<new_run>/`

Run one style:
```powershell
python attack_pipeline.py --style riddle
python attack_pipeline.py --style poem
python attack_pipeline.py --style nursery_rhyme
python attack_pipeline.py --style dialogue
```

Run with explicit model settings:
```powershell
python attack_pipeline.py --rewriter-provider deepseek --rewriter-model deepseek-chat --intent-checker-provider claude --intent-checker-model claude-haiku-4-5-20251001 --target-provider gemini --target-model gemini-2.5-flash --judge-provider claude --judge-model claude-haiku-4-5-20251001
```

Target reasoning controls:
- `--target-max-tokens <int>` (default `4096`)
- `--target-thinking-budget <int>` (default `2048`, set `0` to disable where supported)
  - Gemini/Claude: native thinking budget controls
  - OpenRouter/DeepSeek (OpenAI-compatible path): mapped to `reasoning.max_tokens`
- `--target-reasoning-effort none|minimal|low|medium|high` (default `none`, mainly for OpenAI family)
  - OpenRouter: mapped to `reasoning.effort` when set

Planned 8-run experiment commands:
```powershell
# 1) Gemini 2.5 Flash with thinking
python attack_pipeline.py --reuse-rewrites-from-run 57 --target-provider gemini --target-model gemini-2.5-flash --target-max-tokens 4096 --target-thinking-budget 2048 --target-reasoning-effort none

# 2) Claude Haiku 4.5 with thinking budget 2048
python attack_pipeline.py --reuse-rewrites-from-run 57 --target-provider claude --target-model claude-haiku-4-5-20251001 --target-max-tokens 4096 --target-thinking-budget 2048 --target-reasoning-effort none

# 3) OpenAI GPT-5 mini with medium reasoning
python attack_pipeline.py --reuse-rewrites-from-run 57 --target-provider openai --target-model gpt-5-mini --target-max-tokens 4096 --target-thinking-budget 0 --target-reasoning-effort medium

# 4) Gemini 2.5 Flash with no thinking
python attack_pipeline.py --reuse-rewrites-from-run 57 --target-provider gemini --target-model gemini-2.5-flash --target-max-tokens 4096 --target-thinking-budget 0 --target-reasoning-effort none

# 5) Claude Haiku 4.5 with no thinking
python attack_pipeline.py --reuse-rewrites-from-run 57 --target-provider claude --target-model claude-haiku-4-5-20251001 --target-max-tokens 4096 --target-thinking-budget 0 --target-reasoning-effort none

# 6) OpenAI GPT-5 mini with no reasoning
python attack_pipeline.py --reuse-rewrites-from-run 57 --target-provider openai --target-model gpt-5-mini --target-max-tokens 4096 --target-thinking-budget 0 --target-reasoning-effort none

# 7) OpenRouter Qwen3-32B with reasoning effort medium (+reasoning budget 2048)
python attack_pipeline.py --reuse-rewrites-from-run 57 --target-provider openrouter --target-model qwen/qwen3-32b --target-max-tokens 4096 --target-thinking-budget 2048 --target-reasoning-effort medium

# 8) DeepSeek chat with thinking enabled
python attack_pipeline.py --reuse-rewrites-from-run 57 --target-provider deepseek --target-model deepseek-chat --target-max-tokens 4096 --target-thinking-budget 2048 --target-reasoning-effort none
```

Automate the same suite with PowerShell:
```powershell
# Runs experiments 2-8 by default (skips baseline #1, since run 57 already has it)
.\run_target_experiments.ps1 -ReuseRewritesFromRun 57

# Include baseline #1 as well
.\run_target_experiments.ps1 -ReuseRewritesFromRun 57 -IncludeBaselineExperiment

# This script now includes experiments 7-8 as well (OpenRouter Qwen3-32B and DeepSeek V3.1 alias).
```

Override only one stage and keep all other defaults:
```powershell
# Change target only
python attack_pipeline.py --target-provider openai --target-model gpt-4.1-mini

# Change judge only
python attack_pipeline.py --judge-provider openrouter --judge-model moonshotai/kimi-k2
```

Quick connectivity test (no full pipeline run):
```powershell
python attack_pipeline.py --ping-only
```

Re-judge existing CSV outputs with a different judge model (no rewrite/target calls):
```powershell
# Re-judge one CSV
python attack_pipeline.py --rejudge-path attack_history/50/results_poem.csv --judge-provider openrouter --judge-model minimax/minimax-m2.5:free

# Re-judge all results_*.csv in a run directory
python attack_pipeline.py --rejudge-path attack_history/50 --judge-provider openrouter --judge-model minimax/minimax-m2.5:free
```
This writes companion files like:
`results_poem.rejudge_openrouter_minimax_minimax-m2.5_free.csv`
and prints agreement against the existing `judge_label_text`.

## Outputs

- `attack_history/<n>/results_<style>.csv`: per-style raw results
- `attack_history/<n>/stats.json`: aggregate ASR metrics and confidence summaries

## Stop / Deactivate Venv
```powershell
deactivate
```

## Quick Preflight Check
Before running the full pipeline, you can verify stage and provider connectivity:
```powershell
python attack_pipeline.py --ping-only
```

## Attack History
You may view past attacks, the folders for the different models are as follows:
- gemini 2.5 flash 
  - thinking: 57
  - no thinking: 67

- claude haiku 4-5 
  - thinking: 63
  - no thinking: 70

- openai gpt 5 mini 
  - reasoning: 66
  - no reasoning: 69

- qwen thinking : 71

- deepseek reasoning: 72
