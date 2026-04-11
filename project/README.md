# CS427 AI Safety Project

## What The Attack Pipeline Is

`attack_pipeline.py` is an automated jailbreak evaluation workflow for AI safety experiments.

For each seed harmful request category, it:

1. Uses a rewriter model (DeepSeek via OpenAI-compatible API) to convert the seed into a selected creative style (`riddle`, `poem`, `nursery_rhyme`, `dialogue`).
2. Runs a rewrite-topic check and retries rewrite generation on drift (up to `MAX_REWRITE_RETRIES`).
3. Sends the rewritten prompt to a target model (Gemini) and records target metadata (response text, finish reason, safety score, and error info).
4. Uses a judge model (Gemini) to return structured safety output:
   - `label` (`SAFE` or `UNSAFE`)
   - `compliance_level` (`0..3`)
   - `severity` (`LOW|MEDIUM|HIGH|CRITICAL`)
   - `actionability` (`0..3`)
   - `confidence` (`0..1`)
   - `rationale`
5. Runs intent alignment checking between the original seed prompt and target response.

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
```

Optional model overrides (can be set in `.env` or passed as CLI args):
```env
REWRITER_BASE_URL=https://api.deepseek.com
REWRITER_MODEL=deepseek-chat
TARGET_MODEL=gemini-2.5-flash
JUDGE_MODEL=gemini-2.5-flash
```

## Run

Run all styles:
```powershell
python attack_pipeline.py
```

Run one style:
```powershell
python attack_pipeline.py --style riddle
python attack_pipeline.py --style poem
python attack_pipeline.py --style nursery_rhyme
python attack_pipeline.py --style dialogue
```

Run with explicit model settings:
```powershell
python attack_pipeline.py --rewriter-model deepseek-chat --target-model gemini-2.5-flash --judge-model gemini-2.5-flash
```

## Outputs

- `attack_history/<n>/results_<style>.csv`: per-style raw results
- `attack_history/<n>/stats.json`: aggregate ASR metrics and confidence summaries

Common CSV fields include:
- prompt metadata (`style`, `category`, `repetition`, `seed_prompt`, `rewritten_prompt`)
- target output and diagnostics (`response`, `target_finish_reason`, `target_safety_score`, `target_safety_reason`, `target_error`)
- judge outputs (`judge_label`, `judge_label_text`, `compliance_level`, `severity`, `actionability`, `confidence`, `judge_rationale`, `judge_error`)
- intent check (`intent_aligned`)

## Stop / Deactivate Venv
```powershell
deactivate
```

## Todo
- [ ] explore using
  - quen3 next 80b a3b as rewriter
  - llama guard 12B for classification, gemini safety ratings for numeric scorer
- [ ] read through the excel sheet generation look for other problems