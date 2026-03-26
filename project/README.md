# CS427 AI Safety Project

## What The Attack Pipeline Is

`attack_pipeline.py` is an automated jailbreak evaluation workflow for AI safety experiments.

For each seed harmful request category, it:

1. Uses a rewriter model (DeepSeek via OpenAI-compatible API) to convert the seed into a selected creative style (`riddle`, `poem`, `nursery_rhyme`, `dialogue`).
2. Sends the rewritten prompt to a target model (Gemini) and collects the response.
3. Uses a judge model (Gemini) to label the response as `SAFE` or `UNSAFE`.
4. If a style/category repeatedly fails to elicit unsafe output, uses an improver model (Gemini) to mutate the style instruction and retries.

The script saves per-style run results to `results_*.csv` and persists improved style instructions to `improved_instructions.json` so runs can resume from prior mutations.

## Setup

### 1. Go to project folder
```powershell
cd project
```

### 2. Create and activate a virtual environment
```powershell
python -m venv .venv
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

## Outputs

- `results_*.csv`: results per style run
- `improved_instructions.json`: mutated/improved style instructions saved across runs

## Recent Updates

- Refactored judge output from binary-only labels to a structured rubric.
  - `judge_label`, `judge_label_text`
  - `compliance_level` (0-3)
  - `actionability` (0-3)
  - `severity` (`low|medium|high|critical`)
  - `confidence` (0.0-1.0)
- Added intent-retention scoring between `seed_prompt` and `rewritten_prompt` for debugging/attribution.
- Added end-to-end intent-fulfillment scoring between `seed_prompt` and final `response`.
- Added richer summary reporting and CSV schema validation checks.
- Added configurable model-stack flags for rewriter, judge backend, and intent backend.
- Added configurable backend/model flags for fulfillment evaluator.
- Added extraction of Gemini target safety ratings (`target_safety_score`, `target_safety_ratings`).

## Todo
- [x] add end-to-end intent fulfillment metric (`seed_prompt` -> `response`) as primary outcome
- [ ] keep rewrite intent retention (`seed_prompt` -> `rewritten_prompt`) as debugging/attribution metric
  - low rewrite retention + low end-to-end fulfillment -> rewriter quality issue
  - high rewrite retention + low end-to-end fulfillment -> stronger target guardrails
  - high rewrite retention + high end-to-end fulfillment -> effective jailbreak path
  - low rewrite retention + high end-to-end fulfillment -> accidental but valid bypass path
- [ ] run baseline with the updated judge schema
- [ ] explore using
  - quen3 next 80b a3b as rewriter
  - llama guard 12B for classification, gemini safety ratings for numeric scorer
