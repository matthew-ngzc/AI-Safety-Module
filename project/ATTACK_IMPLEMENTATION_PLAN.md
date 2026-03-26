# Attack Pipeline Implementation Plan

## Why these changes

The current attack pipeline reports only binary `SAFE/UNSAFE`, which is useful for ASR but too coarse to analyze partial leaks, actionability, and judge certainty.

These changes make attack evaluation more diagnostic before we move into model swap experiments or defense integration.

## Ordered implementation

1. Implement structured judge output (primary priority).
   - Replace binary-only judge output with a schema that includes:
     - `judge_label` (`SAFE` / `UNSAFE`)
     - `compliance_level` (0-3)
     - `actionability` (0-3)
     - `confidence` (0.0-1.0)
     - optional `severity` (low/medium/high/critical)
   - Reason: this upgrades evaluation signal quality with minimal architecture changes.

2. Add rubric fields to CSV + summary reporting.
   - Persist new fields per row.
   - Extend aggregate reporting beyond ASR to include means/distributions for compliance and actionability.
   - Reason: metrics are only useful if consistently logged and summarized.

3. Measure intent retention (attack-phase version).
   - Use a paired check with known inputs: compare `seed_prompt` vs `rewritten_prompt`.
   - Output retention score/label (for example: `retained`, `partially_retained`, `not_retained` or numeric 0-1).
   - Reason: in attack phase, we control both seed and transformed prompts, so this is the cleanest and most reliable test of whether rewriting preserved malicious intent.

4. CSV/excel generation sanity pass.
   - Validate column names, type consistency, null handling, and backward compatibility in analysis notebooks.
   - Reason: avoid downstream analysis errors before larger experiments.

5. Run baseline with new schema.
   - Re-run current model stack and establish new baseline tables.
   - Reason: create a stable before/after reference point.

6. Explore alternative model stack.
   - Rewriter: `qwen3-next-80b-a3b`.
   - Classification/scoring: `Llama Guard 12B` + Gemini safety ratings numeric scorer.
   - Reason: model comparison should happen only after metrics are stable.

## Intent retention decision

Short answer: yes, this should be done now, but as a smaller scoped attack-phase problem.

There are two possible formulations:

1. Harder defense-style formulation:
   - infer harmful intent from transformed prompt alone.
   - this is closer to production defense behavior.

2. Attack-phase paired formulation (recommended now):
   - compare transformed prompt against known seed intent.
   - this directly measures whether rewriting preserved or diluted intent.

For this stage, use the paired formulation because:

- the seed prompt is known and controlled by the experiment;
- it gives higher signal and lower variance;
- it isolates attack quality without conflating defense inference difficulty.

Later, during defense evaluation, add the transformed-only inference variant as a separate metric to test robustness under realistic uncertainty.
