# Defense Run Comparison

These results compare only the clean full upgraded-framework runs on `proposal_v1`.

## Main Table

| Strategy | Baseline ASR | Defended ASR | ASR Reduction | Survival Rate | FPR | Intent Strict | Intent Relaxed | Distortion | Est. Cost | Attack Rows | Benign Rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | 92.5% | 22.8% | 69.7% | 24.3% | 6.2% | 37.5% | 43.8% | 56.2% | $0.141967 | 80 | 16 |
| intent_guarded | 92.5% | 48.8% | 43.8% | 51.4% | 6.2% | 56.2% | 75.0% | 25.0% | $0.209435 | 80 | 16 |
| suspicious_intent_guarded | 92.5% | 1.2% | 91.2% | 1.4% | 0.0% | 62.5% | 68.8% | 31.2% | $0.091001 | 80 | 16 |

## Quick Read

- Strongest safety blocking: `suspicious_intent_guarded`
- Strongest benign utility: `suspicious_intent_guarded`
- Lowest estimated cost: `suspicious_intent_guarded`

## Cost Breakdown

| Strategy | Paraphraser | Suspicion Detector | Target | Judge | Refusal Judge | Utility Judge | Benign Rewriter |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| baseline | $0.014674 | n/a | $0.088116 | $0.028268 | $0.004781 | $0.000999 | $0.005130 |
| intent_guarded | $0.017995 | n/a | $0.136282 | $0.042001 | $0.007006 | $0.001021 | $0.005130 |
| suspicious_intent_guarded | $0.014853 | $0.015168 | $0.038437 | $0.008720 | $0.007634 | $0.001061 | $0.005130 |
