# Experiment comparison

Validation metrics describe the checkpoint selected by each run. No test evidence was read and no winning configuration was selected.

| run_id | initialization | seed | topk_weight | steps | best_validation_cross_entropy | best_validation_token_agreement | observed_completion_tokens | elapsed_seconds |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| run-866fb1a604bea2b1 | random | 42 | 0 | 64 | 0.710612 | 0.85 | 112 | 0.556954 |
| run-ee711bca75f68fa9 | random | 42 | 1 | 64 | 3.90535 | 0.475 | 112 | 0.606947 |

Token counts are recorded evidence counts, not optimization exposures. Unknown counts are shown as —.
