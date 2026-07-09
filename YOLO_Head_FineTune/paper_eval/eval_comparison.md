# Per-domain model comparison

Recall/Precision/F1 at operating confidence 0.25, imgsz 960. mAP is threshold-independent. Recall is the headline metric (a missed head is a privacy leak). Pooled 'ALL' is reference only.

## Recall by domain (headline)

| Domain | Heads | head_n_old | head_n_new |
|---|---|---|---|
| (other) | 1120 | 0.763 | 0.763 |
| ALL _(reference only)_ | 1120 | 0.763 | 0.763 |

## Full metrics

| Model | Domain | Images | Heads | Recall | Precision | F1 | mAP50 | mAP50-95 |
|---|---|---|---|---|---|---|---|---|
| head_n_old | (other) | 508 | 1120 | 0.763 | 0.885 | 0.820 | 0.826 | 0.417 |
| head_n_old | ALL | 508 | 1120 | 0.763 | 0.885 | 0.820 | 0.826 | 0.417 |
| head_n_new | (other) | 508 | 1120 | 0.763 | 0.885 | 0.820 | 0.826 | 0.417 |
| head_n_new | ALL | 508 | 1120 | 0.763 | 0.885 | 0.820 | 0.826 | 0.417 |
